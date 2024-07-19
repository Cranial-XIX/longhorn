import torch
import torch.nn as nn
import triton 
import triton.language as tl

from einops import rearrange
from fla.ops.gla import fused_chunk_gla, chunk_gla, fused_recurrent_gla
from torch.nn import functional as F
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from typing import List, Optional, Tuple, Union


class GLAConfig(PretrainedConfig):
    model_type = "transformer"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        d_model=2048,
        n_head=8,
        n_layer=24,
        use_gk=True,
        use_gv=False,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=0,
        eos_token_id=0,
        context_length=2048,
        vocab_size=50432,
        tie_word_embeddings=True,
        load_from_llama=False,
        **kwargs,
    ):

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_head = n_head
        self.n_layer = n_layer
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.context_length = context_length
        self.use_cache = use_cache

        self.use_gk = use_gk
        self.use_gv = use_gv
        self.load_from_llama = load_from_llama

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class GatedLinearAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.n_head
        
        self.gate_fn = nn.functional.silu
        assert config.use_gk and not config.use_gv, "Only use_gk is supported for simplicity."

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim//2, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim//2, bias=False)
        self.k_gate =  nn.Sequential(nn.Linear(self.embed_dim, 16, bias=False), nn.Linear(16, self.embed_dim // 2))

        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.g_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.head_dim = self.embed_dim // self.num_heads
        self.key_dim = self.embed_dim // self.num_heads
        self.scaling = self.key_dim ** -0.5
        self.group_norm = nn.LayerNorm(self.head_dim, eps=1e-5, elementwise_affine=False)
        self.post_init()

    def post_init(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -2.5)
        if isinstance(self.k_gate, nn.Sequential):
            nn.init.xavier_uniform_(self.k_gate[0].weight, gain=2 ** -2.5)
            nn.init.xavier_uniform_(self.k_gate[1].weight, gain=2 ** -2.5)
        else:
            nn.init.xavier_uniform_(self.k_gate.weight, gain=2 ** -2.5)

    def forward(self, x, hidden_states=None):
        q = self.q_proj(x)
        k = self.k_proj(x) * self.scaling
        k_gate = self.k_gate(x)
        v = self.v_proj(x)
        g = self.g_proj(x)

        output, new_hidden_states = self.gated_linear_attention(q, k, v, k_gate, hidden_states=hidden_states)
        output = self.gate_fn(g) * output
        output = self.out_proj(output)
        return output, new_hidden_states

    def gated_linear_attention(self, q, k, v, gk, normalizer=16, hidden_states=None):
        q = rearrange(q, 'b l (h d) -> b h l d', h = self.num_heads).contiguous()
        k = rearrange(k, 'b l (h d) -> b h l d', h = self.num_heads).contiguous()
        v = rearrange(v, 'b l (h d) -> b h l d', h = self.num_heads).contiguous()
        gk = rearrange(gk, 'b l (h d) -> b h l d', h = self.num_heads).contiguous()
        gk = F.logsigmoid(gk) / normalizer

        #if self.training:
        o, new_hidden_states = fused_chunk_gla(q, k, v, gk, initial_state=hidden_states, output_final_state=True)
        # else:
        #     o = fused_recurrent_gla(q, k, v, gk)
        #     new_hidden_states = None

        o = self.group_norm(o)
        o = rearrange(o, 'b h l d -> b l (h d)')
        return o, new_hidden_states


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output


class GLABlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.d_model, eps=1e-5)
        
        self.attn = GatedLinearAttention(config)
        self.ln_2 = RMSNorm(config.d_model, eps=1e-5)

        mlp_ratio = 4
        multiple_of = 256
        mlp_hidden = int(config.d_model * mlp_ratio * 2 / 3)
        mlp_hidden = multiple_of * ((mlp_hidden + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(config.d_model, mlp_hidden, bias=False)
        self.w2 = nn.Linear(mlp_hidden, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, mlp_hidden, bias=False)
        self.mlp = lambda x: self.w2(F.silu(self.w1(x)) * self.w3(x))

        self.resid_dropout = nn.Dropout(config.resid_pdrop)
    

    def forward(self, x, hidden_states=None):
        # attention/rnn
        x_att, new_hidden_states = self.attn(self.ln_1(x), hidden_states)
        x = x + self.resid_dropout(x_att)
        # ffn
        x_mlp = self.mlp(self.ln_2(x))
        x = x + self.resid_dropout(x_mlp)
        return x


class GLAPreTrainedModel(PreTrainedModel):
    config_class = GLAConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["GLABlock"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)


class GLAModel(GLAPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.h = nn.ModuleList([GLABlock(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.d_model, eps=1e-5)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # todo: support more options
        assert attention_mask is None
        assert output_attentions is None
        assert output_hidden_states is None
        assert use_cache is None
        assert position_ids is None
        assert past_key_values is None
        assert return_dict is True

        # useful for inference
        attn_chunk_size = 128
        bz, seq_len = input_ids.shape
        if seq_len % attn_chunk_size != 0:
            pad_token_id = self.config.pad_token_id
            padding_length = (attn_chunk_size - (seq_len % attn_chunk_size)) % attn_chunk_size
            pad_mat = torch.empty(bz, padding_length, dtype=torch.long).fill_(pad_token_id).to(input_ids.device)
            input_ids = torch.cat([input_ids, pad_mat], dim=1)
        else:
            padding_length = None

        x = self.wte(input_ids)
        for block in self.h:
            x = block(x)

        x = self.ln_f(x)

        return BaseModelOutputWithPast(
            past_key_values=None,
            last_hidden_state=x,
        )


class GLA(GLAPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.transformer = GLAModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.n_layer = config.n_layer

        self.apply(self._init_weights)
        self._post_init()
        self.lm_head.weight = self.transformer.wte.weight

    def _init_weights(self, module):
        """general init strategy"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm) or isinstance(module, RMSNorm):
            if hasattr(module, "bias") and  module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            if hasattr(module, "weight") and module.weight is not None:
                torch.nn.init.ones_(module.weight)
    
    def _post_init(self):
        """custom init strategy"""
        for name, module in self.named_modules():
            if hasattr(module, "post_init"):
                module.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # TODO: support more options
        assert attention_mask is None
        assert output_attentions is None
        assert output_hidden_states is None
        assert use_cache is None
        assert position_ids is None
        assert past_key_values is None
        assert return_dict is True

        # only useful for inference
        attn_chunk_size = 128
        bz, seq_len = input_ids.shape
        if seq_len % attn_chunk_size != 0:
            pad_token_id = self.config.pad_token_id
            padding_length = (attn_chunk_size - (seq_len % attn_chunk_size)) % attn_chunk_size
            pad_mat = torch.empty(bz, padding_length, dtype=torch.long).fill_(pad_token_id).to(input_ids.device)
            input_ids = torch.cat([input_ids, pad_mat], dim=1)
        else:
            padding_length = None

        xmr_output = self.transformer(input_ids)
        logits = self.lm_head(xmr_output.last_hidden_state)

        if padding_length is not None:
            logits = logits[:, :-padding_length]
        
        assert logits.shape[1] == seq_len

        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-1)
        else:
            loss = None

        # output
        return logits, loss
