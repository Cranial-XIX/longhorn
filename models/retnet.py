# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairscale.nn import checkpoint_wrapper, wrap
from timm.models.layers import drop_path

try:
    from apex.normalization import FusedLayerNorm as LayerNorm
except ModuleNotFoundError:
    from torch.nn import LayerNorm


class RetNetConfig(object):
    def __init__(self, **kwargs):
        self.decoder_embed_dim = kwargs.pop("decoder_embed_dim", 768)
        self.decoder_value_embed_dim = kwargs.pop("decoder_value_embed_dim", 1280)
        self.decoder_retention_heads = kwargs.pop("decoder_retention_heads", 3)
        self.decoder_ffn_embed_dim = kwargs.pop("decoder_ffn_embed_dim", 1280)
        self.decoder_layers = kwargs.pop("decoder_layers", 12)
        self.decoder_normalize_before = kwargs.pop("decoder_normalize_before", True)
        self.activation_fn = kwargs.pop("activation_fn", "gelu")
        self.dropout = kwargs.pop("dropout", 0.0)
        self.drop_path_rate = kwargs.pop("drop_path_rate", 0.0)
        self.activation_dropout = kwargs.pop("activation_dropout", 0.0)
        self.no_scale_embedding = kwargs.pop("no_scale_embedding", True)
        self.layernorm_embedding = kwargs.pop("layernorm_embedding", False)
        self.moe_freq = kwargs.pop("moe_freq", 0)
        self.moe_top1_expert = kwargs.pop("moe_top1_expert", False)
        self.moe_expert_count = kwargs.pop("moe_expert_count", 0)
        self.moe_gating_use_fp32 = kwargs.pop("moe_gating_use_fp32", True)
        self.moe_eval_capacity_token_fraction = kwargs.pop(
            "moe_eval_capacity_token_fraction", 0.25
        )
        self.moe_second_expert_policy = kwargs.pop("moe_second_expert_policy", "random")
        self.moe_normalize_gate_prob_before_dropping = kwargs.pop(
            "moe_normalize_gate_prob_before_dropping", False
        )
        self.use_xmoe = kwargs.pop("use_xmoe", False)
        self.rel_pos_buckets = kwargs.pop("rel_pos_buckets", 0)
        self.max_rel_pos = kwargs.pop("max_rel_pos", 0)
        self.deepnorm = kwargs.pop("deepnorm", False)
        self.subln = kwargs.pop("subln", True)
        self.multiway = kwargs.pop("multiway", False)
        self.share_decoder_input_output_embed = kwargs.pop(
            "share_decoder_input_output_embed", True
        )
        self.max_target_positions = kwargs.pop("max_target_positions", 1024)
        self.no_output_layer = kwargs.pop("no_output_layer", False)
        self.layernorm_eps = kwargs.pop("layernorm_eps", 1e-6)
        # Blockwise
        self.chunkwise_recurrent = kwargs.pop("chunkwise_recurrent", False)
        self.recurrent_chunk_size = kwargs.pop("recurrent_chunk_size", 512)
        # Text
        self.vocab_size = kwargs.pop("vocab_size", -1)
        # Fairscale
        self.checkpoint_activations = kwargs.pop("checkpoint_activations", False)
        self.fsdp = kwargs.pop("fsdp", False)
        self.ddp_rank = kwargs.pop("ddp_rank", 0)
        self.xpos_rel_pos = kwargs.pop("xpos_rel_pos", False)
        self.xpos_scale_base = kwargs.pop("xpos_scale_base", 512)

        if self.deepnorm:
            self.decoder_normalize_before = False
            self.subln = False
        if self.subln:
            self.decoder_normalize_before = True
            self.deepnorm = False
        if self.use_xmoe:
            self.moe_normalize_gate_prob_before_dropping = True
            self.moe_second_expert_policy = "random"
            assert self.moe_freq > 0 and self.moe_expert_count > 0

    def override(self, args):
        for hp in self.__dict__.keys():
            if getattr(args, hp, None) is not None:
                self.__dict__[hp] = getattr(args, hp, None)


def fixed_pos_embedding(x):
    seq_len, dim = x.shape
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim) / dim))
    sinusoid_inp = torch.einsum(
        "i , j -> i j", torch.arange(0, seq_len, dtype=torch.float), inv_freq
    ).to(x)
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def rotate_every_two(x):
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\


def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m


def apply_rotary_pos_emb(x, sin, cos, scale=1):
    sin, cos = map(lambda t: duplicate_interleave(t * scale), (sin, cos))
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)


class XPOS(nn.Module):
    def __init__(self, head_dim, scale_base=512):
        super().__init__()
        self.head_dim = head_dim
        self.scale_base = scale_base
        self.register_buffer(
            "scale", (torch.arange(0, head_dim, 2) + 0.4 * head_dim) / (1.4 * head_dim)
        )

    def forward(self, x, offset=0, downscale=False):
        length = x.shape[1]
        min_pos = -(length + offset) // 2
        max_pos = length + offset + min_pos
        scale = (
            self.scale
            ** torch.arange(min_pos, max_pos, 1)
            .to(self.scale)
            .div(self.scale_base)[:, None]
        )
        sin, cos = fixed_pos_embedding(scale)

        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]

        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, sin, cos, scale)
        return x


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        args,
        embed_dim,
        num_heads,
        dropout=0.0,
        self_attention=False,
        encoder_decoder_attention=False,
        subln=False,
    ):
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        assert self.self_attention ^ self.encoder_decoder_attention

        self.k_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))
        self.v_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))
        self.q_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))
        self.out_proj = MultiwayWrapper(
            args, nn.Linear(embed_dim, embed_dim, bias=True)
        )
        self.inner_attn_ln = (
            MultiwayWrapper(args, LayerNorm(self.embed_dim, eps=args.layernorm_eps))
            if subln and self.self_attention
            else None
        )
        self.dropout_module = torch.nn.Dropout(dropout)
        self.xpos = (
            XPOS(self.head_dim, args.xpos_scale_base)
            if args.xpos_rel_pos and self.self_attention
            else None
        )

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query,
        key,
        value,
        incremental_state=None,
        key_padding_mask=None,
        attn_mask=None,
        rel_pos=None,
        is_first_step=False,
    ):
        bsz, tgt_len, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"

        key_bsz, src_len, _ = key.size()
        assert key_bsz == bsz, f"{query.size(), key.size()}"
        assert value is not None
        assert bsz, src_len == value.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q *= self.scaling

        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.reshape(bsz * self.num_heads, tgt_len, self.head_dim)
        k = k.reshape(bsz * self.num_heads, src_len, self.head_dim)
        v = v.reshape(bsz * self.num_heads, src_len, self.head_dim)

        if incremental_state is not None:
            if "prev_key" in incremental_state:
                prev_key = incremental_state["prev_key"].view(
                    bsz * self.num_heads, -1, self.head_dim
                )
                prev_value = incremental_state["prev_value"].view(
                    bsz * self.num_heads, -1, self.head_dim
                )
                k = torch.cat([prev_key, k], dim=1)
                v = torch.cat([prev_value, v], dim=1)
            incremental_state["prev_key"] = k.view(
                bsz, self.num_heads, -1, self.head_dim
            )
            incremental_state["prev_value"] = v.view(
                bsz, self.num_heads, -1, self.head_dim
            )
            src_len = k.size(1)

        if self.xpos is not None:
            if incremental_state is not None and not is_first_step:
                offset = src_len - 1
            else:
                offset = 0
            k = self.xpos(k, offset=0, downscale=True)
            q = self.xpos(q, offset=offset, downscale=False)

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_weights = torch.nan_to_num(attn_weights)
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if rel_pos is not None:
            rel_pos = rel_pos.view(attn_weights.size())
            attn_weights = attn_weights + rel_pos

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )
        attn_probs = self.dropout_module(attn_weights)

        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).reshape(tgt_len, bsz, embed_dim).transpose(0, 1)

        if self.inner_attn_ln is not None:
            attn = self.inner_attn_ln(attn)

        attn = self.out_proj(attn)
        attn_weights = attn_weights.view(
            bsz, self.num_heads, tgt_len, src_len
        ).transpose(1, 0)

        return attn, attn_weights


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return "p={}".format(self.drop_prob)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output


def MultiwayWrapper(args, module, dim=1):
    if args.multiway:
        return MultiwayNetwork(module, dim=dim)
    return module


def set_split_position(position):
    def apply_fn(module):
        if hasattr(module, "split_position"):
            module.split_position = position

    return apply_fn


class MultiwayNetwork(nn.Module):
    def __init__(self, module, dim=1):
        super().__init__()
        self.dim = dim
        self.A = module
        self.B = copy.deepcopy(module)
        self.B.reset_parameters()
        self.split_position = -1

    def forward(self, x, **kwargs):
        if self.split_position == -1:
            return self.A(x, **kwargs)
        if self.split_position == 0:
            return self.B(x, **kwargs)
        x1, x2 = torch.split(
            x,
            [self.split_position, x.size(self.dim) - self.split_position],
            dim=self.dim,
        )
        # x1, x2 = x[:self.split_position], x[self.split_position:]
        y1, y2 = self.A(x1, **kwargs), self.B(x2, **kwargs)
        return torch.cat([y1, y2], dim=self.dim)


class MutliwayEmbedding(MultiwayNetwork):
    def __init__(self, modules, dim=1):
        super(MultiwayNetwork, self).__init__()
        self.dim = dim
        assert len(modules) == 2
        self.A = modules[0]
        self.B = modules[1]
        self.split_position = -1


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\


def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m


def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)


def get_activation_fn(activation):
    if activation == "swish":
        return F.silu
    elif activation == "gelu":
        return F.gelu
    else:
        raise NotImplementedError


class MultiScaleRetention(nn.Module):
    def __init__(
        self,
        args,
        embed_dim,
        value_dim,
        num_heads,
        gate_fn="swish",
    ):
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.head_dim = self.value_dim // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim**-0.5

        self.gate_fn = get_activation_fn(activation=str(gate_fn))

        self.q_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=False))
        self.k_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=False))
        self.v_proj = MultiwayWrapper(args, nn.Linear(embed_dim, value_dim, bias=False))
        self.g_proj = MultiwayWrapper(args, nn.Linear(embed_dim, value_dim, bias=False))

        self.out_proj = MultiwayWrapper(
            args, nn.Linear(value_dim, embed_dim, bias=False)
        )

        self.group_norm = MultiwayWrapper(
            args,
            RMSNorm(self.head_dim, eps=args.layernorm_eps, elementwise_affine=False),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2**-2.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2**-2.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2**-2.5)
        nn.init.xavier_uniform_(self.g_proj.weight, gain=2**-2.5)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=2**-1)

    def parallel_forward(self, qr, kr, v, mask):
        bsz, tgt_len, embed_dim = v.size()

        vr = v.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

        qk_mat = qr @ kr.transpose(-1, -2)  # bsz * m * tgt_len * tgt_len
        qk_mat = qk_mat * mask
        # invariant after normalization
        qk_mat = qk_mat / qk_mat.detach().abs().sum(dim=-1, keepdim=True).clamp(
            min=1, max=5e4
        )
        output = torch.matmul(qk_mat, vr)
        output = output.transpose(1, 2)
        return output

    def recurrent_forward(self, qr, kr, v, decay, incremental_state):
        bsz = v.size(0)

        v = v.view(bsz, self.num_heads, self.head_dim, 1)
        kv = kr * v
        if "prev_key_value" in incremental_state:
            prev_kv = incremental_state["prev_key_value"]
            prev_scale = incremental_state["scale"]
            scale = prev_scale * decay + 1
            kv = prev_kv * (prev_scale.sqrt() * decay / scale.sqrt()).view(
                self.num_heads, 1, 1
            ) + kv / scale.sqrt().view(self.num_heads, 1, 1)
            # kv = prev_kv * decay.view(self.num_heads, 1, 1) + kv
        else:
            scale = torch.ones_like(decay)

        incremental_state["prev_key_value"] = kv
        incremental_state["scale"] = scale

        output = torch.sum(qr * kv, dim=3)
        return output

    def chunk_recurrent_forward(self, qr, kr, v, inner_mask):
        mask, cross_decay, query_inner_decay, value_inner_decay = inner_mask
        bsz, tgt_len, embed_dim = v.size()
        chunk_len = mask.size(1)
        num_chunks = tgt_len // chunk_len

        assert tgt_len % chunk_len == 0

        qr = qr.view(
            bsz, self.num_heads, num_chunks, chunk_len, self.key_dim
        ).transpose(1, 2)
        kr = kr.view(
            bsz, self.num_heads, num_chunks, chunk_len, self.key_dim
        ).transpose(1, 2)
        v = v.view(bsz, num_chunks, chunk_len, self.num_heads, self.head_dim).transpose(
            2, 3
        )

        kr_t = kr.transpose(-1, -2)

        qk_mat = qr @ kr_t  # bsz * num_heads * chunk_len * chunk_len
        qk_mat = qk_mat * mask
        inner_scale = qk_mat.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1)
        qk_mat = qk_mat / inner_scale
        inner_output = torch.matmul(
            qk_mat, v
        )  # bsz * num_heads * num_value_heads * chunk_len * head_dim

        # reduce kv in one chunk
        kv = kr_t @ (v * value_inner_decay)

        kv_recurrent = []
        cross_scale = []
        kv_state = torch.zeros(bsz, self.num_heads, self.key_dim, self.head_dim).to(v)
        kv_scale = torch.ones(bsz, self.num_heads, 1, 1).to(v)

        # accumulate kv by loop
        for i in range(num_chunks):
            kv_recurrent.append(kv_state / kv_scale)
            cross_scale.append(kv_scale)
            kv_state = kv_state * cross_decay + kv[:, i]
            kv_scale = (
                kv_state.detach()
                .abs()
                .sum(dim=-2, keepdim=True)
                .max(dim=-1, keepdim=True)
                .values.clamp(min=1)
            )

        kv_recurrent = torch.stack(kv_recurrent, dim=1)
        cross_scale = torch.stack(cross_scale, dim=1)

        all_scale = torch.maximum(inner_scale, cross_scale)
        align_inner_scale = all_scale / inner_scale
        align_cross_scale = all_scale / cross_scale

        cross_output = (qr * query_inner_decay) @ kv_recurrent
        output = inner_output / align_inner_scale + cross_output / align_cross_scale
        # output = inner_output / cross_scale + cross_output / inner_scale

        output = output.transpose(2, 3)
        return output

    def forward(self, x, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        bsz, tgt_len, _ = x.size()
        (sin, cos), inner_mask = rel_pos

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        g = self.g_proj(x)

        k *= self.scaling
        q = q.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
        k = k.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)

        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)

        if incremental_state is not None:
            output = self.recurrent_forward(qr, kr, v, inner_mask, incremental_state)
        elif chunkwise_recurrent:
            output = self.chunk_recurrent_forward(qr, kr, v, inner_mask)
        else:
            output = self.parallel_forward(qr, kr, v, inner_mask)

        output = self.group_norm(output).reshape(
            bsz, tgt_len, self.head_dim * self.num_heads
        )

        output = self.gate_fn(g) * output

        output = self.out_proj(output)

        return output


def init_bert_params(module):
    def normal_(data):
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        if isinstance(module.q_proj, MultiwayNetwork):
            normal_(module.q_proj.A.weight.data)
            normal_(module.q_proj.B.weight.data)
            normal_(module.k_proj.A.weight.data)
            normal_(module.k_proj.B.weight.data)
            normal_(module.v_proj.A.weight.data)
            normal_(module.v_proj.B.weight.data)
        else:
            normal_(module.q_proj.weight.data)
            normal_(module.k_proj.weight.data)
            normal_(module.v_proj.weight.data)


class GLU(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn,
        dropout,
        activation_dropout,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = get_activation_fn(activation=str(activation_fn))
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim, bias=False)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim, bias=False)
        self.gate = nn.Linear(self.embed_dim, ffn_dim, bias=False)

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.gate.reset_parameters()

    def forward(self, x):
        x_shape = x.shape
        x = x.reshape(-1, x.size(-1))
        g = self.gate(x)
        x = self.fc1(x)
        x = self.activation_fn(x.float()).type_as(x) * g
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = x.view(x_shape)
        x = self.dropout_module(x)
        return x


class RetNetRelPos(nn.Module):
    def __init__(self, args):
        super().__init__()
        angle = 1.0 / (
            10000
            ** torch.linspace(
                0, 1, args.decoder_embed_dim // args.decoder_retention_heads // 2
            )
        )
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        decay = torch.log(
            1
            - 2 ** (-5 - torch.arange(args.decoder_retention_heads, dtype=torch.float))
        )
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)
        self.recurrent_chunk_size = args.recurrent_chunk_size

    def forward(self, slen, activate_recurrent=False, chunkwise_recurrent=False):
        if activate_recurrent:
            sin = torch.sin(self.angle * (slen - 1))
            cos = torch.cos(self.angle * (slen - 1))
            retention_rel_pos = ((sin, cos), self.decay.exp())
        elif chunkwise_recurrent:
            index = torch.arange(slen).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            cos = torch.cos(index[:, None] * self.angle[None, :])

            block_index = torch.arange(self.recurrent_chunk_size).to(self.decay)
            mask = torch.tril(
                torch.ones(self.recurrent_chunk_size, self.recurrent_chunk_size).to(
                    self.decay
                )
            )
            mask = torch.masked_fill(
                block_index[:, None] - block_index[None, :], ~mask.bool(), float("inf")
            )
            mask = torch.exp(mask * self.decay[:, None, None])
            mask = torch.nan_to_num(mask)

            value_inner_decay = mask[:, -1] / mask[:, -1].sum(dim=-1, keepdim=True)
            value_inner_decay = value_inner_decay.unsqueeze(-1)
            scale = mask.sum(dim=-1, keepdim=True).sqrt()
            inner_mask = mask / scale

            cross_decay = torch.exp(self.decay * self.recurrent_chunk_size)
            query_inner_decay = torch.exp(self.decay[:, None] * (block_index + 1))
            query_inner_decay = query_inner_decay[:, :, None] / (
                scale / mask[:, -1].sum(dim=-1)[:, None, None]
            )
            cross_decay = cross_decay[:, None, None]
            retention_rel_pos = (
                (sin, cos),
                (inner_mask, cross_decay, query_inner_decay, value_inner_decay),
            )
        else:
            index = torch.arange(slen).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            cos = torch.cos(index[:, None] * self.angle[None, :])
            mask = torch.tril(torch.ones(slen, slen).to(self.decay))
            mask = torch.masked_fill(
                index[:, None] - index[None, :], ~mask.bool(), float("inf")
            )
            mask = torch.exp(mask * self.decay[:, None, None])
            mask = torch.nan_to_num(mask)
            mask = mask / mask.sum(dim=-1, keepdim=True).sqrt()
            retention_rel_pos = ((sin, cos), mask)

        return retention_rel_pos


class DecoderLayer(nn.Module):
    def __init__(
        self,
        args,
        depth,
        is_moe_layer=False,
    ):
        super().__init__()
        self.args = args
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = torch.nn.Dropout(args.dropout)

        if args.drop_path_rate > 0:
            drop_path_prob = np.linspace(0, args.drop_path_rate, args.decoder_layers)[
                depth
            ]
            self.drop_path = DropPath(drop_path_prob)
        else:
            self.drop_path = None

        self.retention = self.build_retention(self.embed_dim, args)

        self.normalize_before = args.decoder_normalize_before

        self.retention_layer_norm = RMSNorm(self.embed_dim, eps=args.layernorm_eps)

        self.is_moe_layer = is_moe_layer
        self.ffn_dim = args.decoder_ffn_embed_dim

        self.ffn = self.build_ffn(
            self.embed_dim,
            self.args,
        )

        self.final_layer_norm = RMSNorm(self.embed_dim, eps=args.layernorm_eps)

        if args.deepnorm:
            self.alpha = math.pow(2.0 * args.decoder_layers, 0.25)
        else:
            self.alpha = 1.0

    def build_ffn(self, embed_dim, args):
        return GLU(
            embed_dim,
            self.ffn_dim,
            args.activation_fn,
            args.dropout,
            args.activation_dropout,
        )

    def build_retention(self, embed_dim, args):
        return MultiScaleRetention(
            args,
            embed_dim,
            args.decoder_value_embed_dim,
            args.decoder_retention_heads,
        )

    def residual_connection(self, x, residual):
        return residual * self.alpha + x

    def forward(
        self,
        x,
        incremental_state=None,
        chunkwise_recurrent=False,
        retention_rel_pos=None,
    ):
        residual = x
        if self.normalize_before:
            x = self.retention_layer_norm(x)

        x = self.retention(
            x,
            incremental_state=incremental_state,
            rel_pos=retention_rel_pos,
            chunkwise_recurrent=chunkwise_recurrent,
        )
        x = self.dropout_module(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.retention_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.ffn(x)
        l_aux = None

        if self.drop_path is not None:
            x = self.drop_path(x)

        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        return x, l_aux


class RetNetDecoder(nn.Module):
    def __init__(self, args, embed_tokens=None, output_projection=None, **kwargs):
        super().__init__(**kwargs)
        self.args = args

        self.dropout_module = torch.nn.Dropout(args.dropout)

        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_tokens = embed_tokens

        if (
            output_projection is None
            and not args.no_output_layer
            and args.vocab_size > 0
        ):
            self.output_projection = self.build_output_projection(args)
        else:
            self.output_projection = output_projection

        if args.layernorm_embedding:
            self.layernorm_embedding = RMSNorm(embed_dim, eps=args.layernorm_eps)
        else:
            self.layernorm_embedding = None

        self.layers = nn.ModuleList([])

        is_moe_layer = False
        for i in range(args.decoder_layers):
            self.layers.append(
                self.build_decoder_layer(
                    args,
                    depth=i,
                    is_moe_layer=False,
                )
            )

        self.num_layers = len(self.layers)

        if args.decoder_normalize_before:
            self.layer_norm = RMSNorm(embed_dim, eps=args.layernorm_eps)
        else:
            self.layer_norm = None

        self.retnet_rel_pos = RetNetRelPos(args)
        self.chunkwise_recurrent = args.chunkwise_recurrent
        self.recurrent_chunk_size = args.recurrent_chunk_size

        if args.deepnorm:
            init_scale = math.pow(8.0 * args.decoder_layers, 0.25)
            for name, p in self.named_parameters():
                if (
                    "fc1" in name
                    or "fc2" in name
                    or "out_proj" in name
                    or "v_proj" in name
                ):
                    p.data.div_(init_scale)

    def build_output_projection(
        self,
        args,
    ):
        if args.share_decoder_input_output_embed:
            output_projection = torch.nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            output_projection.weight = self.embed_tokens.weight
        else:
            output_projection = torch.nn.Linear(
                args.decoder_embed_dim, args.vocab_size, bias=False
            )
            torch.nn.init.normal_(
                output_projection.weight, mean=0, std=args.decoder_embed_dim**-0.5
            )
        return output_projection

    def build_decoder_layer(self, args, depth, is_moe_layer=False):
        layer = DecoderLayer(
            args,
            depth,
            is_moe_layer=is_moe_layer,
        )
        if args.checkpoint_activations:
            layer = checkpoint_wrapper(layer)
        if args.fsdp:
            layer = wrap(layer)
        return layer

    def forward_embedding(
        self,
        tokens,
        token_embedding=None,
        incremental_state=None,
    ):
        if incremental_state is not None and not self.is_first_step(incremental_state):
            tokens = tokens[:, -1:]

        if token_embedding is None:
            token_embedding = self.embed_tokens(tokens)

        x = embed = self.embed_scale * token_embedding

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        return x, embed

    def is_first_step(self, incremental_state):
        if incremental_state is None:
            return False
        return incremental_state.get("is_first_step", False)

    def forward(
        self,
        prev_output_tokens,
        incremental_state=None,
        features_only=False,
        return_all_hiddens=False,
        token_embeddings=None,
        **kwargs,
    ):
        # embed tokens
        x, _ = self.forward_embedding(
            prev_output_tokens, token_embeddings, incremental_state
        )
        is_first_step = self.is_first_step(incremental_state)

        if (
            self.chunkwise_recurrent
            and prev_output_tokens.size(1) % self.recurrent_chunk_size != 0
        ):
            padding_len = (
                self.recurrent_chunk_size
                - prev_output_tokens.size(1) % self.recurrent_chunk_size
            )
            slen = prev_output_tokens.size(1) + padding_len
            x = F.pad(x, (0, 0, 0, padding_len))
        else:
            slen = prev_output_tokens.size(1)
        # relative position
        retention_rel_pos = self.retnet_rel_pos(
            slen,
            incremental_state is not None and not is_first_step,
            chunkwise_recurrent=self.chunkwise_recurrent,
        )
        # decoder layers
        inner_states = [x]

        l_aux = []

        for idx, layer in enumerate(self.layers):
            if incremental_state is None or is_first_step:
                if is_first_step and incremental_state is not None:
                    if idx not in incremental_state:
                        incremental_state[idx] = {}
            else:
                if idx not in incremental_state:
                    incremental_state[idx] = {}

            x, l_aux_i = layer(
                x,
                incremental_state[idx] if incremental_state is not None else None,
                retention_rel_pos=retention_rel_pos,
                chunkwise_recurrent=self.chunkwise_recurrent,
            )
            l_aux.append(l_aux_i)
            inner_states.append(x)

        if (
            self.chunkwise_recurrent
            and prev_output_tokens.size(1) % self.recurrent_chunk_size != 0
        ):
            x = x[:, : prev_output_tokens.size(1), :]

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if not features_only:
            x = self.output_layer(x)

        return x, {
            "inner_states": inner_states,
            "l_aux": l_aux,
            "attn": None,
        }

    def output_layer(self, features):
        return self.output_projection(features)


class RetNet(nn.Module):
    def __init__(self, config: RetNetConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.decoder_embed_dim)
        self.transformer = RetNetDecoder(config, embed_tokens=self.embed_tokens)

        # init all weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        init_bert_params(module)

    def forward(self, idx, targets=None):
        logits = self.transformer(idx)[0]

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        else:
            logits = logits[:, [-1], :]
            loss = None
        return logits, loss
