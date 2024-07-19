import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig


class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        config.ssm_cfg = {
            "d_state": config.d_state
        }
        self.transformer = MambaLMHeadModel(config)

    def forward(self, idx, targets=None):
        logits = self.transformer(idx).logits

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
