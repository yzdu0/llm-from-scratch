import math

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight).to(x.dtype)


def make_norm(config):
    return RMSNorm(config.n_embd) if config.normalization == "rmsnorm" else nn.LayerNorm(config.n_embd)


def apply_rope(x, cos, sin):
    first, second = x.reshape(*x.shape[:-1], -1, 2).unbind(dim=-1)
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.stack((first * cos - second * sin,
                        first * sin + second * cos), dim=-1).flatten(-2)


class SwiGLU(nn.Module):
    def __init__(self, config, hidden=None):
        super().__init__()
        hidden = hidden or math.ceil((8 * config.n_embd / 3) / 8) * 8
        self.gate = nn.Linear(config.n_embd, hidden, bias=False)
        self.up = nn.Linear(config.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))
