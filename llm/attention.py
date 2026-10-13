import math

import torch
from torch import nn
from torch.nn import functional as F

from .layers import RMSNorm, apply_rope


def register_positions(module, config, rope_dim):
    module.register_buffer("mask", torch.ones(config.context_size, config.context_size,
                                             dtype=torch.bool).tril(), persistent=False)
    frequencies = 10000.0 ** (-torch.arange(0, rope_dim, 2).float() / rope_dim)
    angles = torch.arange(config.context_size).float()[:, None] * frequencies
    module.register_buffer("rope_cos", angles.cos(), persistent=False)
    module.register_buffer("rope_sin", angles.sin(), persistent=False)


class CausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.use_rope = config.position_encoding == "rope"
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.projection = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        register_positions(self, config, config.n_embd // config.n_heads)

    def forward(self, x, cache=None):
        batch, length, width = x.shape
        start = 0 if cache is None else cache[0].size(-2)
        end = start + length
        q, k, v = [t.view(batch, length, self.n_heads, width // self.n_heads).transpose(1, 2)
                   for t in self.qkv(x).chunk(3, dim=-1)]
        if self.use_rope:
            q = apply_rope(q, self.rope_cos[start:end], self.rope_sin[start:end])
            k = apply_rope(k, self.rope_cos[start:end], self.rope_sin[start:end])
        if cache is not None:
            k, v = [torch.cat((past, current), dim=-2) for past, current in zip(cache, (k, v))]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        scores = scores.masked_fill(~self.mask[start:end, :end], float("-inf"))
        weights = self.dropout(F.softmax(scores, dim=-1))
        output = (weights @ v).transpose(1, 2).contiguous().view(batch, length, width)
        return self.dropout(self.projection(output)), (k, v)


class LatentAttention(nn.Module):
    """MLA with decoupled RoPE and a cache of latent KV plus one shared rotary key."""

    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.n_embd // config.n_heads
        self.rank = config.kv_rank or max(1, config.n_embd // 2)
        self.rope_dim = config.rope_dim or max(2, (self.head_dim // 4) * 2)
        self.q_down = nn.Linear(config.n_embd, self.rank, bias=False)
        self.q_norm = RMSNorm(self.rank)
        self.q_up = nn.Linear(self.rank, self.n_heads * (self.head_dim + self.rope_dim), bias=False)
        self.kv_down = nn.Linear(config.n_embd, self.rank + self.rope_dim, bias=False)
        self.kv_norm = RMSNorm(self.rank)
        self.kv_up = nn.Linear(self.rank, 2 * config.n_embd, bias=False)
        self.projection = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        register_positions(self, config, self.rope_dim)

    def forward(self, x, cache=None):
        batch, length, width = x.shape
        start = 0 if cache is None else cache[0].size(-2)
        end = start + length
        q = self.q_up(self.q_norm(self.q_down(x)))
        q = q.view(batch, length, self.n_heads, -1).transpose(1, 2)
        q_content, q_rope = q.split((self.head_dim, self.rope_dim), dim=-1)
        latent, k_rope = self.kv_down(x).split((self.rank, self.rope_dim), dim=-1)
        latent = self.kv_norm(latent)
        q_rope = apply_rope(q_rope, self.rope_cos[start:end], self.rope_sin[start:end])
        k_rope = apply_rope(k_rope.unsqueeze(1), self.rope_cos[start:end],
                            self.rope_sin[start:end]).squeeze(1)
        if cache is not None:
            latent, k_rope = [torch.cat((past, current), dim=1)
                              for past, current in zip(cache, (latent, k_rope))]
        k_weight, v_weight = self.kv_up.weight.view(self.n_heads, 2 * self.head_dim, self.rank).chunk(2, dim=1)
        # Absorb the key up-projection into queries; full per-head KV is never cached.
        q_latent = torch.einsum("bhtd,hdr->bhtr", q_content, k_weight)
        scores = (torch.einsum("bhtr,bsr->bhts", q_latent, latent)
                  + torch.einsum("bhtd,bsd->bhts", q_rope, k_rope))
        scores = scores / math.sqrt(self.head_dim + self.rope_dim)
        scores = scores.masked_fill(~self.mask[start:end, :end], float("-inf"))
        weights = self.dropout(F.softmax(scores, dim=-1))
        mixed = torch.einsum("bhts,bsr->bhtr", weights, latent)
        output = torch.einsum("bhtr,hdr->bhtd", mixed, v_weight)
        output = output.transpose(1, 2).contiguous().view(batch, length, width)
        return self.dropout(self.projection(output)), (latent, k_rope)
