import math

import torch
from torch import nn

from .layers import SwiGLU


class MixtureOfExperts(nn.Module):
    """One always-active shared expert and top-k routed experts, without token dropping."""

    def __init__(self, config):
        super().__init__()
        self.top_k = config.experts_per_token
        hidden = config.expert_hidden or math.ceil(8 * config.n_embd / (3 * (self.top_k + 1) * 8)) * 8
        self.router = nn.Linear(config.n_embd, config.n_experts, bias=False)
        self.shared = SwiGLU(config, hidden)
        self.experts = nn.ModuleList(SwiGLU(config, hidden) for _ in range(config.n_experts))
        self.register_buffer("routing_bias", torch.zeros(config.n_experts))
        self.register_buffer("counts", torch.zeros(config.n_experts), persistent=False)

    def route(self, x):
        scores = self.router(x).float().sigmoid()
        selected = (scores + self.routing_bias).topk(self.top_k, dim=-1).indices
        weights = scores.gather(-1, selected)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
        return selected, weights.to(x.dtype)

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        selected, weights = self.route(flat)
        if self.training:
            with torch.no_grad():
                self.counts.add_(torch.bincount(selected.flatten(), minlength=len(self.experts)))
        output = self.shared(flat)
        for expert_id, expert in enumerate(self.experts):
            tokens, slots = torch.where(selected == expert_id)
            if tokens.numel():
                contribution = expert(flat[tokens]) * weights[tokens, slots, None]
                output = output.index_add(0, tokens, contribution)
        return output.view(shape)

    @torch.no_grad()
    def update_bias(self, speed):
        self.routing_bias.add_(speed * (self.counts.mean() - self.counts).sign())
        self.counts.zero_()
