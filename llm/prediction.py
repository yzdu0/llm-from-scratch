import torch
from torch import nn

from .layers import RMSNorm, make_norm


class MultiTokenPrediction(nn.Module):
    """One sequential prediction depth, conditioned on the next ground-truth token."""

    def __init__(self, config, block):
        super().__init__()
        self.hidden_norm = RMSNorm(config.n_embd)
        self.embedding_norm = RMSNorm(config.n_embd)
        self.merge = nn.Linear(2 * config.n_embd, config.n_embd, bias=False)
        self.block = block
        self.final_norm = make_norm(config)

    def forward(self, hidden, next_embeddings):
        combined = torch.cat((self.hidden_norm(hidden), self.embedding_norm(next_embeddings)), dim=-1)
        hidden, _ = self.block(self.merge(combined))
        return self.final_norm(hidden)
