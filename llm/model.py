from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 256
    context_size: int = 128
    n_layers: int = 4
    n_heads: int = 4
    n_embd: int = 128
    dropout: float = 0.1

    def __post_init__(self):
        if min(self.vocab_size, self.context_size, self.n_layers,
               self.n_heads, self.n_embd) < 1:
            raise ValueError("Model dimensions must be positive")
        if self.n_embd % self.n_heads:
            raise ValueError("n_embd must be divisible by n_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


class CausalAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.projection = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "mask", torch.ones(config.context_size, config.context_size,
                               dtype=torch.bool).tril(), persistent=False
        )

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(batch, length, self.n_heads, width // self.n_heads)
                   .transpose(1, 2) for t in (q, k, v)]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        scores = scores.masked_fill(~self.mask[:length, :length], float("-inf"))
        weights = self.dropout(F.softmax(scores, dim=-1))
        output = (weights @ v).transpose(1, 2).contiguous().view(batch, length, width)
        return self.dropout(self.projection(output))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_attention = nn.LayerNorm(config.n_embd)
        self.attention = CausalAttention(config)
        self.ln_mlp = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd), nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd), nn.Dropout(config.dropout)
        )

    def forward(self, x):
        x = x + self.attention(self.ln_attention(x))
        return x + self.mlp(self.ln_mlp(x))


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.context_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*(Block(config) for _ in range(config.n_layers)))
        self.ln_final = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens, targets=None):
        if tokens.ndim != 2 or not 1 <= tokens.size(1) <= self.config.context_size:
            raise ValueError("tokens must have shape (batch, length) within context_size")
        positions = torch.arange(tokens.size(1), device=tokens.device)
        x = self.dropout(self.token_embedding(tokens) + self.position_embedding(positions))
        logits = self.head(self.ln_final(self.blocks(x)))
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size), targets.reshape(-1)
        )
        return logits, loss

    @torch.no_grad()
    def generate(self, tokens, max_new_tokens=100, temperature=0.8, top_k=40):
        if max_new_tokens < 0 or not math.isfinite(temperature) or temperature < 0:
            raise ValueError("max_new_tokens and temperature must be nonnegative and finite")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be positive or None")
        if tokens.ndim != 2 or tokens.size(1) == 0:
            raise ValueError("Generation requires a nonempty prompt")
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                logits, _ = self(tokens[:, -self.config.context_size:])
                logits = logits[:, -1, :]
                if temperature == 0:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if top_k is not None:
                        threshold = logits.topk(min(top_k, logits.size(-1))).values[:, -1:]
                        logits = logits.masked_fill(logits < threshold, float("-inf"))
                    next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
                tokens = torch.cat((tokens, next_token), dim=1)
        finally:
            self.train(was_training)
        return tokens
