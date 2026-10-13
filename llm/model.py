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
    position_encoding: str = "rope"
    feed_forward: str = "swiglu"

    def __post_init__(self):
        if min(self.vocab_size, self.context_size, self.n_layers,
               self.n_heads, self.n_embd) < 1:
            raise ValueError("Model dimensions must be positive")
        if self.n_embd % self.n_heads:
            raise ValueError("n_embd must be divisible by n_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.position_encoding not in ("learned", "rope"):
            raise ValueError("position_encoding must be learned or rope")
        if self.feed_forward not in ("gelu", "swiglu"):
            raise ValueError("feed_forward must be gelu or swiglu")
        if self.position_encoding == "rope" and (self.n_embd // self.n_heads) % 2:
            raise ValueError("RoPE requires an even head dimension")


def apply_rope(x, cos, sin):
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    first, second = pairs.unbind(dim=-1)
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.stack((first * cos - second * sin,
                        first * sin + second * cos), dim=-1).flatten(-2)


class CausalAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.use_rope = config.position_encoding == "rope"
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.projection = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "mask", torch.ones(config.context_size, config.context_size,
                               dtype=torch.bool).tril(), persistent=False
        )
        if self.use_rope:
            head_dim = config.n_embd // config.n_heads
            frequencies = 10000.0 ** (-torch.arange(0, head_dim, 2).float() / head_dim)
            angles = torch.arange(config.context_size).float()[:, None] * frequencies
            self.register_buffer("rope_cos", angles.cos(), persistent=False)
            self.register_buffer("rope_sin", angles.sin(), persistent=False)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(batch, length, self.n_heads, width // self.n_heads)
                   .transpose(1, 2) for t in (q, k, v)]
        if self.use_rope:
            q = apply_rope(q, self.rope_cos[:length], self.rope_sin[:length])
            k = apply_rope(k, self.rope_cos[:length], self.rope_sin[:length])
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        scores = scores.masked_fill(~self.mask[:length, :length], float("-inf"))
        weights = self.dropout(F.softmax(scores, dim=-1))
        output = (weights @ v).transpose(1, 2).contiguous().view(batch, length, width)
        return self.dropout(self.projection(output))


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # Three projections at 8/3 width keep the budget near a 4x GELU MLP.
        hidden = math.ceil((8 * config.n_embd / 3) / 8) * 8
        self.gate = nn.Linear(config.n_embd, hidden, bias=False)
        self.up = nn.Linear(config.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_attention = nn.LayerNorm(config.n_embd)
        self.attention = CausalAttention(config)
        self.ln_mlp = nn.LayerNorm(config.n_embd)
        self.mlp = SwiGLU(config) if config.feed_forward == "swiglu" else nn.Sequential(
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
        self.position_embedding = (nn.Embedding(config.context_size, config.n_embd)
                                   if config.position_encoding == "learned" else None)
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
        x = self.token_embedding(tokens)
        if self.position_embedding is not None:
            x = x + self.position_embedding(positions)
        x = self.dropout(x)
        logits = self.head(self.ln_final(self.blocks(x)))
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size), targets.reshape(-1)
        )
        return logits, loss

    @torch.no_grad()
    def generate(self, tokens, max_new_tokens=100, temperature=0.8, top_k=40, eos_token_id=None):
        if max_new_tokens < 0 or not math.isfinite(temperature) or temperature < 0:
            raise ValueError("max_new_tokens and temperature must be nonnegative and finite")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be positive or None")
        if tokens.ndim != 2 or tokens.size(1) == 0:
            raise ValueError("Generation requires a nonempty prompt")
        was_training = self.training
        self.eval()
        finished = torch.zeros(tokens.size(0), dtype=torch.bool, device=tokens.device)
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
                if eos_token_id is not None:
                    next_token[finished] = eos_token_id
                    finished |= next_token.squeeze(-1) == eos_token_id
                tokens = torch.cat((tokens, next_token), dim=1)
                if finished.all():
                    break
        finally:
            self.train(was_training)
        return tokens
