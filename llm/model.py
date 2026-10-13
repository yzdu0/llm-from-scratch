from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .attention import CausalAttention, LatentAttention
from .experts import MixtureOfExperts
from .layers import SwiGLU, apply_rope, make_norm
from .prediction import MultiTokenPrediction


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
    attention: str = "mha"
    normalization: str = "layernorm"
    kv_rank: int = 0
    rope_dim: int = 0
    n_experts: int = 0
    experts_per_token: int = 2
    expert_hidden: int = 0
    dense_layers: int = 1
    router_bias_speed: float = 0.001
    mtp_weight: float = 0.0

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
        if self.attention not in ("mha", "mla") or self.normalization not in ("layernorm", "rmsnorm"):
            raise ValueError("Invalid attention or normalization")
        if self.position_encoding == "rope" and (self.n_embd // self.n_heads) % 2:
            raise ValueError("RoPE requires an even head dimension")
        if min(self.kv_rank, self.rope_dim, self.n_experts, self.expert_hidden, self.dense_layers) < 0:
            raise ValueError("Ranks, expert counts, and dimensions must be nonnegative")
        if self.rope_dim % 2 or (self.attention == "mla" and self.position_encoding != "rope"):
            raise ValueError("MLA requires RoPE with an even rotary dimension")
        if self.n_experts and (not 1 <= self.experts_per_token <= self.n_experts
                               or self.feed_forward != "swiglu"):
            raise ValueError("MoE requires SwiGLU and 1 <= experts_per_token <= n_experts")
        if (not math.isfinite(self.mtp_weight) or self.mtp_weight < 0
                or not math.isfinite(self.router_bias_speed) or self.router_bias_speed < 0):
            raise ValueError("Auxiliary weight and routing speed must be finite and nonnegative")


class Block(nn.Module):
    def __init__(self, config, use_experts=False):
        super().__init__()
        self.ln_attention = make_norm(config)
        self.attention = LatentAttention(config) if config.attention == "mla" else CausalAttention(config)
        self.ln_mlp = make_norm(config)
        if use_experts:
            self.mlp = MixtureOfExperts(config)
        elif config.feed_forward == "swiglu":
            self.mlp = SwiGLU(config)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(config.n_embd, 4 * config.n_embd), nn.GELU(),
                nn.Linear(4 * config.n_embd, config.n_embd), nn.Dropout(config.dropout)
            )

    def forward(self, x, cache=None):
        attended, cache = self.attention(self.ln_attention(x), cache)
        x = x + attended
        return x + self.mlp(self.ln_mlp(x)), cache


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = (nn.Embedding(config.context_size, config.n_embd)
                                   if config.position_encoding == "learned" else None)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config, bool(config.n_experts and i >= config.dense_layers))
                                    for i in range(config.n_layers))
        self.ln_final = make_norm(config)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.mtp = MultiTokenPrediction(config, Block(config)) if config.mtp_weight else None
        self.loss_components = {}
        self.apply(self._init_weights)
        self.head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens, targets=None, cache=None, use_cache=False):
        if cache is not None and (not use_cache or len(cache) != len(self.blocks)):
            raise ValueError("Pass one cache per block with use_cache=True")
        start = 0 if cache is None else cache[0][0].size(-2)
        if (tokens.ndim != 2 or tokens.size(1) < 1
                or tokens.size(1) + start > self.config.context_size):
            raise ValueError("tokens must have shape (batch, length) within context_size")
        if use_cache and (self.training or targets is not None):
            raise ValueError("Caching is for evaluation without targets")
        positions = torch.arange(start, start + tokens.size(1), device=tokens.device)
        x = self.token_embedding(tokens)
        if self.position_embedding is not None:
            x = x + self.position_embedding(positions)
        x = self.dropout(x)
        next_cache = []
        for i, block in enumerate(self.blocks):
            x, layer_cache = block(x, None if cache is None else cache[i])
            if use_cache:
                next_cache.append(layer_cache)
        logits = self.head(self.ln_final(x))
        loss = None
        self.loss_components = {}
        if targets is not None:
            if targets.shape != tokens.shape:
                raise ValueError("targets must have the same shape as tokens")
            loss = F.cross_entropy(logits.reshape(-1, self.config.vocab_size), targets.reshape(-1))
            self.loss_components["next_token"] = loss.detach()
            if self.training and self.mtp is not None and tokens.size(1) > 1:
                predicted = self.mtp(x[:, :-1], self.token_embedding(targets[:, :-1]))
                mtp_loss = F.cross_entropy(self.head(predicted).reshape(-1, self.config.vocab_size),
                                           targets[:, 1:].reshape(-1))
                self.loss_components["mtp"] = mtp_loss.detach()
                loss = loss + self.config.mtp_weight * mtp_loss
        return (logits, loss, next_cache) if use_cache else (logits, loss)

    @torch.no_grad()
    def update_router_bias(self):
        if self.training:
            for module in self.modules():
                if isinstance(module, MixtureOfExperts):
                    module.update_bias(self.config.router_bias_speed)

    @torch.no_grad()
    def generate(self, tokens, max_new_tokens=100, temperature=0.8, top_k=40,
                 eos_token_id=None, use_cache=True):
        if max_new_tokens < 0 or not math.isfinite(temperature) or temperature < 0:
            raise ValueError("max_new_tokens and temperature must be nonnegative and finite")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be positive or None")
        if tokens.ndim != 2 or tokens.size(1) == 0:
            raise ValueError("Generation requires a nonempty prompt")
        was_training = self.training
        self.eval()
        finished = torch.zeros(tokens.size(0), dtype=torch.bool, device=tokens.device)
        cache = None
        try:
            for _ in range(max_new_tokens):
                if use_cache:
                    if cache is None or cache[0][0].size(-2) >= self.config.context_size:
                        logits, _, cache = self(tokens[:, -self.config.context_size:], use_cache=True)
                    else:
                        logits, _, cache = self(tokens[:, -1:], cache=cache, use_cache=True)
                else:
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
                if eos_token_id is not None and finished.all():
                    break
        finally:
            self.train(was_training)
        return tokens
