# Multi-Head Latent Attention

MLA follows [DeepSeek-V2](https://arxiv.org/abs/2405.04434) and
[DeepSeek-V3](https://arxiv.org/abs/2412.19437):
low-rank query and joint KV projections, decoupled RoPE, and absorbed key/value
projections. Only the KV latent and shared rotary key are cached.

Training defaults to `--attention mla`; use `--attention mha` for the baseline.
`--kv-rank` controls latent width (default: half model width).
The transformer keeps its existing LayerNorm and dense SwiGLU layers;
RMSNorm is used only inside MLA's compressed projections.

At width 512 with four heads, MLA caches 320 scalars per token per block versus 1,024 for MHA.
Cached and uncached greedy generation are tested for equivalence, including
sliding-window rollover. `--no-cache` disables caching.

Fresh training is required to switch attention types. Earlier dense checkpoints
remain loadable; incompatible architectures must be retrained.
Tests verify correctness, not improved generation quality or runtime speed.

Reference: [DeepSeek's official attention implementation](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py).
