# Attention implementation

The model uses Multi-Head Latent Attention (MLA): low-rank query and joint KV
projections, decoupled RoPE, and absorbed key/value projections. Generation caches
only the KV latent and shared rotary key, without reconstructing full per-token
keys and values.

At width 384 with four heads, each block caches 192 latent scalars and 48 rotary
scalars per token. LayerNorm surrounds the attention and dense SwiGLU blocks;
RMSNorm normalizes MLA's compressed projections. Input embeddings and output
weights are shared.

Tests compare absorbed attention against explicitly expanded projections and
verify cached/uncached greedy generation, including sliding-window rollover.
`--no-cache` disables generation caching. These checks establish implementation
correctness; generation quality depends on training and evaluation.

## References

- [DeepSeek-V2](https://arxiv.org/abs/2405.04434)
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437)
- [DeepSeek attention implementation](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py)
