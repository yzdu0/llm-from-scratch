# DeepSeek Adaptation

The optional `--architecture deepseek` profile implements small, single-device
versions of methods in [DeepSeek-V2](https://arxiv.org/abs/2405.04434),
[DeepSeekMoE](https://arxiv.org/abs/2401.06066), and
[DeepSeek-V3](https://arxiv.org/abs/2412.19437).

| Method | Implementation |
| --- | --- |
| Multi-head latent attention | Low-rank query and joint KV projections, decoupled RoPE, absorbed key/value projections. Caches only the KV latent and shared rotary key. |
| Sparse experts | First block dense; later blocks use four routed SwiGLU experts, top-two selection, and one shared expert. No token dropping. |
| Routing balance | Sigmoid affinities; correction bias affects selection, not mixture weights. Bias updates after each optimizer step; evaluation leaves routing state unchanged. |
| Multi-token prediction | One additional causal block combines the current representation with the next ground-truth token embedding to predict the following token. Shares embeddings and output head; training only. |
| RMSNorm | Replaces LayerNorm in the new profile; also normalizes compressed attention projections. |

Modules: `attention.py`, `experts.py`, `prediction.py`, and `layers.py`.
The MTP block is dense. Sequence-level balance penalties, distributed routing,
FP8 kernels, and speculative decoding are omitted. This is an educational
adaptation, not a reproduction of DeepSeek's training system.

## Use and Compare

`--kv-rank` controls the MLA latent width (default: half model width).
`--n-experts` and `--experts-per-token` control routing.
`--mtp-weight` scales the additional loss (default: 0.1; zero disables it).
Reported train/validation metrics are ordinary next-token cross-entropy;
the training objective additionally includes MTP.

At the current width of 512, MLA stores 320 scalars per token per block versus 1,024 for a
standard MHA cache. Cached and uncached generation are tested for equivalent
greedy output. Once the context window fills, both recompute the sliding window
to preserve its original semantics.

New architecture features require fresh training; old checkpoints retain their
saved architecture. Compare `dense` and `deepseek` on identical data, vocabulary,
context, seeds, and training budgets. The new profile has more total parameters
(42.61M versus 29.43M at eight layers and width 512); sparse Python dispatch can be slower on a small GPU.
Smoke tests establish correctness, not better generation quality.

[DeepSeek-R1](https://arxiv.org/abs/2501.12948) adds reasoning post-training with
reinforcement learning. That needs suitable tasks and rewards; a story corpus
alone does not provide them. No reasoning RL is implemented here.

Reference: [DeepSeek's official inference code](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py).

## Newer Research

[DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) adds learned sparse attention
for long contexts and more extensive reasoning/agent post-training.
[mHC](https://arxiv.org/abs/2512.24880) constrains mixing between multiple
residual streams. [DeepSeek-V4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
develops compressed attention and uses mHC. These are not implemented here:
the current 128-token context offers little room for long-context savings,
and widened residual streams add complexity beyond the compact V3 adaptation.
This is a scope choice, not a claim that newer methods cannot help small models.
V4 also uses Muon optimization; this implementation retains AdamW so architecture
comparisons can share the existing training setup.
