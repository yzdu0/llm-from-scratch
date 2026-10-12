# Tiny LLM

A minimal GPT-style language model trained from random weights. The model,
attention, training loop, and tokenizer are implemented here; PyTorch provides
tensor operations, automatic differentiation, and AdamW. No pretrained weights
or transformer library are used.

The default model has approximately 0.84 million parameters. It is an educational
small language model, not a general-purpose assistant. The included original
sample text is only a smoke-test corpus; useful generation needs substantially
more varied text and training.

## Run

Python 3.11 or newer is required. From the project directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m llm.train
python -m llm.generate --prompt "The little "
python -m unittest discover -s tests -v
```

Alternatively, use `uv sync` and prefix the Python commands with `uv run`.
Training automatically selects CUDA, Apple MPS, or CPU. Override with
`--device cpu`, `--device cuda`, or `--device mps`.

A quick small run:

```sh
python -m llm.train --steps 100 --context-size 32 --n-layers 2 --n-embd 64 --batch-size 8 --eval-every 50
python -m llm.generate --prompt "The " --tokens 100
```

Train on your own UTF-8 text with `--data path/to/corpus.txt`. The last 10% of
bytes are held out for validation. Both splits must contain more bytes than the
context length (roughly 1,300 total bytes for the default context). The corpus
is loaded into memory. Related or repeated passages across the split can make
validation optimistic; use a clean, varied corpus for meaningful evaluation.

Checkpoints contain model configuration and weights, saved to
`checkpoints/model.pt`. They are for inference; optimizer state and training
resumption are intentionally omitted. `--out` selects a different destination,
and generation accepts `--checkpoint`. Running training again starts fresh and
overwrites that destination.

Generation samples one byte at a time. `--temperature 0` selects the most likely
byte; positive temperatures control randomness. `--top-k` restricts sampling to
the highest-scoring candidates. The seed defaults to 42. Invalid or incomplete
generated UTF-8 is displayed with replacement characters. Long prompts use the
most recent context window, and generation recomputes that window each step.

## Modules

| File | Responsibility |
| --- | --- |
| `llm/tokenizer.py` | UTF-8 text to/from a fixed 256-byte vocabulary |
| `llm/model.py` | Configuration, attention, transformer blocks, sampling |
| `llm/train.py` | Data split, shifted batches, evaluation, optimization, saving |
| `llm/generate.py` | Checkpoint loading and text generation CLI |

## Basics and Research

1. **Tokens and positions:** each byte becomes a learned embedding. A learned
   position embedding adds its location in the context window.
2. **Causal attention:** queries and keys produce scores scaled by the square
   root of the head width. A triangular mask prevents reading future tokens.
   Softmax weights combine values. Multiple heads learn different relationships.
3. **Transformer blocks:** pre-layer normalization, attention, a GELU feed-forward
   network, and residual connections transform the embeddings. Input embeddings
   and the final vocabulary projection share weights.
4. **Next-token prediction:** input bytes at positions `0..T-1` predict bytes at
   positions `1..T`. Cross-entropy measures the error. Backpropagation and AdamW
   update the parameters; gradient clipping limits large updates.
5. **Generation:** predict the next byte, sample it, append it, and repeat.

The attention mechanism follows [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
The decoder-only, autoregressive design and pre-normalized blocks draw on
[the GPT-2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
Attention is written explicitly to keep its arithmetic visible; compare it with
[PyTorch's documented scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
when adding optimized kernels later.

To scale this foundation, the next practical steps are a subword tokenizer,
larger curated datasets, efficient attention, mixed precision, and resumable
training. Increasing layers or width alone does not supply knowledge.
