# LLM

A minimal GPT-style language model trained from random weights. The model,
attention, and training loop are implemented here; PyTorch provides tensor
operations, automatic differentiation, and AdamW. Hugging Face Tokenizers trains
a byte-level BPE subword tokenizer from scratch on the training text. No
pretrained weights or pretrained vocabulary are used.

The default model has approximately 1.07 million parameters with a 2,048-token
vocabulary. The default corpus is Shakespeare's Complete Works from Project
Gutenberg.

## Run

Python 3.11 or newer is required. From the project directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m llm.prepare
python -m llm.train
python -m llm.generate --prompt "To be, or not to be"
python -m unittest discover -s tests -v
```

Alternatively, use `uv sync` and prefix the Python commands with `uv run`.
Training automatically selects CUDA, Apple MPS, or CPU. Override with
`--device cpu`, `--device cuda`, or `--device mps`.

A quick small run:

```sh
python -m llm.train --steps 100 --context-size 32 --n-layers 2 --n-embd 64 --batch-size 8 --eval-every 50
python -m llm.generate --prompt "ROMEO:" --tokens 100
```

`python -m llm.prepare` downloads the complete Gutenberg edition, removes its
header and footer, and preserves the original source and license alongside the
cleaned text. See [data/README.md](data/README.md) for provenance and offline use.
Preparation is only required once; training itself does not use the network.

Train on your own UTF-8 text with `--data path/to/corpus.txt`. Text is split
approximately 90/10 at a newline before tokenization. The tokenizer learns
merges only from the training split, then encodes both splits. Both must have
more tokens than the context length. The corpus is loaded into memory. The
contiguous holdout measures prediction on the final portion of the collection;
it is not a per-play benchmark or a general language evaluation.

The BPE vocabulary defaults to 2,048 tokens; change it with `--vocab-size` (at
least 256). Common words and word pieces receive learned token IDs. A complete
byte alphabet handles unseen words and Unicode without unknown tokens. Case,
spaces, punctuation, and line breaks are preserved. Vocabulary size may be
smaller than requested for very small corpora. Context and output lengths are
measured in BPE tokens, so a 128-token context spans more text than 128 bytes.

Checkpoints contain model configuration, weights, and the exact tokenizer JSON,
saved to `checkpoints/model.pt`. A readable `model.tokenizer.json` is also saved
beside the checkpoint. Generation uses the embedded tokenizer so token IDs
always agree with the model. Old byte-model checkpoints require retraining.
Checkpoints are for inference; optimizer state and training
resumption are intentionally omitted. `--out` selects a different destination,
and generation accepts `--checkpoint`. Running training again starts fresh and
overwrites that destination.

Generation samples one subword token at a time. `--temperature 0` selects the most
likely token; positive temperatures control randomness. `--top-k` restricts sampling to
the highest-scoring candidates. The seed defaults to 42. Invalid or incomplete
generated UTF-8 is displayed with replacement characters. Long prompts use the
most recent context window, and generation recomputes that window each step.

## Modules

| File | Responsibility |
| --- | --- |
| `llm/tokenizer.py` | Train, serialize, encode, and decode a BPE vocabulary |
| `llm/prepare.py` | Download and clean the Shakespeare corpus |
| `llm/model.py` | Configuration, attention, transformer blocks, sampling |
| `llm/train.py` | Data split, shifted batches, evaluation, optimization, saving |
| `llm/generate.py` | Checkpoint loading and text generation CLI |

## Basics and Research

1. **Tokens and positions:** each subword token becomes a learned embedding. A learned
   position embedding adds its location in the context window.
2. **Causal attention:** queries and keys produce scores scaled by the square
   root of the head width. A triangular mask prevents reading future tokens.
   Softmax weights combine values. Multiple heads learn different relationships.
3. **Transformer blocks:** pre-layer normalization, attention, a GELU feed-forward
   network, and residual connections transform the embeddings. Input embeddings
   and the final vocabulary projection share weights.
4. **Next-token prediction:** input tokens at positions `0..T-1` predict tokens at
   positions `1..T`. Cross-entropy measures the error. Backpropagation and AdamW
   update the parameters; gradient clipping limits large updates.
5. **Generation:** predict the next token, sample it, append it, and repeat.

The attention mechanism follows [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
The decoder-only, autoregressive design and pre-normalized blocks draw on
[the GPT-2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
Attention is written explicitly to keep its arithmetic visible; compare it with
[PyTorch's documented scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
when adding optimized kernels later.

The tokenizer follows [Hugging Face's BPE training API](https://huggingface.co/docs/tokenizers/python/latest/quicktour.html)
with byte-level pre-tokenization and decoding. The corpus comes from
[Project Gutenberg's Complete Works of William Shakespeare](https://www.gutenberg.org/ebooks/100).

To scale this foundation, the next practical steps are larger curated datasets,
efficient attention, mixed precision, and resumable
training. Increasing layers or width alone does not supply knowledge.
