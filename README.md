# LLM

A small decoder-only language model trained from scratch with PyTorch and a
byte-level BPE tokenizer. Trains in ~2 hours with 8,000 steps on M4 MacBook Pro.

| Model | Configuration |
|---|---|
| Parameters | 29,152,256 (29.15M) with an 8,192-token vocabulary |
| Transformer | 8 layers, width 512, 4 attention heads |
| Attention | MLA, KV latent width 256, decoupled RoPE |
| Feed-forward | SwiGLU, hidden width 1,368 |
| Context | 512 tokens in the recipe below; CLI default is 256 |
| Regularization | Dropout 0.15, AdamW weight decay 0.1 |

Input embeddings and output weights are shared. MLA caches compressed KV latents
and rotary keys, using absorbed projections without reconstructing full per-token
keys and values. See [attention implementation notes](docs/research.md).

<img width="1898" height="756" alt="image" src="https://github.com/user-attachments/assets/19cb16ba-d6ec-42fc-a9f6-51e4f3e3fcb1" />

## Train and generate

Python 3.11+. Training samples **80% fiction and 20% education**. Education
includes 6% general nonfiction, 6% maths books, and 8% worked maths as shares
of the full training budget. All 115 fiction works remain included.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m llm.prepare --corpus balanced
python -m llm.train \
  --data-mixture data/balanced.json --token-cache-dir data/token-cache \
  --attention mla --out checkpoints/balanced-80-20-mla.pt \
  --context-size 512 --batch-size 4 --grad-accum-steps 4 \
  --steps 12000 --eval-every 200 --eval-batches 20
python -m llm.generate \
  --checkpoint checkpoints/balanced-80-20-mla.best.pt \
  --prompt "At midnight, someone knocked on the locked door." \
  --temperature 0.6 --top-k 0 --top-p 0.9 --tokens 300
```

Training starts fresh and uses learning-rate warmup with cosine decay. The recipe
processes 98.3M token positions with an effective batch of 16. Disk-backed token
caches avoid loading the full corpus into RAM. Progress includes validation loss
and ETA; `.best.pt` stores the best validation weights and `.pt` the final weights.
Checkpoints include their tokenizer.

## Other corpora

The [80/20 experiment](docs/balanced.md) approximates dialogue and Q&A
with historical texts. Preparation supports `balanced`, `books`, `darkfiction`, `tinystories`, `shakespeare`, and
`enzh` (English–Chinese translation). TinyStories defaults to 10,000 training
stories; `--story-version gpt4 --all-stories` selects the full GPT-4-only splits.
Weighted training across two corpora uses `--extra-data`,
`--extra-validation-data`, and `--extra-data-ratio` with a token cache.

See the [corpus guide](data/README.md), [TinyStories experiments](docs/tinystories.md),
and [GPT-4 stories with dark fiction](docs/gpt4-dark.md). Use `--help` for CLI options.

```sh
python -m unittest discover -s tests -v
```
