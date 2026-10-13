# LLM

A small large language model trained from scratch with PyTorch and a
byte-level BPE tokenizer. Trains on books or GPT-4 TinyStories.
Trains 8000 steps in ~2 hours on M4 MacBook.

| Model | Configuration |
|---|---|
| Parameters | 29,152,256 (29.15M) with an 8,192-token vocabulary |
| Transformer | 8 layers, width 512, 4 attention heads |
| Attention | MLA, KV latent width 256, decoupled RoPE |
| Feed-forward | SwiGLU, hidden width 1,368 |
| Context | 512 tokens in the recipe below; CLI default is 256 |
| Regularization | Dropout 0, AdamW weight decay 0.1 |

Input embeddings and output weights are shared. MLA caches compressed KV latents
and rotary keys, using absorbed projections without reconstructing full per-token
keys and values.

<img width="1898" height="756" alt="image" src="https://github.com/user-attachments/assets/19cb16ba-d6ec-42fc-a9f6-51e4f3e3fcb1" />

<img width="480" alt="image" src="https://github.com/user-attachments/assets/c562b23d-67be-48a5-92a0-f17739667f35" />

## Train and generate

Python 3.11+. Automatically uses CUDA, Apple MPS, or CPU.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The earlier 80% fiction / 20% education recipe remains available. It retains
115 fiction works, with general educational books and maths books each receiving
10% of training tokens. Generated worked maths is excluded.

```sh
python -m llm.prepare --corpus balanced
python -m llm.train \
  --data-mixture data/balanced.json --token-cache-dir data/token-cache \
  --attention mla --out checkpoints/balanced-books-mla.pt \
  --context-size 512 --batch-size 4 --grad-accum-steps 4 \
  --steps 12000 --eval-every 200 --eval-batches 20
python -m llm.generate \
  --checkpoint checkpoints/balanced-books-mla.best.pt \
  --prompt "At midnight, someone knocked on the locked door." --tokens 300
```

For the full GPT-4-only TinyStories dataset:

```sh
python -m llm.prepare --corpus tinystories --story-version gpt4 --all-stories
python -m llm.train \
  --data data/tinystories-gpt4.txt --validation-data data/tinystories-gpt4-valid.txt \
  --token-cache-dir data/token-cache --attention mla \
  --out checkpoints/tinystories-gpt4.pt \
  --context-size 512 --batch-size 4 --grad-accum-steps 4 \
  --steps 12000 --eval-every 200 --eval-batches 20
python -m llm.generate \
  --checkpoint checkpoints/tinystories-gpt4.best.pt \
  --prompt "Jack and Lily liked to watch the moon at night." --tokens 300
```

Both recipes train from scratch with warmup and cosine learning-rate decay,
processing 98.3M token positions with an effective batch of 16. Disk-backed
caches keep the corpus out of RAM. Logs report validation loss and ETA;
`.best.pt` holds the best validation weights and `.pt` the final weights.
Checkpoints include their tokenizer. Use a different `--out` to keep an existing run.

Use `--help` for CLI options.

```sh
python -m unittest discover -s tests -v
```

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [PyTorch attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [BPE tokenizer training](https://huggingface.co/docs/tokenizers/python/latest/quicktour.html)
- [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories)
- [TinyStories paper](https://arxiv.org/abs/2305.07759)
- [RoPE](https://arxiv.org/abs/2104.09864)
- [SwiGLU](https://arxiv.org/abs/2002.05202)
- [LLaMA](https://arxiv.org/abs/2302.13971)
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434)
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437)
- [Attention implementation notes](docs/research.md)
- [Project Gutenberg books](https://www.gutenberg.org/)
- [80/20 recipe](docs/balanced.md)
- [GPT-4 TinyStories recipe](docs/tinystories.md)
- [Corpus guide](data/README.md)
