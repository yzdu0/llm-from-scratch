# LLM

A minimal language model trained from scratch with PyTorch and a trained BPE
tokenizer. Defaults: 8,192 tokens, eight layers, width 512.
Uses RoPE, SwiGLU, and learning-rate warmup with cosine decay.
Uses MLA with compressed-cache decoding; `--attention mha` selects the baseline.

<img width="1372" height="408" alt="image" src="https://github.com/user-attachments/assets/a32af13a-5cbc-4308-bc38-3e2c8bc4a719" />

## Run

Python 3.11+. Automatically uses CUDA, Apple MPS, or CPU.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m llm.prepare --corpus tinystories
python -m llm.train --data data/tinystories.txt --validation-data data/tinystories-valid.txt --out checkpoints/tinystories.pt
python -m llm.generate --checkpoint checkpoints/tinystories.best.pt --prompt "At nine o'clock, the baker"
```

Alternatively, use `uv sync` and `uv run python ...`.

`--kv-rank` controls MLA compression; `--no-cache` disables cached generation.
See [implementation notes](docs/research.md).

TinyStories preparation selects 10,000 training stories and 1,000 validation
stories. Change with `--stories` and `--validation-stories`.
For Shakespeare, use `python -m llm.prepare --corpus shakespeare`, then
`python -m llm.train`. See [corpus details](data/README.md).

Checkpoints include the tokenizer: `<name>.best.pt` holds the best validation
weights; `<name>.pt` holds the final weights. Training starts fresh each run.
Earlier dense BPE checkpoints remain loadable. Switching attention requires retraining.
Use `--help` for training and generation options.

```sh
python -m unittest discover -s tests -v
```

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [PyTorch attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [BPE tokenizer training](https://huggingface.co/docs/tokenizers/python/latest/quicktour.html)
- [Shakespeare: Complete Works](https://www.gutenberg.org/ebooks/100)
- [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories)
- [TinyStories paper](https://arxiv.org/abs/2305.07759)
- [RoPE](https://arxiv.org/abs/2104.09864)
- [SwiGLU](https://arxiv.org/abs/2002.05202)
- [LLaMA](https://arxiv.org/abs/2302.13971)
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434)
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437)
