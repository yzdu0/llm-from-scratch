# LLM

A minimal language model trained from scratch with PyTorch and a trained BPE
tokenizer. Defaults: 29.43M parameters, 8,192 tokens, eight layers, width 512.
Uses RoPE, SwiGLU, and learning-rate warmup with cosine decay.
An optional DeepSeek-style profile adds MLA, sparse experts, RMSNorm, and multi-token prediction.

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

For the DeepSeek-V3-style profile (42.61M parameters), train fresh:

```sh
python -m llm.train --architecture deepseek --data data/tinystories.txt --validation-data data/tinystories-valid.txt --out checkpoints/deepseek.pt
python -m llm.generate --checkpoint checkpoints/deepseek.best.pt --prompt "At nine o'clock, the baker"
```

Both profiles use cached decoding; `--no-cache` disables it. See [implementation notes](docs/research.md).

TinyStories preparation selects 10,000 training stories and 1,000 validation
stories. Change with `--stories` and `--validation-stories`.
For Shakespeare, use `python -m llm.prepare --corpus shakespeare`, then
`python -m llm.train`. See [corpus details](data/README.md).

Checkpoints include the tokenizer: `<name>.best.pt` holds the best validation
weights; `<name>.pt` holds the final weights. Training starts fresh each run.
Existing BPE checkpoints remain loadable; new architecture features require retraining.
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
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)
- [mHC](https://arxiv.org/abs/2512.24880)
- [DeepSeek-V4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
