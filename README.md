# LLM

A small language model trained from scratch with PyTorch and byte-level BPE.
The pipeline produces separate raw and Q&A-tuned checkpoints.

| Model | Configuration |
|---|---|
| Parameters | 31,193,856 |
| Transformer | 16 layers, width 384, 4 heads |
| Attention | MLA, KV latent width 192, decoupled RoPE |
| Feed-forward | SwiGLU, hidden width 1,024 |
| Vocabulary / context | 8,192 tokens / 512 tokens |
| Regularization | Dropout 0, weight decay 0.1 |

The corpus contains **99,992,938 training tokens**: **80% fiction / 20% education**.
Education covers literature and criticism, humanities, history, and science.
Q&A fine-tuning uses 6,485 training examples and 330 validation examples.

## Run

Python 3.11+. Training automatically uses CUDA, Apple MPS, or CPU.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[data]'
python -m llm.prepare
python -m llm.pipeline --name books-100m-16l
```

Preparation downloads and caches sources. The pipeline also prepares missing data.
Pretraining runs 20,000 updates with an effective batch of 16 and shuffled
windows. Fine-tuning reuses the best raw weights and tokenizer,
trains only on answers and EOS, and runs up to 500 updates / one pass.
Use a unique `--name` for each run.

```sh
python -m llm.generate \
  --checkpoint checkpoints/books-100m-16l.raw.best.pt \
  --prompt "At midnight, someone knocked on the locked door." --tokens 300
python -m llm.generate \
  --checkpoint checkpoints/books-100m-16l.tuned.best.pt \
  --question "Why do leaves look green?" --tokens 150
```

`.raw.pt` and `.tuned.pt` contain final weights; `.best.pt` files contain the best
validation weights. Every checkpoint includes its tokenizer. Use `--help` for options.

```sh
python -m unittest discover -s tests -v
```

## References

- [Training](docs/training.md)
- [Corpus](data/README.md)
- [Attention implementation](docs/research.md)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [BPE tokenizer training](https://huggingface.co/docs/tokenizers/python/latest/quicktour.html)
- [RoPE](https://arxiv.org/abs/2104.09864)
- [SwiGLU](https://arxiv.org/abs/2002.05202)
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434)
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437)
- [Project Gutenberg](https://www.gutenberg.org/)
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [Smol-SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk)
