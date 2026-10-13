# LLM

A small large language model trained from scratch with PyTorch and a
byte-level BPE tokenizer. Supports pretraining and answer-only Q&A fine-tuning.
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

Python 3.11+. The recipe retains all collected books and adds modern educational
writing, giving **78.9M training tokens**, then applies a short Q&A fine-tuning stage.
Sources follow their token counts and shuffle before repeating.

```sh
python3 -m venv .venv
source .venv/bin/activate
uv pip install -e '.[data]'
python -m llm.general_data
python -m llm.pipeline
python -m llm.generate \
  --checkpoint checkpoints/general-mla-qa.best.pt \
  --question "Why do leaves look green?" --tokens 150
```

Pretraining uses MLA, context 512 and an effective batch of 16. Its 12,000 updates
process 98.3M token positions, approximately 1.25 corpus passes. Q&A fine-tuning
loads the best checkpoint through `--init-from`, reuses its tokenizer, and trains
only on answers and EOS for at most 500 updates / one pass.

Training uses learning-rate warmup with cosine decay. Disk-backed token caches
avoid loading the full corpus into RAM. Progress includes validation loss and ETA;
`.best.pt` stores the best validation weights and `.pt` the final weights.
Checkpoints include their tokenizer. Continued training starts a new optimizer
and schedule. Use a new `--name` when running the pipeline again.

## Other corpora

The earlier 80% fiction / 20% education recipe remains available. Preparation
supports `balanced`, `books`, `darkfiction`, `tinystories`, `shakespeare`, and
`enzh` (English–Chinese translation). TinyStories defaults to 10,000 training
stories; `--story-version gpt4 --all-stories` selects the full GPT-4-only splits.
Weighted training across two corpora uses `--extra-data`,
`--extra-validation-data`, and `--extra-data-ratio` with a token cache.

Use `--help` for CLI options.

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
- [FineWeb-Edu dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [Smol-SmolTalk dataset](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk)
- [Two-stage training guide](docs/general-language.md)
- [Attention implementation notes](docs/research.md)