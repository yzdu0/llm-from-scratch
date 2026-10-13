# GPT-4 TinyStories

Preparation uses the full official GPT-4-only training and validation splits.
Stories retain their EOS marker; incomplete trailing records are discarded.
The model trains from scratch with a vocabulary learned from training text only.

```sh
.venv/bin/python -m llm.prepare --corpus tinystories --story-version gpt4 --all-stories
.venv/bin/python -m llm.train \
  --data data/tinystories-gpt4.txt --validation-data data/tinystories-gpt4-valid.txt \
  --token-cache-dir data/token-cache --attention mla \
  --out checkpoints/tinystories-gpt4.pt \
  --context-size 512 --batch-size 4 --grad-accum-steps 4 \
  --steps 12000 --eval-every 200 --eval-batches 20
.venv/bin/python -m llm.generate \
  --checkpoint checkpoints/tinystories-gpt4.best.pt \
  --prompt "Jack and Lily liked to watch the moon at night." \
  --temperature 0.6 --top-k 0 --top-p 0.9 --tokens 300
```

The text download is approximately 2.25 GB. Token caches use bounded encoding
batches and memory-mapped arrays; keep several GB free. Preparation metadata
records source URLs, license and actual story counts. Downloads publish both
splits only after successful preparation. Omit `--all-stories` and supply
`--stories` / `--validation-stories` for smaller GPT-4-only subsets.

This recipe processes 98.3M token positions, rather than a full pass through
all stories. Logs report corpus passes, validation loss and ETA. Check coherence,
character consistency and repetition across several prompts and seeds; lower
validation loss alone does not guarantee better stories.
