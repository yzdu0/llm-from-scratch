# 80% fiction / 20% education

This MLA recipe samples 80% fiction, 10% general educational books, and 10%
maths books. It includes 115 fiction works and 16 educational works. Complete
books are held out for validation; generated worked maths is excluded.

```sh
.venv/bin/python -m llm.prepare --corpus balanced
.venv/bin/python -m llm.train \
  --data-mixture data/balanced.json --token-cache-dir data/token-cache \
  --attention mla --out checkpoints/balanced-books-mla.pt \
  --context-size 512 --batch-size 4 --grad-accum-steps 4 \
  --steps 12000 --eval-every 200 --eval-batches 20
.venv/bin/python -m llm.generate \
  --checkpoint checkpoints/balanced-books-mla.best.pt \
  --prompt "At midnight, someone knocked on the locked door." --tokens 300
```

The manifest controls exposure independently of source sizes. One source is
chosen for each sequence, followed by a contiguous window within that source.
The shared BPE vocabulary learns from training text only. Per-source validation
losses and their weighted average select the best checkpoint. Logs show expected
passes for each source so excessive repetition is visible.

The 12,000-update recipe processes 98.3M token positions. Original Gutenberg
files and licenses remain in `data/books-sources`; category metadata records
titles, authors, URLs, hashes and splits. Historical facts and perspectives
reflect their sources. Whole-book splits do not eliminate shared quotations.
