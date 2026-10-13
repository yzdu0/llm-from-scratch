# 80% fiction / 20% education MLA experiment

The active mixture keeps all 115 fiction editions and samples them for 80% of
training sequences. Education receives the remaining 20%, preserving its prior
internal balance: 30% general education, 30% maths books, and 40% worked maths.

| Source | Share of all training tokens |
|---|---:|
| Fiction | 80% |
| General education | 6% |
| Maths books | 6% |
| Worked maths | 8% |

Maths therefore receives 14% of total exposure, about 13.8M sampled positions
within the existing 98.3M-token budget. The active book sources contain 131
editions, with whole works held out. All previously downloaded books and source
files remain on disk. The standalone dialogue, technical Q&A, and essay sources
are no longer sampled by this manifest.

```sh
.venv/bin/python -m llm.prepare --corpus balanced
.venv/bin/python -m llm.train \
  --data-mixture data/balanced.json --token-cache-dir data/token-cache \
  --attention mla --out checkpoints/balanced-80-20-mla.pt \
  --context-size 512 --batch-size 4 --grad-accum-steps 4 \
  --steps 12000 --eval-every 200 --eval-batches 20
```

Preparation is already complete for this experiment. The first command rebuilds
the files from the cached source editions when needed. It also updates the
standalone books-only corpus. No TinyStories is included in this mixture.
Source originals and licenses remain in `data/books-sources`; per-category
provenance files record the titles, authors, URLs, hashes, and split assignments.
The catalog is in `llm/book_catalog.py` and `llm/prepare.py`.

`data/balanced.json` contains the sampling weights. Training chooses a category
independently for each fixed-length sequence, then samples a contiguous token
window within that category. Thus the shares control training token exposure
in expectation, regardless of each file's size. Small batches fluctuate around
the requested shares. The tokenizer is shared and trained on all training files
only, using their natural text frequencies rather than the sampling weights.

Validation reports a separate loss for every category. The weighted average
using the same shares selects `balanced-80-20-mla.best.pt`; final weights are saved
to `balanced-80-20-mla.pt`. Checkpoints record the actual source weights and token
counts. The existing context, model size, and 98.3M-token training budget remain
unchanged. `--attention mla` explicitly selects latent attention with decoupled
RoPE and compressed KV caching. No previous checkpoint is resumed.

This is next-token pretraining over fiction, nonfiction, maths exposition,
and exact worked examples. Historical scientific facts and social perspectives
reflect their sources. Maths capability needs evaluation after training.

Whole editions are held out, rather than random passages from the same books.
Different historical works may still quote one another; this split does not
guarantee there are no shared quotations across editions.

For fiction-only training, use `--data data/books.txt --validation-data
data/books-valid.txt` instead of `--data-mixture`, with the same cache and MLA
flags and a separate output checkpoint.

## Worked maths data

`llm/maths.py` produces deterministic Question/Solution/Answer records using
exact integers and rational fractions, plus the polynomial power rule. The
12 task families span small and large integer arithmetic, remainders, fraction
addition and multiplication, percentages, linear equations, rectangular and
triangular areas, equally likely probability, and polynomial differentiation.
Answers for all 110,000 generated examples were independently recalculated
and checked. No floating-point approximation is used for numeric answers.

Each canonical problem has a hash ID that determines its split. Duplicate IDs
are removed, and training/validation IDs are disjoint. Validation shares task
templates with training but uses different problems; it tests generalization
to new parameters rather than new task types. Audit JSONL files and generator
metadata are retained in `data/maths-sources/`.

Readable textbook editions include De Morgan's *Elements of Arithmetic*,
Rivenburg's *A Review of Algebra*, and Marks' geometry lessons. Probability
and number-concept books are held out. Books available only as images or as
placeholder text were excluded. Historical diagrams are not reconstructed.

The maths books and worked problems have separate validation losses and source
weights in checkpoints. The shared tokenizer and token cache have already
been prepared for this mixture. The 12,000-update budget remains unchanged;
training a new checkpoint is necessary to use the new mixture.
