# GPT-4 TinyStories and dark fiction

Train a fresh model with the full prepared GPT-4-only corpus and the existing
Gutenberg dark-fiction corpus. Keep the model and optimizer settings from the
GPT-4 run. Choose the dark-fiction source for 10% of training sequences and
TinyStories for 90%, independently per sequence. Simply concatenating these
files would leave dark fiction at about 0.1% of the text.

```sh
.venv/bin/python -m llm.train \
  --data data/tinystories-gpt4.txt \
  --validation-data data/tinystories-gpt4-valid.txt \
  --extra-data data/darkfiction.txt \
  --extra-validation-data data/darkfiction-valid.txt \
  --extra-data-ratio 0.1 \
  --token-cache-dir data/token-cache \
  --out checkpoints/tinystories-gpt4-dark.pt \
  --context-size 512 --batch-size 4 --grad-accum-steps 4 \
  --steps 12000 --eval-every 200 --eval-batches 20
```

The shared 8,192-token BPE vocabulary is learned from both training files only.
Each corpus is encoded and cached separately; sampled windows do not cross
between source files. Both validation corpora stay held out, including all of
The Turn of the Screw. At each evaluation, logs show the individual validation
losses and a fixed 90/10 weighted validation loss for checkpoint selection.
The expected per-source corpus passes are logged to show oversampling of the
much smaller dark-fiction corpus. It may overfit; watch its validation loss.

The training budget remains 98,304,000 token positions, with approximately 9.8M
coming from dark fiction. This is an experiment in mixing narrative styles,
not a demonstrated coherence improvement. The classic fiction has longer,
more complex prose than TinyStories. It does not guarantee dark continuations
for every prompt. Adjust `--extra-data-ratio` for subsequent experiments.

Existing corpora and checkpoints are preserved. Generate from the best mixed
checkpoint after sufficient training:

```sh
.venv/bin/python -m llm.generate \
  --checkpoint checkpoints/tinystories-gpt4-dark.best.pt \
  --prompt "Lily heard a scratching sound behind the locked door." \
  --temperature 0.6 --top-k 0 --top-p 0.9 --tokens 300
```

The launched background run writes its progress to
`checkpoints/tinystories-gpt4-dark.train.log` and records its process ID in
`checkpoints/tinystories-gpt4-dark.pid`. Follow progress with:

```sh
tail -f checkpoints/tinystories-gpt4-dark.train.log
```

Initial preprocessing trains a new shared tokenizer and builds its token cache
before model updates begin. The step-0 checkpoint contains random initial
weights; wait for meaningful training progress before trying generation.
