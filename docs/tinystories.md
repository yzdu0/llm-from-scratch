# Improving TinyStories coherence

The inspected `tinystories.best.pt` has 29M parameters, eight layers, width 512,
MLA attention, a 256-token context, and dropout 0.15. Its best validation loss
was 2.2123 at step 7,900 of 8,000. Re-evaluation on ten deterministic batches per
split gives training loss 1.7252 and validation loss 2.2123. The gap suggests
some overfitting; this is not evidence that simply training longer on the same
small subset will solve the problem.

The training file contains only 10,000 stories (2,114,387 BPE tokens). The original
run sampled 16,384,000 token positions, roughly 7.75 corpus passes, with repeated
and overlapping random windows. Median story length is 184 tokens; 14.3% of
training stories exceed the 256-token context. A 512-token context covers at
least 95% of these stories and lets the model see more of each plot. Random
windows can still start midway through a story, so this does not guarantee a
complete story in every example.

## Try the existing checkpoint first

```sh
.venv/bin/python -m llm.generate \
  --checkpoint checkpoints/tinystories.best.pt \
  --prompt "Jack and Lily liked to watch the moon at night." \
  --temperature 0.6 --top-k 0 --top-p 0.9 --repetition-penalty 1 \
  --tokens 300 --seed 42
```

Nucleus sampling keeps a set of likely next tokens whose cumulative probability
reaches top-p. With top-k enabled it applies to the distribution after that cap.
It filters unlikely continuations; it cannot teach physical reasoning or plot
consistency. Global repetition penalties also penalize names and necessary
function words, so keep them off initially. For a comparison with the previous
CLI defaults, use `--temperature 0.7 --top-k 40 --top-p 1 --repetition-penalty 1.1`.
Keep prompts on one line unless a line break is intentional.

Sampling comparisons on the original weights still produced contradictions and
repetition, even without the penalty. A new sampling default is a control to
try, not a measured solution to coherence. Increasing `--tokens` only allows
a longer continuation; generation still stops at the learned end-of-story token.

## Retrain with more distinct stories and longer context

Keep the original data and checkpoint for comparison:

```sh
.venv/bin/python -m llm.prepare --corpus tinystories \
  --stories 100000 --validation-stories 2000 \
  --out data/tinystories-100k.txt

.venv/bin/python -m llm.train \
  --data data/tinystories-100k.txt \
  --validation-data data/tinystories-100k-valid.txt \
  --out checkpoints/tinystories-100k.pt \
  --context-size 512 --batch-size 4 --grad-accum-steps 4 \
  --steps 12000 --eval-every 200 --eval-batches 20
```

This keeps the existing model architecture, vocabulary target, dropout, and
optimizer settings. More distinct stories are the first intervention for the
observed training/validation gap. The larger context targets plot continuity;
gradient accumulation averages four microbatches into each update, for an
effective batch of 16 sequences without keeping four activation graphs in memory.
`--steps` counts optimizer updates, not microbatches. The token budget is
98,304,000, six times the original run, and the longer context adds attention
cost. Expect substantially longer training; use the reported ETA to assess it.
If memory is tight, use `--batch-size 2 --grad-accum-steps 8` for the same budget.

Training logs show sampled token positions, approximate corpus passes, and the
validation-minus-training loss gap. Checkpoints record training arguments and
corpus token counts. Corpus passes describe sampling volume, not exact coverage;
for translation the token budget includes padding and masked prompt positions.
New runs start from random weights and train a new tokenizer. Use the new
`.best.pt` checkpoint for generation.

## Full GPT-4-only dataset with the same training settings

The official [dataset card](https://huggingface.co/datasets/roneneldan/TinyStories)
identifies `TinyStoriesV2-GPT4-train.txt` as the GPT-4-only version. Use its
matching validation split. The published text files total approximately 2.25 GB.

```sh
.venv/bin/python -m llm.prepare --corpus tinystories \
  --story-version gpt4 --all-stories

.venv/bin/python -m llm.train \
  --data data/tinystories-gpt4.txt \
  --validation-data data/tinystories-gpt4-valid.txt \
  --token-cache-dir data/token-cache \
  --out checkpoints/tinystories-gpt4.pt \
  --context-size 512 --batch-size 4 --grad-accum-steps 4 \
  --steps 12000 --eval-every 200 --eval-batches 20
```

This retains the 100k recipe's architecture, training hyperparameters, and
98.3M-token budget. Training time should be similar to that recipe once tokens
are ready, though random disk reads can add overhead. Downloading and training
the tokenizer over the full corpus add initial preparation time. This fixed
budget samples from the entire dataset; it does not train through every story.
Check the logged `corpus_passes` before interpreting how much data was seen.

Preparation streams one story at a time to temporary files and publishes both
splits only after they finish successfully. Metadata records actual story counts.
The published GPT-4 files end with stories cut off mid-sentence;
full-file preparation discards unfinished trailing text and reports it.
For smaller experiments, omit `--all-stories` and supply story counts as before.

Use `--token-cache-dir` for the full dataset. It trains the tokenizer using only
the training file, encodes bounded batches of lines into compact token arrays,
and memory-maps those arrays. Only the sampled training batches become int64
tensors; the full corpus does not become a Python token list or an int64 tensor.
Line-wise encoding preserves the text but may segment whitespace differently
from the original whole-string loader. Cache keys include source paths, sizes,
modification times, and requested vocabulary size; subsequent runs reuse the
tokenizer and token arrays. Remove the cache to force rebuilding it. Tokenizer
training still uses memory for its vocabulary and word frequencies.

Keep several GB free for the text, token cache, and checkpoints. This starts a
fresh model; it does not resume the 100k checkpoint. Run one training job at a
time on the Mac, using `tinystories-gpt4.best.pt` for the resulting generations.

## Judge the result across several prompts and seeds

Use the same prompts and sampling settings on both checkpoints, with seeds
42, 43, and 44. Include the moon prompt, a lost toy, a child helping a friend,
and a problem that needs a simple solution. Check recurring character names,
physical plausibility, whether events follow from previous events, repetition,
and whether the story resolves its opening problem. Validation loss is useful
for model selection but does not capture all of these properties. Tokenizers
trained on different corpora can differ, so their token losses are not perfectly
comparable either.

If training loss falls while validation stalls or rises, keep the best checkpoint
and add more distinct training data before increasing capacity or steps. If both
losses are still improving, a larger training budget is worth testing. Do not
increase model size first: the [TinyStories paper](https://arxiv.org/abs/2305.07759)
demonstrates coherent stories with models below 10M parameters, so this model's
size alone does not explain the sample. This recipe is a proposed experiment;
its quality improvement needs a completed training run and story comparisons.
