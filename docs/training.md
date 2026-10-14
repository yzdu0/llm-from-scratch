# Training

```sh
python -m llm.pipeline --name books-200m-16l-16k
```

The pipeline prepares missing data, pretrains a raw MLA model, then fine-tunes its
best validation checkpoint on Q&A. Outputs are:

- `<name>.raw.pt` and `<name>.raw.best.pt`.
- `<name>.tuned.pt` and `<name>.tuned.best.pt`.

Pretraining uses 16 layers, width 384, context 512, batch size 4,
and four gradient-accumulation steps,
learning rate `3e-4`, weight decay `0.1`, and 24,000 updates. Warmup and cosine
decay control the learning rate. The 196,608,000-token budget is about 0.98 passes
through each source. Shuffled, non-overlapping windows avoid repeating positions
within a source epoch; fragments shorter than a window are omitted. The pipeline completes the full pretraining step budget.

Q&A fine-tuning uses learning rate `3e-5`, weight decay `0.01`, and at most 500
updates / one pass. The prepared examples fit context 512, giving 405 updates.
Questions and padding are masked; answers and EOS contribute to loss.
Overlength custom examples are skipped and counted. Validation does not advance
training cursors. Logs report losses, token exposure and ETA.

To fine-tune an MLA checkpoint and save a raw copy alongside the tuned model:

```sh
python -m llm.pipeline --name custom-qa \
  --raw-checkpoint checkpoints/books-200m-16l-16k.raw.best.pt
```

`llm.train --init-from` inherits weights, architecture and tokenizer, with a fresh
optimizer and schedule. Conflicting architecture settings are rejected.
Custom Q&A JSONL uses `{"question": "...", "answer": "..."}`; supply it with
`--task instruction`, `--data` and `--validation-data`. Identical normalized
questions across splits are rejected. `llm.generate --question` uses the training
template, prints only the answer and defaults to greedy decoding.
