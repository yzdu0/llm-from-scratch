import argparse
from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
import math
from pathlib import Path
import time

import torch
import numpy as np

from .model import LanguageModel, ModelConfig
from .tokenizer import BPETokenizer
from .translation import TranslationData, load_translation_data


def choose_device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_data(path, context_size, vocab_size=8192, validation_path=None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found: {path}. Run python -m llm.prepare first.")
    text = path.read_text(encoding="utf-8")
    if validation_path is not None:
        train_text, validation_text = text, Path(validation_path).read_text(encoding="utf-8")
    else:
        # Split before learning merges, so validation cannot influence the vocabulary.
        split = text.rfind("\n", 0, int(0.9 * len(text)))
        split = int(0.9 * len(text)) if split <= 0 else split + 1
        train_text, validation_text = text[:split], text[split:]
    tokenizer = BPETokenizer.train(train_text, vocab_size)
    train, validation = [torch.tensor(tokenizer.encode(part), dtype=torch.long)
                         for part in (train_text, validation_text)]
    if min(len(train), len(validation)) <= context_size:
        raise ValueError("Corpus needs enough tokens for a context plus target in both splits")
    return train, validation, tokenizer


def load_cached_data(path, context_size, vocab_size=8192, validation_path=None,
                     cache_dir="data/token-cache", extra_path=None,
                     extra_validation_path=None, extra_ratio=0.1):
    """Encode bounded batches into disk-backed arrays for multi-GB story corpora."""
    if validation_path is None:
        raise ValueError("Disk-backed training requires a separate validation file")
    paths = [Path(path).resolve(), Path(validation_path).resolve()]
    if (extra_path is None) != (extra_validation_path is None):
        raise ValueError("Supply both extra training and validation files")
    if extra_path is not None:
        paths.extend([Path(extra_path).resolve(), Path(extra_validation_path).resolve()])
    if len(set(paths)) != len(paths):
        raise ValueError("Training and validation files must all differ")
    fingerprint = {"format": 1, "vocab_size": vocab_size, "files": [
        {"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
        for p in paths]}
    key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:20]
    root = Path(cache_dir) / key
    root.mkdir(parents=True, exist_ok=True)
    tokenizer_path = root / "tokenizer.json"
    if tokenizer_path.exists():
        tokenizer = BPETokenizer.from_json(tokenizer_path.read_text(encoding="utf-8"))
    else:
        tokenizer = BPETokenizer.train_file(paths[::2], vocab_size)
        temporary = tokenizer_path.with_suffix(".part")
        temporary.write_text(tokenizer.to_json(), encoding="utf-8")
        temporary.replace(tokenizer_path)
    dtype = np.uint16 if tokenizer.vocab_size <= 65536 else np.uint32
    splits = []
    for split, source in zip(("train", "validation", "extra_train", "extra_validation"), paths):
        encoded = root / f"{split}.bin"
        if not encoded.exists():
            print(f"Encoding {split} to {encoded}...", flush=True)
            temporary = encoded.with_suffix(".part")
            try:
                with source.open(encoding="utf-8", newline="") as stream, temporary.open("wb") as destination:
                    batch = []
                    for line in stream:
                        batch.append(line)
                        if len(batch) == 1024:
                            for record in tokenizer.tokenizer.encode_batch(batch, add_special_tokens=False):
                                np.asarray(record.ids, dtype=dtype).tofile(destination)
                            batch = []
                    if batch:
                        for record in tokenizer.tokenizer.encode_batch(batch, add_special_tokens=False):
                            np.asarray(record.ids, dtype=dtype).tofile(destination)
                if temporary.stat().st_size // np.dtype(dtype).itemsize <= context_size:
                    raise ValueError("Corpus needs enough tokens for a context plus target in both splits")
                temporary.replace(encoded)
            finally:
                temporary.unlink(missing_ok=True)
        data = np.memmap(encoded, dtype=dtype, mode="r")
        if len(data) <= context_size:
            raise ValueError("Corpus needs enough tokens for a context plus target in both splits")
        splits.append(data)
    if extra_path is not None:
        return (MixedData(splits[0], splits[2], extra_ratio),
                MixedData(splits[1], splits[3], extra_ratio), tokenizer)
    return *splits, tokenizer


class MixedData:
    """Choose the source independently per sequence, then sample a contiguous window."""

    def __init__(self, primary, extra, ratio):
        if not math.isfinite(ratio) or not 0 < ratio < 1:
            raise ValueError("Extra data ratio must be finite and in (0, 1)")
        self.primary, self.extra, self.ratio = primary, extra, ratio

    def __len__(self):
        return len(self.primary) + len(self.extra)

    def batch(self, batch_size, context_size, device, generator=None):
        selected = torch.rand(batch_size, generator=generator) < self.ratio
        x = torch.empty((batch_size, context_size), dtype=torch.long)
        y = torch.empty_like(x)
        for mask, data in ((~selected, self.primary), (selected, self.extra)):
            count = int(mask.sum())
            if count:
                bx, by = get_batch(data, count, context_size, "cpu", generator)
                x[mask], y[mask] = bx, by
        return x.to(device), y.to(device)


def learning_rate(step, total_steps, warmup_steps, peak_lr, min_ratio=0.1):
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
    progress = min(max(progress, 0.0), 1.0)
    return peak_lr * (min_ratio + (1 - min_ratio) * (1 + math.cos(math.pi * progress)) / 2)


def save_checkpoint(path, model, tokenizer, step, validation_loss, training_info=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    torch.save({"config": asdict(model.config), "model": state, "step": step,
                "validation_loss": validation_loss, "tokenizer": tokenizer.to_json(),
                "training_info": training_info or {}}, path)
    path.with_suffix(".tokenizer.json").write_text(tokenizer.to_json(), encoding="utf-8")


def get_batch(data, batch_size, context_size, device, generator=None):
    if isinstance(data, MixedData):
        return data.batch(batch_size, context_size, device, generator)
    if isinstance(data, TranslationData):
        return data.batch(batch_size, device, generator)
    starts = torch.randint(len(data) - context_size, (batch_size,), generator=generator)
    offsets = starts[:, None] + torch.arange(context_size)
    if isinstance(data, np.ndarray):
        indices = offsets.numpy()
        return (torch.from_numpy(data[indices].astype(np.int64)).to(device),
                torch.from_numpy(data[indices + 1].astype(np.int64)).to(device))
    return data[offsets].to(device), data[offsets + 1].to(device)


def make_optimizer(model, lr, weight_decay):
    parameters = list(model.parameters())
    groups = [{"params": [p for p in parameters if p.ndim >= 2], "weight_decay": weight_decay},
              {"params": [p for p in parameters if p.ndim < 2], "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr)


def train_step(model, data, optimizer, batch_size, device, accumulation_steps=1):
    """Average microbatch gradients, then clip and update once per optimizer step."""
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    optimizer.zero_grad(set_to_none=True)
    for _ in range(accumulation_steps):
        x, y = get_batch(data, batch_size, model.config.context_size, device)
        _, loss = model(x, y)
        (loss / accumulation_steps).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()


@torch.no_grad()
def evaluate(model, splits, batch_size, device, batches=10):
    was_training = model.training
    model.eval()
    generator = torch.Generator().manual_seed(0)
    losses = {}
    try:
        for name, data in splits.items():
            total = 0.0
            for _ in range(batches):
                x, y = get_batch(data, batch_size, model.config.context_size, device, generator)
                _, loss = model(x, y)
                total += loss.item()
            losses[name] = total / batches
    finally:
        model.train(was_training)
    return losses


def main():
    parser = argparse.ArgumentParser(description="Train a BPE-tokenized GPT from random weights")
    parser.add_argument("--task", choices=["language-model", "translation"], default="language-model")
    parser.add_argument("--data", help="Defaults to Shakespeare text or enzh JSONL for translation")
    parser.add_argument("--validation-data", help="Separate validation text; otherwise use a 90/10 split")
    parser.add_argument("--token-cache-dir", help="Stream corpus into disk-backed tokens here; requires validation-data")
    parser.add_argument("--extra-data", help="Second language-model training corpus, sampled independently")
    parser.add_argument("--extra-validation-data", help="Held-out split for extra-data")
    parser.add_argument("--extra-data-ratio", type=float, default=0.1,
                        help="Fraction of sequences sampled from extra-data (default: 0.1)")
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--attention", choices=["mha", "mla"], default="mha")
    parser.add_argument("--kv-rank", type=int, default=0, help="MLA latent width; 0 selects half model width")
    parser.add_argument("--out", help="Defaults to checkpoints/model.pt (translation: enzh.pt)")
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help="Microbatches per optimizer step; increases effective batch without larger activations")
    parser.add_argument("--context-size", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, help="Defaults to min(200, steps // 10)")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    translating = args.task == "translation"
    if args.token_cache_dir and (translating or not args.validation_data):
        parser.error("token-cache-dir requires language-model training and validation-data")
    if ((args.extra_data is None) != (args.extra_validation_data is None)
            or (args.extra_data and (translating or not args.token_cache_dir))):
        parser.error("extra-data requires extra-validation-data, token-cache-dir, and language-model training")
    if not math.isfinite(args.extra_data_ratio) or not 0 < args.extra_data_ratio < 1:
        parser.error("extra-data-ratio must be finite and in (0, 1)")
    args.data = args.data or ("data/enzh.jsonl" if translating else "data/shakespeare.txt")
    args.out = args.out or ("checkpoints/enzh.pt" if translating else "checkpoints/model.pt")
    if translating and args.validation_data is None and args.data == "data/enzh.jsonl":
        args.validation_data = "data/enzh-valid.jsonl"
    if (min(args.steps, args.batch_size, args.grad_accum_steps, args.eval_every, args.eval_batches) < 1
            or not math.isfinite(args.lr) or args.lr <= 0):
        parser.error("steps, batch-size, grad-accum-steps, eval-every, eval-batches, and lr must be positive")
    warmup = min(200, args.steps // 10) if args.warmup_steps is None else args.warmup_steps
    if not 0 <= warmup < args.steps or not 0 <= args.min_lr_ratio <= 1:
        parser.error("warmup-steps must be in [0, steps); min-lr-ratio must be in [0, 1]")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight-decay must be finite and nonnegative")
    torch.manual_seed(args.seed)
    device = choose_device() if args.device == "auto" else args.device
    print("Training BPE tokenizer and encoding corpus...", flush=True)
    if args.token_cache_dir:
        train, validation, tokenizer = load_cached_data(args.data, args.context_size,
                                                       args.vocab_size, args.validation_data,
                                                       args.token_cache_dir, args.extra_data,
                                                       args.extra_validation_data, args.extra_data_ratio)
    else:
        loader = load_translation_data if translating else load_data
        train, validation, tokenizer = loader(args.data, args.context_size, args.vocab_size,
                                             args.validation_data)
    config = ModelConfig(vocab_size=tokenizer.vocab_size,
                         context_size=args.context_size, n_layers=args.n_layers,
                         n_heads=args.n_heads, n_embd=args.n_embd, dropout=args.dropout,
                         attention=args.attention, kv_rank=args.kv_rank)
    model = LanguageModel(config).to(device)
    optimizer = make_optimizer(model, args.lr, args.weight_decay)
    print(f"device={device} attention={args.attention} "
          f"parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
    train_tokens = train.token_count if translating else len(train)
    validation_tokens = validation.token_count if translating else len(validation)
    print(f"vocabulary={tokenizer.vocab_size:,} train_tokens={train_tokens:,} "
          f"validation_tokens={validation_tokens:,}", flush=True)
    tokens_per_step = args.batch_size * args.grad_accum_steps * config.context_size
    training_info = {"args": vars(args).copy(), "train_tokens": train_tokens,
                     "validation_tokens": validation_tokens, "tokens_per_step": tokens_per_step}
    if isinstance(train, MixedData):
        training_info["source_tokens"] = {"primary": len(train.primary), "extra": len(train.extra)}
        budget = args.steps * tokens_per_step
        print(f"source_mix=primary:{1 - train.ratio:.0%},extra:{train.ratio:.0%} "
              f"primary_tokens={len(train.primary):,} extra_tokens={len(train.extra):,} "
              f"expected_primary_passes≈{budget * (1 - train.ratio) / len(train.primary):.2f} "
              f"expected_extra_passes≈{budget * train.ratio / len(train.extra):.2f}", flush=True)
    print(f"effective_batch={args.batch_size * args.grad_accum_steps} "
          f"tokens_per_step={tokens_per_step:,} token_budget={args.steps * tokens_per_step:,} "
          f"corpus_passes≈{args.steps * tokens_per_step / train_tokens:.2f}", flush=True)
    output = Path(args.out)
    best_output = output.with_name(output.stem + ".best" + output.suffix)
    best_loss = float("inf")
    training_started = None
    for step in range(args.steps + 1):
        lr = learning_rate(min(step, args.steps - 1), args.steps, warmup,
                           args.lr, args.min_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = lr
        if step % args.eval_every == 0 or step == args.steps:
            splits = {"train": train, "validation": validation}
            if isinstance(validation, MixedData):
                splits.update(primary_validation=validation.primary, extra_validation=validation.extra)
            losses = evaluate(model, splits,
                              args.batch_size, device, args.eval_batches)
            if isinstance(validation, MixedData):
                # Select checkpoints with a fixed source weighting rather than a noisy source draw.
                losses["validation"] = ((1 - validation.ratio) * losses["primary_validation"]
                                        + validation.ratio * losses["extra_validation"])
            eta = "--"
            if training_started is not None:
                remaining = (time.perf_counter() - training_started) / step * (args.steps - step)
                eta = str(timedelta(seconds=round(remaining)))
            print(f"step={step} train={losses['train']:.4f} validation={losses['validation']:.4f} "
                  f"gap={losses['validation'] - losses['train']:.4f} "
                  f"tokens_seen={step * tokens_per_step:,} lr={lr:.2e} eta={eta}",
                  flush=True)
            if isinstance(validation, MixedData):
                print(f"primary_validation={losses['primary_validation']:.4f} "
                      f"extra_validation={losses['extra_validation']:.4f}", flush=True)
            if losses["validation"] < best_loss:
                best_loss = losses["validation"]
                save_checkpoint(best_output, model, tokenizer, step, best_loss, training_info)
        if step == args.steps:
            break
        if training_started is None:
            training_started = time.perf_counter()
        train_step(model, train, optimizer, args.batch_size, device, args.grad_accum_steps)
    save_checkpoint(output, model, tokenizer, args.steps, losses["validation"], training_info)
    print(f"Saved {output}; best validation={best_loss:.4f}: {best_output}")


if __name__ == "__main__":
    main()
