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

from .checkpoint import load_model
from .instruction import InstructionData, load_instruction_data
from .model import LanguageModel, ModelConfig
from .tokenizer import BPETokenizer, DEFAULT_VOCAB_SIZE


def choose_device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_data(path, context_size, vocab_size=DEFAULT_VOCAB_SIZE, validation_path=None, tokenizer=None):
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
    tokenizer = tokenizer or BPETokenizer.train(train_text, vocab_size)
    train, validation = [torch.tensor(tokenizer.encode(part), dtype=torch.long)
                         for part in (train_text, validation_text)]
    if min(len(train), len(validation)) <= context_size:
        raise ValueError("Corpus needs enough tokens for a context plus target in both splits")
    return train, validation, tokenizer


def load_cached_data(path, context_size, vocab_size=DEFAULT_VOCAB_SIZE, validation_path=None,
                     cache_dir="data/token-cache", tokenizer=None):
    """Encode corpus pairs into disk-backed tokens."""
    if validation_path is None:
        raise ValueError("Disk-backed training requires a separate validation file")
    splits, tokenizer = load_cached_files([path, validation_path], context_size, vocab_size, cache_dir, tokenizer)
    return *splits, tokenizer


def load_cached_files(paths, context_size, vocab_size, cache_dir, tokenizer=None):
    """Paths alternate training and validation; only training files teach BPE merges."""
    paths = [Path(path).resolve() for path in paths]
    if not paths or len(paths) % 2:
        raise ValueError("Supply one training/validation pair per source")
    if len(set(paths)) != len(paths):
        raise ValueError("Training and validation files must all differ")
    fingerprint = {"format": 1, "vocab_size": vocab_size, "files": [
        {"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
        for p in paths]}
    if tokenizer is not None:
        fingerprint["tokenizer"] = hashlib.sha256(tokenizer.to_json().encode()).hexdigest()
    key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:20]
    root = Path(cache_dir) / key
    root.mkdir(parents=True, exist_ok=True)
    tokenizer_path = root / "tokenizer.json"
    if tokenizer is not None:
        tokenizer_path.write_text(tokenizer.to_json())
    elif tokenizer_path.exists():
        tokenizer = BPETokenizer.from_json(tokenizer_path.read_text(encoding="utf-8"))
    else:
        tokenizer = BPETokenizer.train_file(paths[::2], vocab_size)
        temporary = tokenizer_path.with_suffix(".part")
        temporary.write_text(tokenizer.to_json(), encoding="utf-8")
        temporary.replace(tokenizer_path)
    dtype = np.uint16 if tokenizer.vocab_size <= 65536 else np.uint32
    splits = []
    for index, source in enumerate(paths):
        names = ("train", "validation", "extra_train", "extra_validation")
        split = names[index] if index < 4 else f"source_{index // 2}_{'train' if index % 2 == 0 else 'validation'}"
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
    return splits, tokenizer


def load_mixture(path, context_size, vocab_size=DEFAULT_VOCAB_SIZE, cache_dir="data/token-cache", tokenizer=None):
    manifest_path = Path(path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tokenizer is None and manifest.get("tokenizer"):
        tokenizer = BPETokenizer.from_json((manifest_path.parent / manifest["tokenizer"]).read_text())
        if tokenizer.vocab_size != vocab_size:
            raise ValueError("Requested vocabulary differs from prepared corpus tokenizer")
    records = manifest.get("sources", [])
    if not records:
        raise ValueError("Mixture must contain sources")
    names = [record["name"] for record in records]
    if len(set(names)) != len(names) or not all(name.isidentifier() for name in names):
        raise ValueError("Mixture source names must be unique identifiers")
    weights = {record["name"]: record["weight"] for record in records}
    # Validate weights before any expensive tokenizer training.
    WeightedData({name: [] for name in names}, weights)
    paths = [manifest_path.parent / record[split] for record in records for split in ("train", "validation")]
    splits, tokenizer = load_cached_files(paths, context_size, vocab_size, cache_dir, tokenizer)
    total = sum(map(len, splits[::2]))
    if not manifest.get("min_tokens", 0) <= total <= manifest.get("max_tokens", float("inf")):
        raise ValueError(f"Corpus has {total:,} tokens, outside manifest bounds")
    train = WeightedData(dict(zip(names, splits[::2])), weights)
    validation = WeightedData(dict(zip(names, splits[1::2])), weights)
    return train, validation, tokenizer


class WeightedData:
    """Sample fixed-length sequences by source weight, independent of source size."""

    def __init__(self, sources, weights):
        if (not sources or set(sources) != set(weights)
                or not all(math.isfinite(w) and w > 0 for w in weights.values())
                or not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-8)):
            raise ValueError("Source weights must be positive, finite, and sum to one")
        self.sources, self.weights = sources, weights

    def __len__(self):
        return sum(map(len, self.sources.values()))

    def batch(self, batch_size, context_size, device, generator=None):
        probabilities = torch.tensor([self.weights[name] for name in self.sources])
        selected = torch.multinomial(probabilities, batch_size, replacement=True, generator=generator)
        x = torch.empty((batch_size, context_size), dtype=torch.long)
        y = torch.empty_like(x)
        for index, data in enumerate(self.sources.values()):
            mask = selected == index
            count = int(mask.sum())
            if count:
                bx, by = get_batch(data, count, context_size, "cpu", generator)
                x[mask], y[mask] = bx, by
        return x.to(device), y.to(device)




class ShuffledWindows:
    """Non-overlapping windows visited once before repeating; evaluation uses random windows."""
    def __init__(self, data, context_size):
        self.data, self.context_size = data, context_size
        self.count = (len(data) - 1) // context_size
        self.order, self.cursor = [], 0
        if self.count < 1:
            raise ValueError("Source is shorter than one window")

    def __len__(self):
        return len(self.data)

    def batch(self, batch_size, context_size, device, generator=None):
        if generator is not None:
            return get_batch(self.data, batch_size, context_size, device, generator)
        selected = []
        while len(selected) < batch_size:
            if self.cursor == len(self.order):
                self.order, self.cursor = torch.randperm(self.count).tolist(), 0
            take = min(batch_size - len(selected), len(self.order) - self.cursor)
            selected.extend(self.order[self.cursor:self.cursor + take])
            self.cursor += take
        indices = torch.tensor(selected)[:, None] * context_size + torch.arange(context_size)
        if isinstance(self.data, np.ndarray):
            return (torch.from_numpy(self.data[indices.numpy()].astype(np.int64)).to(device),
                    torch.from_numpy(self.data[indices.numpy() + 1].astype(np.int64)).to(device))
        return self.data[indices].to(device), self.data[indices + 1].to(device)


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
    if isinstance(data, InstructionData):
        return data.batch(batch_size, device, generator)
    if isinstance(data, ShuffledWindows):
        return data.batch(batch_size, context_size, device, generator)
    if isinstance(data, WeightedData):
        return data.batch(batch_size, context_size, device, generator)
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
    target_tokens = 0
    for _ in range(accumulation_steps):
        x, y = get_batch(data, batch_size, model.config.context_size, device)
        target_tokens += int((y != -100).sum()) if isinstance(data, InstructionData) else y.numel()
        _, loss = model(x, y)
        (loss / accumulation_steps).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return target_tokens


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
    parser = argparse.ArgumentParser(description="Train the language model or fine-tune a checkpoint")
    parser.add_argument("--patience", type=int, help="Stop after this many evaluations without validation improvement")
    parser.add_argument("--init-from", help="Initialize weights and tokenizer; fresh optimizer and schedule")
    parser.add_argument("--task", choices=["language-model", "instruction"], default="language-model")
    parser.add_argument("--max-qa-passes", type=float, default=1.0)
    parser.add_argument("--sampling", choices=["random", "shuffled"], default="shuffled")
    parser.add_argument("--data", help="Optional custom training text")
    parser.add_argument("--data-mixture", help="JSON manifest of weighted training/validation sources; requires token-cache-dir")
    parser.add_argument("--validation-data", help="Separate validation text; otherwise use a 90/10 split")
    parser.add_argument("--token-cache-dir", help="Stream corpus into disk-backed tokens here; requires validation-data")
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--attention", choices=["mha", "mla"], default=None)
    parser.add_argument("--kv-rank", type=int, default=None, help="MLA latent width; 0 selects half model width")
    parser.add_argument("--out", help="Output checkpoint (default: books-200m-16l-16k.raw.pt)")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4,
                        help="Microbatches per optimizer step; increases effective batch without larger activations")
    parser.add_argument("--context-size", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--n-embd", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, help="Defaults to min(200, steps // 10)")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.patience is not None and args.patience < 1:
        parser.error("patience must be positive")
    instructing = args.task == "instruction"
    if instructing and (not args.init_from or not args.data or args.data_mixture or args.token_cache_dir):
        parser.error("Instruction training needs --init-from and --data JSONL, without mixture or token cache")
    if not math.isfinite(args.max_qa_passes) or args.max_qa_passes <= 0:
        parser.error("max-qa-passes must be positive and finite")
    model, tokenizer = load_model(args.init_from) if args.init_from else (None, None)
    defaults = dict(vocab_size=DEFAULT_VOCAB_SIZE, attention="mla", kv_rank=0, context_size=512,
                    n_layers=16, n_heads=4, n_embd=384, dropout=0.0)
    for key, default in defaults.items():
        value = getattr(model.config, key) if model is not None else default
        requested = getattr(args, key)
        if model is not None and requested is not None and requested != value:
            parser.error(f"{key} differs from the checkpoint ({value})")
        setattr(args, key, value if requested is None else requested)
    if not instructing and not args.data and not args.data_mixture:
        args.data_mixture = "data/expanded.json"
    if args.data_mixture:
        args.token_cache_dir = args.token_cache_dir or "data/token-cache"
        if args.data or args.validation_data:
            parser.error("data-mixture cannot be combined with data/validation-data")
    if args.token_cache_dir and not args.validation_data and not args.data_mixture:
        parser.error("token-cache-dir requires validation-data or data-mixture")
    args.out = args.out or ("checkpoints/books-200m-16l-16k.tuned.pt" if instructing else
                            "checkpoints/books-200m-16l-16k.raw.pt")
    if (min(args.steps, args.batch_size, args.grad_accum_steps, args.eval_every, args.eval_batches) < 1
            or not math.isfinite(args.lr) or args.lr <= 0):
        parser.error("steps, batch-size, grad-accum-steps, eval-every, eval-batches, and lr must be positive")
    warmup = min(200, args.steps // 10) if args.warmup_steps is None else args.warmup_steps
    if not 0 <= warmup < args.steps or not 0 <= args.min_lr_ratio <= 1:
        parser.error("warmup-steps must be in [0, steps); min-lr-ratio must be in [0, 1]")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight-decay must be finite and nonnegative")
    if args.init_from and Path(args.init_from).resolve() in {
            Path(args.out).resolve(), Path(args.out).with_name(Path(args.out).stem + ".best" + Path(args.out).suffix).resolve()}:
        parser.error("Output must differ from the parent checkpoint")
    torch.manual_seed(args.seed)
    device = choose_device() if args.device == "auto" else args.device
    print("Encoding with checkpoint tokenizer..." if args.init_from else
          "Training BPE tokenizer and encoding corpus...", flush=True)
    if instructing:
        train, validation, tokenizer = load_instruction_data(args.data, args.context_size, tokenizer, args.validation_data)
    elif args.data_mixture:
        train, validation, tokenizer = load_mixture(args.data_mixture, args.context_size,
                                                   args.vocab_size, args.token_cache_dir, tokenizer)
    elif args.token_cache_dir:
        train, validation, tokenizer = load_cached_data(args.data, args.context_size,
                                                       args.vocab_size, args.validation_data,
                                                       args.token_cache_dir, tokenizer)
    else:
        train, validation, tokenizer = load_data(args.data, args.context_size, args.vocab_size,
                                             args.validation_data, tokenizer)
    config = ModelConfig(vocab_size=tokenizer.vocab_size,
                         context_size=args.context_size, n_layers=args.n_layers,
                         n_heads=args.n_heads, n_embd=args.n_embd, dropout=args.dropout,
                         attention=args.attention, kv_rank=args.kv_rank)
    if instructing:
        limit = int(len(train) * args.max_qa_passes / (args.batch_size * args.grad_accum_steps))
        if limit < 1:
            parser.error("Q&A budget is smaller than one effective batch")
        args.steps = min(args.steps, limit)
        warmup = min(200, args.steps // 10) if args.warmup_steps is None else args.warmup_steps
        if warmup >= args.steps:
            parser.error("Warmup exceeds capped Q&A steps")
    elif args.sampling == "shuffled":
        if isinstance(train, WeightedData):
            train.sources = {k: ShuffledWindows(v, config.context_size) for k, v in train.sources.items()}
        else:
            train = ShuffledWindows(train, config.context_size)
    model = (model or LanguageModel(config)).to(device)
    model.train()
    optimizer = make_optimizer(model, args.lr, args.weight_decay)
    print(f"device={device} attention={args.attention} "
          f"parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
    train_tokens = train.token_count if instructing else len(train)
    validation_tokens = validation.token_count if instructing else len(validation)
    print(f"vocabulary={tokenizer.vocab_size:,} train_tokens={train_tokens:,} "
          f"validation_tokens={validation_tokens:,}", flush=True)
    tokens_per_step = args.batch_size * args.grad_accum_steps * config.context_size
    training_info = {"args": vars(args).copy(), "train_tokens": train_tokens,
                     "validation_tokens": validation_tokens, "tokens_per_step": tokens_per_step}
    training_info["init_from"] = str(Path(args.init_from).resolve()) if args.init_from else None
    if instructing:
        training_info.update(loss_mask="answer and EOS only", train_examples=len(train), validation_examples=len(validation))
        print(f"QA updates={args.steps} example_passes≈{args.steps * args.batch_size * args.grad_accum_steps / len(train):.2f}", flush=True)
    if isinstance(train, WeightedData):
        training_info["source_tokens"] = {name: len(data) for name, data in train.sources.items()}
        training_info["source_weights"] = train.weights.copy()
        for name, data in train.sources.items():
            passes = args.steps * tokens_per_step * train.weights[name] / len(data)
            print(f"source={name} share={train.weights[name]:.0%} tokens={len(data):,} expected_passes≈{passes:.2f}", flush=True)
    if not instructing:
        print(f"effective_batch={args.batch_size * args.grad_accum_steps} "
          f"tokens_per_step={tokens_per_step:,} token_budget={args.steps * tokens_per_step:,} "
          f"corpus_passes≈{args.steps * tokens_per_step / train_tokens:.2f}", flush=True)
    output = Path(args.out)
    best_output = output.with_name(output.stem + ".best" + output.suffix)
    best_loss = float("inf")
    stale = 0
    training_started = None
    target_tokens_seen = 0
    for step in range(args.steps + 1):
        lr = learning_rate(min(step, args.steps - 1), args.steps, warmup,
                           args.lr, args.min_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = lr
        if step % args.eval_every == 0 or step == args.steps:
            splits = {"train": train, "validation": validation}
            if isinstance(validation, WeightedData):
                splits = {"train": train, **{name + "_validation": data for name, data in validation.sources.items()}}
            losses = evaluate(model, splits,
                              args.batch_size, device, args.eval_batches)
            if isinstance(validation, WeightedData):
                losses["validation"] = sum(validation.weights[name] * losses[name + "_validation"]
                                            for name in validation.sources)
            eta = "--"
            if training_started is not None:
                remaining = (time.perf_counter() - training_started) / step * (args.steps - step)
                eta = str(timedelta(seconds=round(remaining)))
            print(f"step={step} train={losses['train']:.4f} validation={losses['validation']:.4f} "
                  f"gap={losses['validation'] - losses['train']:.4f} "
                  f"tokens_seen={step * tokens_per_step:,} target_tokens_seen={target_tokens_seen:,} lr={lr:.2e} eta={eta}",
                  flush=True)
            if isinstance(validation, WeightedData):
                print(" ".join(f"{name}_validation={losses[name + '_validation']:.4f}"
                               for name in validation.sources), flush=True)
            training_info["target_tokens_seen"] = target_tokens_seen
            if losses["validation"] < best_loss:
                best_loss = losses["validation"]
                stale = 0
                save_checkpoint(best_output, model, tokenizer, step, best_loss, training_info)
            else:
                stale += 1
            if args.patience and stale >= args.patience:
                print(f"Early stopping at step {step}; best validation={best_loss:.4f}", flush=True)
                break
        if step == args.steps:
            break
        if training_started is None:
            training_started = time.perf_counter()
        target_tokens_seen += train_step(model, train, optimizer, args.batch_size, device, args.grad_accum_steps)
    save_checkpoint(output, model, tokenizer, step, losses["validation"], training_info)
    print(f"Saved {output}; best validation={best_loss:.4f}: {best_output}")


if __name__ == "__main__":
    main()
