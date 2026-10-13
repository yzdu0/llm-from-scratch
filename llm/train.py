import argparse
from dataclasses import asdict
import math
from pathlib import Path

import torch

from .model import LanguageModel, ModelConfig
from .tokenizer import BPETokenizer


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


def learning_rate(step, total_steps, warmup_steps, peak_lr, min_ratio=0.1):
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
    progress = min(max(progress, 0.0), 1.0)
    return peak_lr * (min_ratio + (1 - min_ratio) * (1 + math.cos(math.pi * progress)) / 2)


def save_checkpoint(path, model, tokenizer, step, validation_loss):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    torch.save({"config": asdict(model.config), "model": state, "step": step,
                "validation_loss": validation_loss, "tokenizer": tokenizer.to_json()}, path)
    path.with_suffix(".tokenizer.json").write_text(tokenizer.to_json(), encoding="utf-8")


def get_batch(data, batch_size, context_size, device, generator=None):
    starts = torch.randint(len(data) - context_size, (batch_size,), generator=generator)
    offsets = starts[:, None] + torch.arange(context_size)
    return data[offsets].to(device), data[offsets + 1].to(device)


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
    parser.add_argument("--data", default="data/shakespeare.txt")
    parser.add_argument("--validation-data", help="Separate validation text; otherwise use a 90/10 split")
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--architecture", choices=["dense", "deepseek"], default="dense")
    parser.add_argument("--kv-rank", type=int, default=0, help="MLA latent width; 0 selects half model width")
    parser.add_argument("--n-experts", type=int, default=4)
    parser.add_argument("--experts-per-token", type=int, default=2)
    parser.add_argument("--mtp-weight", type=float, default=0.1, help="DeepSeek-profile second-token loss weight")
    parser.add_argument("--out", default="checkpoints/model.pt")
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-size", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, help="Defaults to min(200, steps // 10)")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (min(args.steps, args.batch_size, args.eval_every, args.eval_batches) < 1
            or not math.isfinite(args.lr) or args.lr <= 0):
        parser.error("steps, batch-size, eval-every, eval-batches, and lr must be positive")
    warmup = min(200, args.steps // 10) if args.warmup_steps is None else args.warmup_steps
    if not 0 <= warmup < args.steps or not 0 <= args.min_lr_ratio <= 1:
        parser.error("warmup-steps must be in [0, steps); min-lr-ratio must be in [0, 1]")
    torch.manual_seed(args.seed)
    device = choose_device() if args.device == "auto" else args.device
    print("Training BPE tokenizer and encoding corpus...", flush=True)
    train, validation, tokenizer = load_data(args.data, args.context_size, args.vocab_size,
                                            args.validation_data)
    deepseek = args.architecture == "deepseek"
    config = ModelConfig(vocab_size=tokenizer.vocab_size,
                         context_size=args.context_size, n_layers=args.n_layers,
                         n_heads=args.n_heads, n_embd=args.n_embd, dropout=args.dropout,
                         attention="mla" if deepseek else "mha",
                         normalization="rmsnorm" if deepseek else "layernorm",
                         kv_rank=args.kv_rank, n_experts=args.n_experts if deepseek else 0,
                         experts_per_token=args.experts_per_token,
                         mtp_weight=args.mtp_weight if deepseek else 0)
    model = LanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"device={device} architecture={args.architecture} "
          f"parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"vocabulary={tokenizer.vocab_size:,} train_tokens={len(train):,} "
          f"validation_tokens={len(validation):,}", flush=True)
    output = Path(args.out)
    best_output = output.with_name(output.stem + ".best" + output.suffix)
    best_loss = float("inf")
    for step in range(args.steps + 1):
        lr = learning_rate(min(step, args.steps - 1), args.steps, warmup,
                           args.lr, args.min_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = lr
        if step % args.eval_every == 0 or step == args.steps:
            losses = evaluate(model, {"train": train, "validation": validation},
                              args.batch_size, device, args.eval_batches)
            print(f"step={step} train={losses['train']:.4f} validation={losses['validation']:.4f} "
                  f"lr={lr:.2e}",
                  flush=True)
            if losses["validation"] < best_loss:
                best_loss = losses["validation"]
                save_checkpoint(best_output, model, tokenizer, step, best_loss)
        if step == args.steps:
            break
        x, y = get_batch(train, args.batch_size, config.context_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.update_router_bias()
    save_checkpoint(output, model, tokenizer, args.steps, losses["validation"])
    print(f"Saved {output}; best validation={best_loss:.4f}: {best_output}")


if __name__ == "__main__":
    main()
