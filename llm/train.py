import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from .model import LanguageModel, ModelConfig
from .tokenizer import BPETokenizer


def choose_device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_data(path, context_size, vocab_size=2048):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found: {path}. Run python -m llm.prepare first.")
    text = path.read_text(encoding="utf-8")
    # Split before learning merges, so the tokenizer never trains on validation text.
    split = text.rfind("\n", 0, int(0.9 * len(text)))
    if split <= 0:
        split = int(0.9 * len(text))
    else:
        split += 1
    tokenizer = BPETokenizer.train(text[:split], vocab_size)
    train, validation = [torch.tensor(tokenizer.encode(part), dtype=torch.long)
                         for part in (text[:split], text[split:])]
    if min(len(train), len(validation)) <= context_size:
        raise ValueError("Corpus needs enough tokens for a context plus target in both 90/10 splits")
    return train, validation, tokenizer


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
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--out", default="checkpoints/model.pt")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-size", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.eval_every, args.eval_batches) < 1 or args.lr <= 0:
        parser.error("steps, batch-size, eval-every, eval-batches, and lr must be positive")
    torch.manual_seed(args.seed)
    device = choose_device() if args.device == "auto" else args.device
    print("Training BPE tokenizer and encoding corpus...", flush=True)
    train, validation, tokenizer = load_data(args.data, args.context_size, args.vocab_size)
    config = ModelConfig(vocab_size=tokenizer.vocab_size,
                         context_size=args.context_size, n_layers=args.n_layers,
                         n_heads=args.n_heads, n_embd=args.n_embd, dropout=args.dropout)
    model = LanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"device={device} parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"vocabulary={tokenizer.vocab_size:,} train_tokens={len(train):,} "
          f"validation_tokens={len(validation):,}", flush=True)
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            losses = evaluate(model, {"train": train, "validation": validation},
                              args.batch_size, device, args.eval_batches)
            print(f"step={step} train={losses['train']:.4f} validation={losses['validation']:.4f}",
                  flush=True)
        if step == args.steps:
            break
        x, y = get_batch(train, args.batch_size, config.context_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(config), "model": model.cpu().state_dict(),
                "step": args.steps, "tokenizer": tokenizer.to_json()}, output)
    output.with_suffix(".tokenizer.json").write_text(tokenizer.to_json(), encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
