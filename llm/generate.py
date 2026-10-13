import argparse

import torch

from .model import LanguageModel, ModelConfig
from .tokenizer import BPETokenizer
from .train import choose_device


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "tokenizer" not in checkpoint:
        raise ValueError("Checkpoint has no BPE tokenizer. Retrain with python -m llm.train.")
    tokenizer = BPETokenizer.from_json(checkpoint["tokenizer"])
    config = dict(checkpoint["config"])
    # Checkpoints predating these fields used learned positions and GELU.
    config.setdefault("position_encoding", "learned")
    config.setdefault("feed_forward", "gelu")
    model = LanguageModel(ModelConfig(**config))
    if model.config.vocab_size != tokenizer.vocab_size:
        raise ValueError("Checkpoint model and tokenizer vocabulary sizes do not match")
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), tokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint")
    parser.add_argument("--checkpoint", default="checkpoints/model.pt")
    parser.add_argument("--prompt", default="The ")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cache", action="store_true", help="Recompute the context each step")
    args = parser.parse_args()
    if not args.prompt:
        parser.error("prompt must be nonempty")
    torch.manual_seed(args.seed)
    device = choose_device() if args.device == "auto" else args.device
    model, tokenizer = load_model(args.checkpoint, device)
    prompt = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    result = model.generate(prompt, args.tokens, args.temperature, args.top_k,
                            eos_token_id=tokenizer.eos_token_id, use_cache=not args.no_cache)
    print(tokenizer.decode(result[0].tolist(), skip_special_tokens=True))


if __name__ == "__main__":
    main()
