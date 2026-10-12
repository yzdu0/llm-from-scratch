import argparse

import torch

from .model import LanguageModel, ModelConfig
from .tokenizer import ByteTokenizer
from .train import choose_device


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = LanguageModel(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint")
    parser.add_argument("--checkpoint", default="checkpoints/model.pt")
    parser.add_argument("--prompt", default="The ")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.prompt:
        parser.error("prompt must be nonempty")
    torch.manual_seed(args.seed)
    device = choose_device() if args.device == "auto" else args.device
    tokenizer = ByteTokenizer()
    model = load_model(args.checkpoint, device)
    prompt = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    result = model.generate(prompt, args.tokens, args.temperature, args.top_k)
    print(tokenizer.decode(result[0].tolist()))


if __name__ == "__main__":
    main()
