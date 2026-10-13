import argparse
from dataclasses import fields

import torch

from .model import LanguageModel, ModelConfig
from .tokenizer import BPETokenizer
from .train import choose_device
from .translation import translation_prompt


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "tokenizer" not in checkpoint:
        raise ValueError("Checkpoint has no BPE tokenizer. Retrain with python -m llm.train.")
    tokenizer = BPETokenizer.from_json(checkpoint["tokenizer"])
    config = dict(checkpoint["config"])
    # Checkpoints predating these fields used learned positions and GELU.
    config.setdefault("position_encoding", "learned")
    config.setdefault("feed_forward", "gelu")
    supported = {field.name for field in fields(ModelConfig)}
    model = LanguageModel(ModelConfig(**{key: value for key, value in config.items() if key in supported}))
    if model.config.vocab_size != tokenizer.vocab_size:
        raise ValueError("Checkpoint model and tokenizer vocabulary sizes do not match")
    try:
        model.load_state_dict(checkpoint["model"])
    except RuntimeError as error:
        raise ValueError("Checkpoint architecture is incompatible with this version; retrain the model.") from error
    return model.to(device).eval(), tokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint")
    parser.add_argument("--checkpoint", default="checkpoints/model.pt")
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument("--prompt", default="The ")
    prompts.add_argument("--translate", help="English sentence to translate; prints only Chinese")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, help="Default: 0.7 (translation: greedy)")
    parser.add_argument("--top-k", type=int, default=40, help="Candidate cap; 0 disables it")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Keep this probability mass after top-k filtering; 1 disables it")
    parser.add_argument("--repetition-penalty", type=float,
                        help="Discourage previously used tokens; 1 disables the penalty")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cache", action="store_true", help="Recompute the context each step")
    args = parser.parse_args()
    if not args.prompt:
        parser.error("prompt must be nonempty")
    torch.manual_seed(args.seed)
    device = choose_device() if args.device == "auto" else args.device
    model, tokenizer = load_model(args.checkpoint, device)
    translating = args.translate is not None
    text = translation_prompt(args.translate) if translating else args.prompt
    temperature = args.temperature if args.temperature is not None else (0.0 if translating else 0.7)
    penalty = args.repetition_penalty if args.repetition_penalty is not None else 1.0
    prompt = torch.tensor([tokenizer.encode(text)], dtype=torch.long, device=device)
    if translating and (tokenizer.eos_token_id is None or prompt.size(1) >= model.config.context_size):
        parser.error("Translation needs an EOS tokenizer and an English prompt shorter than the context")
    result = model.generate(prompt, args.tokens, temperature, args.top_k or None,
                            eos_token_id=tokenizer.eos_token_id, use_cache=not args.no_cache,
                            repetition_penalty=penalty, top_p=args.top_p)
    output = result[0, prompt.size(1):] if translating else result[0]
    print(tokenizer.decode(output.tolist(), skip_special_tokens=True))


if __name__ == "__main__":
    main()
