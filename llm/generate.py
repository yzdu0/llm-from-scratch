import argparse

import torch

from .checkpoint import load_model
from .instruction import instruction_prompt
from .train import choose_device




def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint")
    parser.add_argument("--checkpoint", default="checkpoints/books-100m-16l.raw.best.pt")
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument("--prompt", default="The ")
    prompts.add_argument("--question", help="Question for an instruction-tuned checkpoint; print answer only")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, help="Default: 0.7")
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
    text = instruction_prompt(args.question) if args.question is not None else args.prompt
    temperature = args.temperature if args.temperature is not None else (0.0 if args.question is not None else 0.7)
    penalty = args.repetition_penalty if args.repetition_penalty is not None else 1.0
    prompt = torch.tensor([tokenizer.encode(text)], dtype=torch.long, device=device)
    if args.question is not None and (tokenizer.eos_token_id is None or prompt.size(1) >= model.config.context_size):
        parser.error("Question needs an EOS tokenizer and must fit the checkpoint context")
    result = model.generate(prompt, args.tokens, temperature, args.top_k or None,
                            eos_token_id=tokenizer.eos_token_id, use_cache=not args.no_cache,
                            repetition_penalty=penalty, top_p=args.top_p)
    output = result[0, prompt.size(1):] if args.question is not None else result[0]
    print(tokenizer.decode(output.tolist(), skip_special_tokens=True))


if __name__ == "__main__":
    main()
