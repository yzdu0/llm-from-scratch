from dataclasses import fields
import torch
from .model import LanguageModel, ModelConfig
from .tokenizer import BPETokenizer


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

