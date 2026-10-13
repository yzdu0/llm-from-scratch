from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOS_TOKEN = "<|endoftext|>"

class BPETokenizer:
    """Learn subword merges with a byte alphabet for unseen Unicode text."""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def train(cls, text: str, vocab_size: int = 8192):
        special_tokens = [EOS_TOKEN] if EOS_TOKEN in text else []
        if not text or vocab_size < 256 + len(special_tokens):
            raise ValueError("BPE needs nonempty text and room for 256 bytes plus special tokens")
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, min_frequency=2, show_progress=False,
            special_tokens=special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )
        tokenizer.train_from_iterator(text.splitlines(keepends=True), trainer=trainer)
        return cls(tokenizer)

    @classmethod
    def from_json(cls, value: str):
        return cls(Tokenizer.from_str(value))

    def to_json(self) -> str:
        return self.tokenizer.to_str()

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    @property
    def eos_token_id(self):
        return self.tokenizer.token_to_id(EOS_TOKEN)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, tokens: list[int], skip_special_tokens=False) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
