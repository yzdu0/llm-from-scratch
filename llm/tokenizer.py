from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


class BPETokenizer:
    """Learn subword merges with a byte alphabet for unseen Unicode text."""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def train(cls, text: str, vocab_size: int = 8192):
        if not text or vocab_size < 256:
            raise ValueError("BPE training needs nonempty text and vocab_size >= 256")
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, min_frequency=2, show_progress=False,
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

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=False)
