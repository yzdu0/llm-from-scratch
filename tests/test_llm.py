from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import torch

from llm.generate import load_model
from llm.model import LanguageModel, ModelConfig
from llm.prepare import clean_gutenberg
from llm.tokenizer import BPETokenizer
from llm.train import get_batch, load_data


class LanguageModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.config = ModelConfig(context_size=8, n_layers=1, n_heads=2,
                                  n_embd=16, dropout=0)
        self.model = LanguageModel(self.config)

    def test_tokenizer_roundtrip(self):
        tokenizer = BPETokenizer.train("To be, or not to be.\n" * 30, vocab_size=300)
        text = "Hello, \u4e16\u754c! \U0001f30d"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        self.assertEqual(tokenizer.encode(""), [])
        self.assertLess(len(tokenizer.encode("To be, or not to be.")),
                        len("To be, or not to be.".encode("utf-8")))
        restored = BPETokenizer.from_json(tokenizer.to_json())
        self.assertEqual(restored.encode(text), tokenizer.encode(text))

    def test_future_tokens_cannot_change_past_logits(self):
        original = torch.tensor([[1, 2, 3, 4, 5]])
        changed = torch.tensor([[1, 2, 3, 99, 100]])
        logits, _ = self.model(original)
        other, _ = self.model(changed)
        torch.testing.assert_close(logits[:, :3], other[:, :3])

    def test_batch_targets_are_shifted(self):
        x, y = get_batch(torch.arange(40), 4, 8, "cpu")
        torch.testing.assert_close(y, x + 1)

    def test_can_learn_a_repeating_sequence(self):
        x = torch.tensor([[1, 2, 1, 2, 1, 2, 1, 2]])
        y = 3 - x
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.02)
        _, initial = self.model(x, y)
        for _ in range(60):
            _, loss = self.model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        _, final = self.model(x, y)
        self.assertLess(final.item(), initial.item() * 0.2)
        self.assertEqual(self.model.generate(x[:, :1], 4, temperature=0).tolist(),
                         [[1, 2, 1, 2, 1]])

    def test_checkpoint_and_generation_beyond_context(self):
        tokenizer = BPETokenizer.train("To be, or not to be.", vocab_size=256)
        self.model.eval()
        prompt = torch.tensor([[1, 2]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"config": asdict(self.config), "model": self.model.state_dict(),
                        "tokenizer": tokenizer.to_json()}, path)
            restored, restored_tokenizer = load_model(path)
            self.assertEqual(restored_tokenizer.encode("To be"), tokenizer.encode("To be"))
            torch.testing.assert_close(self.model(prompt)[0], restored(prompt)[0])
            result = restored.generate(prompt, 20, temperature=0)
            self.assertEqual(result.shape, (1, 22))
            torch.testing.assert_close(result[:, :2], prompt)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            ModelConfig(n_embd=15, n_heads=2)
        with self.assertRaises(ValueError):
            self.model(torch.zeros((1, 9), dtype=torch.long))
        with self.assertRaises(ValueError):
            self.model.generate(torch.ones((1, 1), dtype=torch.long), temperature=-1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "short.txt"
            path.write_text("too short", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_data(path, 8)

    def test_legacy_checkpoint_requires_retraining(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.pt"
            torch.save({"config": asdict(self.config), "model": self.model.state_dict()}, path)
            with self.assertRaisesRegex(ValueError, "Retrain"):
                load_model(path)

    def test_tokenizer_only_learns_from_training_split(self):
        text = "king queen crown\n" * 200 + "validationuniqueword\n" * 10
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.txt"
            path.write_text(text, encoding="utf-8")
            train, validation, tokenizer = load_data(path, 8, 300)
            split = text.rfind("\n", 0, int(0.9 * len(text))) + 1
            self.assertEqual(tokenizer.decode(train.tolist()), text[:split])
            self.assertEqual(tokenizer.decode(validation.tolist()), text[split:])
            expected = BPETokenizer.train(text[:split], 300)
            self.assertEqual(tokenizer.tokenizer.get_vocab(), expected.tokenizer.get_vocab())

    def test_corpus_cleaning(self):
        raw = ("Header\r\n*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\r\n"
               "\r\nHAMLET\r\nTo be, or not to be.\r\n\r\n"
               "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\r\nLicense")
        self.assertEqual(clean_gutenberg(raw), "HAMLET\nTo be, or not to be.\n")
        with self.assertRaises(ValueError):
            clean_gutenberg("Not a Gutenberg ebook")


if __name__ == "__main__":
    unittest.main()
