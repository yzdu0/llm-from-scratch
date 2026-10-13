from dataclasses import asdict
import io
from pathlib import Path
import tempfile
import unittest

import torch

from llm.generate import load_model
from llm.model import LanguageModel, ModelConfig, SwiGLU, apply_rope
from llm.prepare import clean_gutenberg, read_stories
from llm.tokenizer import BPETokenizer, EOS_TOKEN
from llm.train import get_batch, learning_rate, load_data, save_checkpoint


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

    def test_rotary_positions_preserve_norm_and_relative_dot_product(self):
        attention = self.model.blocks[0].attention
        q = torch.randn(1, 1, 6, 8)
        k = torch.randn(1, 1, 6, 8)
        q[:] = q[:, :, :1].clone()
        k[:] = k[:, :, :1].clone()
        rotated_q = apply_rope(q, attention.rope_cos[:6], attention.rope_sin[:6])
        rotated_k = apply_rope(k, attention.rope_cos[:6], attention.rope_sin[:6])
        torch.testing.assert_close(rotated_q.norm(dim=-1), q.norm(dim=-1))
        torch.testing.assert_close((rotated_q[:, :, 1] * rotated_k[:, :, 3]).sum(),
                                   (rotated_q[:, :, 2] * rotated_k[:, :, 4]).sum())

    def test_swiglu_gate_controls_output(self):
        mlp = SwiGLU(self.config)
        with torch.no_grad():
            mlp.gate.weight.zero_()
        x = torch.randn(2, 8, 16)
        torch.testing.assert_close(mlp(x), torch.zeros_like(x))

    def test_learning_rate_warms_up_then_decays(self):
        rates = [learning_rate(step, 20, 4, 0.001) for step in range(20)]
        self.assertAlmostEqual(rates[0], 0.00025)
        self.assertAlmostEqual(rates[3], 0.001)
        self.assertAlmostEqual(rates[-1], 0.0001)
        self.assertTrue(all(a <= b for a, b in zip(rates[:3], rates[1:4])))
        self.assertTrue(all(a >= b for a, b in zip(rates[4:-1], rates[5:])))
        self.assertAlmostEqual(learning_rate(0, 20, 0, 0.001), 0.001)

    def test_generation_stops_at_story_end(self):
        with torch.no_grad():
            self.model.head.weight.zero_()
        result = self.model.generate(torch.tensor([[1]]), 20, temperature=0, eos_token_id=0)
        self.assertEqual(result.tolist(), [[1, 0]])
        self.assertTrue(self.model.training)

    def test_batch_targets_are_shifted(self):
        x, y = get_batch(torch.arange(40), 4, 8, "cpu")
        torch.testing.assert_close(y, x + 1)

    def test_can_learn_a_repeating_sequence(self):
        x = torch.tensor([[1, 2, 1, 2, 1, 2, 1, 2]])
        y = 3 - x
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.005)
        _, initial = self.model(x, y)
        for _ in range(120):
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

    def test_original_bpe_checkpoint_still_loads(self):
        config = ModelConfig(context_size=8, n_layers=1, n_heads=2, n_embd=16,
                             dropout=0, position_encoding="learned", feed_forward="gelu")
        original = LanguageModel(config).eval()
        old_config = asdict(config)
        del old_config["position_encoding"], old_config["feed_forward"]
        tokenizer = BPETokenizer.train("To be, or not to be.", vocab_size=256)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.pt"
            torch.save({"config": old_config, "model": original.state_dict(),
                        "tokenizer": tokenizer.to_json()}, path)
            restored, _ = load_model(path)
            prompt = torch.tensor([[1, 2, 3]])
            torch.testing.assert_close(original(prompt)[0], restored(prompt)[0])

    def test_saving_checkpoint_preserves_model_and_records_validation(self):
        tokenizer = BPETokenizer.train("To be, or not to be.", vocab_size=256)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, self.model, tokenizer, 100, 3.5)
            self.assertTrue(self.model.training)
            checkpoint = torch.load(path, weights_only=True)
            self.assertEqual(checkpoint["step"], 100)
            self.assertEqual(checkpoint["validation_loss"], 3.5)
            self.assertTrue(path.with_suffix(".tokenizer.json").exists())

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

    def test_separate_validation_and_story_token(self):
        train_text = ("The girl played with her dog.\n" + EOS_TOKEN + "\n") * 30
        validation_text = ("He baked fresh buns.\n" + EOS_TOKEN + "\n") * 10
        with tempfile.TemporaryDirectory() as directory:
            train_path, validation_path = [Path(directory) / name for name in ("train.txt", "valid.txt")]
            train_path.write_text(train_text, encoding="utf-8")
            validation_path.write_text(validation_text, encoding="utf-8")
            train, validation, tokenizer = load_data(train_path, 8, 300, validation_path)
            self.assertEqual(tokenizer.decode(train.tolist()), train_text)
            self.assertEqual(tokenizer.decode(validation.tolist()), validation_text)
            self.assertEqual(tokenizer.encode(EOS_TOKEN), [tokenizer.eos_token_id])
            self.assertNotIn(EOS_TOKEN, tokenizer.decode(validation.tolist(), skip_special_tokens=True))

    def test_corpus_cleaning(self):
        raw = ("Header\r\n*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\r\n"
               "\r\nHAMLET\r\nTo be, or not to be.\r\n\r\n"
               "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\r\nLicense")
        self.assertEqual(clean_gutenberg(raw), "HAMLET\nTo be, or not to be.\n")
        with self.assertRaises(ValueError):
            clean_gutenberg("Not a Gutenberg ebook")

    def test_story_subset_stops_at_complete_records(self):
        source = io.BytesIO(f"First story.\n{EOS_TOKEN}\nSecond story.\n{EOS_TOKEN}\nPartial".encode())
        text = read_stories(source, 1)
        self.assertEqual(text, f"First story.\n{EOS_TOKEN}\n")
        self.assertTrue(source.read().startswith(b"Second story."))
        with self.assertRaises(ValueError):
            read_stories(io.BytesIO(b"Incomplete story"), 1)


if __name__ == "__main__":
    unittest.main()
