from dataclasses import asdict
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from llm.generate import load_model
from llm.model import LanguageModel, ModelConfig, SwiGLU, apply_rope
from llm.prepare import clean_gutenberg, prepare_darkfiction, read_stories
from llm.tokenizer import BPETokenizer, EOS_TOKEN
from llm.train import get_batch, learning_rate, load_data, make_optimizer, save_checkpoint, train_step


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

    def test_repetition_penalty_can_break_a_greedy_repeat(self):
        logits = torch.full((1, 1, 256), -10.)
        logits[0, 0, 1], logits[0, 0, 2] = 2., 1.5
        prompt = torch.tensor([[1, 1]])
        with patch.object(self.model, "forward", return_value=(logits, None)):
            plain = self.model.generate(prompt, 1, temperature=0, use_cache=False)
            penalized = self.model.generate(prompt, 1, temperature=0, use_cache=False,
                                            repetition_penalty=2)
        self.assertEqual(plain[0, -1].item(), 1)
        self.assertEqual(penalized[0, -1].item(), 2)
        logits[0, 0, 1] = 4.
        with patch.object(self.model, "forward", return_value=(logits, None)):
            once = self.model.generate(prompt, 1, temperature=0, use_cache=False,
                                      repetition_penalty=2)
        self.assertEqual(once[0, -1].item(), 1)
        logits[0, 0, 1], logits[0, 0, 2] = -1., -1.5
        with patch.object(self.model, "forward", return_value=(logits, None)):
            penalized = self.model.generate(prompt, 1, temperature=0, use_cache=False,
                                            repetition_penalty=2)
        self.assertEqual(penalized[0, -1].item(), 2)

    def test_nucleus_sampling_keeps_crossing_token_and_at_least_one(self):
        logits = torch.full((2, 1, 256), float("-inf"))
        logits[:, 0, :3] = torch.tensor([[0.6, 0.25, 0.15], [0.15, 0.25, 0.6]]).log()
        prompt = torch.tensor([[1], [1]])
        with patch.object(self.model, "forward", return_value=(logits, None)):
            for top_p, expected in [
                (0.7, [[0.6 / 0.85, 0.25 / 0.85, 0.], [0., 0.25 / 0.85, 0.6 / 0.85]]),
                (0.01, [[1., 0., 0.], [0., 0., 1.]]),
                (1., [[0.6, 0.25, 0.15], [0.15, 0.25, 0.6]]),
            ]:
                with patch("torch.multinomial", return_value=torch.tensor([[0], [2]])) as sample:
                    self.model.generate(prompt, 1, temperature=1, top_k=None,
                                        top_p=top_p, use_cache=False)
                probabilities = sample.call_args.args[0]
                torch.testing.assert_close(probabilities[:, :3], torch.tensor(expected))
                self.assertEqual(probabilities[:, 3:].count_nonzero().item(), 0)

    def test_accumulated_update_matches_a_full_batch(self):
        full_model = LanguageModel(self.config)
        full_model.load_state_dict(self.model.state_dict())
        microbatches = [get_batch(torch.arange(40), 2, 8, "cpu") for _ in range(2)]
        optimizer = make_optimizer(self.model, 3e-4, 0.1)
        full_optimizer = make_optimizer(full_model, 3e-4, 0.1)
        with patch("llm.train.get_batch", side_effect=microbatches):
            train_step(self.model, None, optimizer, 2, "cpu", accumulation_steps=2)
        x, y = [torch.cat(parts) for parts in zip(*microbatches)]
        with patch("llm.train.get_batch", return_value=(x, y)):
            train_step(full_model, None, full_optimizer, 4, "cpu")
        for accumulated, full in zip(self.model.parameters(), full_model.parameters()):
            torch.testing.assert_close(accumulated, full, atol=1e-6, rtol=1e-5)

    def test_weight_decay_excludes_biases_and_normalization_and_deduplicates_tied_weights(self):
        optimizer = make_optimizer(self.model, 3e-4, 0.1)
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(parameters), len({id(p) for p in self.model.parameters()}))
        self.assertEqual(len(parameters), len({id(p) for p in parameters}))
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                self.assertEqual(group["weight_decay"], 0.1 if parameter.ndim >= 2 else 0)

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
        for penalty in (0.9, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.model.generate(torch.ones((1, 1), dtype=torch.long), repetition_penalty=penalty)
        for top_p in (0, -0.1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.model.generate(torch.ones((1, 1), dtype=torch.long), top_p=top_p)
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
            save_checkpoint(path, self.model, tokenizer, 100, 3.5,
                            {"tokens_per_step": 64, "args": {"grad_accum_steps": 2}})
            self.assertTrue(self.model.training)
            checkpoint = torch.load(path, weights_only=True)
            self.assertEqual(checkpoint["step"], 100)
            self.assertEqual(checkpoint["validation_loss"], 3.5)
            self.assertEqual(checkpoint["training_info"]["tokens_per_step"], 64)
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

    def test_darkfiction_mixing_preserves_inputs_and_separates_held_out_work(self):
        books = ((1, "Training work", "Author A", "train"),
                 (2, "Held-out work", "Author B", "validation"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            originals = root / "darkfiction-sources"
            originals.mkdir()
            for ebook, title, _, _ in books:
                text = (f"License\n*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
                        f"{title}\nSome prose.\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\nLicense")
                (originals / f"{ebook}.txt").write_text(text, encoding="utf-8")
            train, valid = root / "train.txt", root / "valid.txt"
            train.write_text("Training story.\n" + EOS_TOKEN + "\n", encoding="utf-8")
            valid.write_text("Validation story.\n" + EOS_TOKEN + "\n", encoding="utf-8")
            before = (train.read_bytes(), valid.read_bytes())
            output = root / "mixed.txt"
            with patch("llm.prepare.DARK_BOOKS", books), patch("llm.prepare.urlopen") as network:
                prepare_darkfiction(output, train, valid)
                network.assert_not_called()
            training = output.read_text(encoding="utf-8")
            validation = (root / "mixed-valid.txt").read_text(encoding="utf-8")
            self.assertIn("Training work", training)
            self.assertNotIn("Held-out work", training)
            self.assertIn("Held-out work", validation)
            self.assertNotIn("Training work", validation)
            self.assertEqual((train.read_bytes(), valid.read_bytes()), before)
            self.assertEqual(training.count(EOS_TOKEN), 2)
            self.assertTrue(output.with_suffix(".json").exists())
            with self.assertRaises(ValueError):
                prepare_darkfiction(train, train, valid)
            with self.assertRaises(ValueError):
                prepare_darkfiction(output, train)


if __name__ == "__main__":
    unittest.main()
