import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from llm.prepare import prepare_books, prepare_stories, write_stories
from llm.tokenizer import BPETokenizer, EOS_TOKEN
from llm.train import MixedData, get_batch, load_cached_data


class StoryDataTests(unittest.TestCase):
    def test_books_corpus_keeps_whole_validation_works_separate(self):
        catalog = ((1, "Dark training novel", "Author A", "train"),
                   (2, "Held-out novel", "Author B", "validation"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = root / "books-sources"
            sources.mkdir()
            for ebook, title, _, _ in catalog:
                text = ("License\n*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
                        + title + "\nA passage.\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\nLicense")
                (sources / f"{ebook}.txt").write_text(text)
            output = root / "books.txt"
            with patch("llm.prepare.BOOKS", catalog), patch("llm.prepare.urlopen") as network:
                prepare_books(output)
                network.assert_not_called()
            train = output.read_text()
            valid = (root / "books-valid.txt").read_text()
            self.assertIn("Dark training novel", train)
            self.assertNotIn("Held-out novel", train)
            self.assertIn("Held-out novel", valid)
            self.assertNotIn("Dark training novel", valid)
            self.assertEqual(train.count(EOS_TOKEN), 1)
            self.assertEqual(valid.count(EOS_TOKEN), 1)
            metadata = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(metadata["train_books"], 1)
            self.assertEqual(metadata["validation_books"], 1)
            original = output.read_bytes()
            (sources / "2.txt").write_text("Invalid source")
            with patch("llm.prepare.BOOKS", catalog):
                with self.assertRaises(ValueError):
                    prepare_books(output)
            self.assertEqual(output.read_bytes(), original)
            self.assertFalse(list(root.glob("*.part")))

    def test_mixed_batches_preserve_sources_targets_and_seed(self):
        data = MixedData(torch.full((50,), 1), torch.full((50,), 2), 0.25)
        x, y = get_batch(data, 1000, 8, "cpu", torch.Generator().manual_seed(42))
        repeated, _ = get_batch(data, 1000, 8, "cpu", torch.Generator().manual_seed(42))
        torch.testing.assert_close(x, y)
        torch.testing.assert_close(x, repeated)
        self.assertTrue(((x == x[:, :1]).all()).item())
        fraction = (x[:, 0] == 2).float().mean().item()
        self.assertTrue(0.20 < fraction < 0.30)
        for ratio in (0, 1, float("nan")):
            with self.assertRaises(ValueError):
                MixedData(data.primary, data.extra, ratio)

    def test_mixed_cache_uses_both_training_files_and_keeps_validation_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / name for name in ("story.txt", "story-valid.txt", "dark.txt", "dark-valid.txt")]
            texts = [(word + "\n" + EOS_TOKEN + "\n") * 30
                     for word in ("Lily played.", "Jack laughed.", "A shadow lurked.", "A haunted room.")]
            for path, text in zip(paths, texts):
                path.write_text(text)
            with patch.object(BPETokenizer, "train_file", wraps=BPETokenizer.train_file) as train_tokenizer:
                train, valid, tokenizer = load_cached_data(paths[0], 8, 300, paths[1], root / "cache",
                                                            paths[2], paths[3], 0.1)
            self.assertEqual(train_tokenizer.call_args.args[0], [paths[0].resolve(), paths[2].resolve()])
            for data, text in zip((train.primary, valid.primary, train.extra, valid.extra), texts):
                self.assertEqual(tokenizer.decode(data.tolist()), text)
            self.assertEqual(train.ratio, 0.1)

    def test_streaming_all_and_subset_preserve_story_endings(self):
        raw = f"First story.\n{EOS_TOKEN}\n\nSecond story.\n{EOS_TOKEN}\n".encode()
        output = io.StringIO()
        self.assertEqual(write_stories(io.BytesIO(raw), output), 2)
        self.assertEqual(output.getvalue(), raw.decode())
        stream = io.BytesIO(raw)
        output = io.StringIO()
        self.assertEqual(write_stories(stream, output, 1), 1)
        self.assertEqual(output.getvalue(), f"First story.\n{EOS_TOKEN}\n")
        self.assertIn(b"Second story.", stream.read())
        for invalid in (b"", b"Partial", raw + b"Partial"):
            with self.assertRaises(ValueError):
                write_stories(io.BytesIO(invalid), io.StringIO())
        output = io.StringIO()
        self.assertEqual(write_stories(io.BytesIO(raw + b"Partial"), output,
                                       allow_incomplete_tail=True), 2)
        self.assertEqual(output.getvalue(), raw.decode())

    def test_gpt4_preparation_uses_official_splits_and_records_actual_counts(self):
        raw = f"A complete story.\n{EOS_TOKEN}\n".encode()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "gpt4.txt"
            with patch("llm.prepare.urlopen", side_effect=[io.BytesIO(raw * 2), io.BytesIO(raw)]) as network:
                prepare_stories(output, None, None, "gpt4")
            urls = [call.args[0] for call in network.call_args_list]
            self.assertTrue(urls[0].endswith("TinyStoriesV2-GPT4-train.txt"))
            self.assertTrue(urls[1].endswith("TinyStoriesV2-GPT4-valid.txt"))
            metadata = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(metadata["train_stories"], 2)
            self.assertEqual(metadata["validation_stories"], 1)
            self.assertEqual(metadata["version"], "gpt4")
            before = output.read_bytes()
            with patch("llm.prepare.urlopen", side_effect=[io.BytesIO(raw), io.BytesIO(b"Partial")]):
                with self.assertRaises(ValueError):
                    prepare_stories(output, None, None, "gpt4")
            self.assertEqual(output.read_bytes(), before)
            self.assertFalse(list(Path(directory).glob("*.part")))
            truncated = io.BytesIO(raw)
            truncated.headers = {"Content-Length": str(len(raw) + 10)}
            with patch("llm.prepare.urlopen", return_value=truncated):
                with self.assertRaises(OSError):
                    prepare_stories(output, None, None, "gpt4")
            self.assertEqual(output.read_bytes(), before)

    def test_disk_tokens_roundtrip_shift_reuse_and_invalidation(self):
        train_text = ("Lily played outside.\n" + EOS_TOKEN + "\n\n") * 20
        validation_text = ("Jack saw a bird.\n" + EOS_TOKEN + "\n\n") * 10
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, valid = root / "train.txt", root / "valid.txt"
            train.write_text(train_text)
            valid.write_text(validation_text)
            args = (train, 8, 300, valid, root / "cache")
            training, validation, tokenizer = load_cached_data(*args)
            self.assertIsInstance(training, np.memmap)
            self.assertEqual(tokenizer.decode(training.tolist()), train_text)
            self.assertEqual(tokenizer.decode(validation.tolist()), validation_text)
            self.assertEqual(tokenizer.encode(EOS_TOKEN), [tokenizer.eos_token_id])
            x, y = get_batch(training, 3, 8, "cpu", torch.Generator().manual_seed(3))
            torch.testing.assert_close(x[:, 1:], y[:, :-1])
            self.assertEqual(x.dtype, torch.long)
            with patch.object(BPETokenizer, "train_file", side_effect=AssertionError("Unexpected retraining")):
                reloaded, _, _ = load_cached_data(*args)
            np.testing.assert_array_equal(training, reloaded)
            train.write_text(train_text + "A new story.\n" + EOS_TOKEN + "\n")
            changed, _, _ = load_cached_data(*args)
            self.assertNotEqual(training.filename, changed.filename)
            with self.assertRaises(ValueError):
                load_cached_data(train, 8, 300)
            with self.assertRaises(ValueError):
                load_cached_data(train, 8, 300, train)


if __name__ == "__main__":
    unittest.main()
