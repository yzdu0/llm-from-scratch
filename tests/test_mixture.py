import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from llm.book_catalog import FICTION_BOOKS, EDUCATION_BOOKS
from llm.tokenizer import BPETokenizer, EOS_TOKEN
from llm.train import WeightedData, get_batch, load_mixture


class MixtureTests(unittest.TestCase):
    def test_balanced_sampling_matches_shares_and_keeps_windows_in_one_source(self):
        weights = dict(zip(("fiction", "education"), (.8, .2)))
        sources = {name: torch.full((40,), index) for index, name in enumerate(weights)}
        data = WeightedData(sources, weights)
        x, y = get_batch(data, 10000, 8, "cpu", torch.Generator().manual_seed(42))
        torch.testing.assert_close(x, y)
        self.assertTrue((x == x[:, :1]).all().item())
        for index, weight in enumerate(weights.values()):
            actual = (x[:, 0] == index).float().mean().item()
            self.assertAlmostEqual(actual, weight, delta=.015)
        same, _ = get_batch(data, 10000, 8, "cpu", torch.Generator().manual_seed(42))
        torch.testing.assert_close(x, same)
        for invalid in ({"fiction": 0}, {"fiction": float("nan")}, {"fiction": .9}):
            with self.assertRaises(ValueError):
                WeightedData({"fiction": sources["fiction"]}, invalid)

    def test_manifest_loads_all_pairs_and_trains_shared_tokenizer_without_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            expected_training = []
            for index, name in enumerate(("fiction", "education")):
                train, valid = root / f"{name}.txt", root / f"{name}-valid.txt"
                train.write_text((f"Training source {index}.\n{EOS_TOKEN}\n") * 10)
                valid.write_text((f"Validation source {index}.\n{EOS_TOKEN}\n") * 10)
                sources.append({"name": name, "weight": .5, "train": train.name, "validation": valid.name})
                expected_training.append(train.resolve())
            manifest = root / "mixture.json"
            manifest.write_text(json.dumps({"sources": sources}))
            with patch.object(BPETokenizer, "train_file", wraps=BPETokenizer.train_file) as learner:
                train, valid, tokenizer = load_mixture(manifest, 8, 300, root / "cache")
            self.assertEqual(learner.call_args.args[0], expected_training)
            for record in sources:
                name = record["name"]
                self.assertEqual(tokenizer.decode(train.sources[name].tolist()), (root / record["train"]).read_text())
                self.assertEqual(tokenizer.decode(valid.sources[name].tolist()), (root / record["validation"]).read_text())
            sources[0]["validation"] = sources[1]["train"]
            manifest.write_text(json.dumps({"sources": sources}))
            with self.assertRaises(ValueError):
                load_mixture(manifest, 8, 300, root / "cache")


    def test_catalogs_have_unique_editions_and_keep_original_fiction_held_out(self):
        entries = list(FICTION_BOOKS) + list(EDUCATION_BOOKS)
        ids = [entry[0] for entry in entries]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreaterEqual(len(FICTION_BOOKS), 100)
        original = {209: "validation", 5230: "validation", 105: "validation", 345: "train", 84: "train"}
        current = {book[0]: book[3] for book in FICTION_BOOKS}
        for ebook, split in original.items():
            self.assertEqual(current[ebook], split)


if __name__ == "__main__":
    unittest.main()
