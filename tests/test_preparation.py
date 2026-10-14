import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llm.prepare import prepare_books, varied_fiction
from llm.tokenizer import EOS_TOKEN


class PreparationTests(unittest.TestCase):
    def test_short_story_collections_are_interleaved_without_repetition(self):
        rows=[{'Text#': str(i), 'Subjects': 'Short stories' if i < 3 else 'Adventure fiction'} for i in range(9)]
        ordered=list(varied_fiction(rows))
        self.assertEqual([r['Text#'] for r in ordered],['0','3','4','1','5','6','2','7','8'])
        self.assertEqual(len({r['Text#'] for r in ordered}),len(rows))

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
            with patch("llm.prepare.FICTION_BOOKS", catalog), patch("llm.prepare.urlopen") as network:
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
            with patch("llm.prepare.FICTION_BOOKS", catalog):
                with self.assertRaises(ValueError):
                    prepare_books(output)
            self.assertEqual(output.read_bytes(), original)
            self.assertFalse(list(root.glob("*.part")))


if __name__ == "__main__":
    unittest.main()
