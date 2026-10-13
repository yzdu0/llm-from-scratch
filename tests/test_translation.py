import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import torch
import torch.nn.functional as F

from llm.model import LanguageModel, ModelConfig
from llm.generate import main as generate_main
from llm.tokenizer import BPETokenizer, EOS_TOKEN
from llm.train import get_batch
from llm.translation import (TranslationData, load_translation_data, prepare_translation,
                             source_key, split_pairs, translation_prompt)


class TranslationTests(unittest.TestCase):
    def setUp(self):
        self.pairs = [{"english": "Hello!", "chinese": "你好！"},
                      {"english": "The sun is bright.", "chinese": "太阳很明亮。"}]
        self.tokenizer = BPETokenizer.train(
            "".join(translation_prompt(p["english"]) + p["chinese"] + EOS_TOKEN
                    for p in self.pairs), 300)

    def test_only_chinese_and_eos_are_targets(self):
        data = TranslationData(self.pairs, self.tokenizer, 128)
        for pair, (inputs, targets) in zip(self.pairs, data.examples):
            prefix = self.tokenizer.encode(translation_prompt(pair["english"]))
            response = self.tokenizer.encode(pair["chinese"]) + [self.tokenizer.eos_token_id]
            self.assertEqual(inputs.tolist(), (prefix + response)[:-1])
            self.assertEqual(targets[:len(prefix) - 1].tolist(), [-100] * (len(prefix) - 1))
            self.assertEqual(targets[len(prefix) - 1:].tolist(), response)
            self.assertEqual(targets[len(prefix) - 1].item(), response[0])
            self.assertEqual(self.tokenizer.decode(response[:-1]), pair["chinese"])

    def test_batch_padding_is_masked_and_loss_ignores_it(self):
        data = TranslationData(self.pairs, self.tokenizer, 128)
        generator = torch.Generator().manual_seed(42)
        inputs, targets = get_batch(data, 16, 128, "cpu", generator)
        self.assertEqual(inputs.shape, targets.shape)
        self.assertTrue(any(targets[:, -1] == -100))
        for x, y in zip(inputs, targets):
            last = (y != -100).nonzero()[-1].item()
            self.assertTrue(torch.all(y[last + 1:] == -100))
            self.assertTrue(torch.all(x[last + 1:] == self.tokenizer.eos_token_id))
            self.assertEqual(y[last].item(), self.tokenizer.eos_token_id)
        model = LanguageModel(ModelConfig(vocab_size=self.tokenizer.vocab_size,
                              context_size=128, n_layers=1, n_heads=2, n_embd=16, dropout=0))
        logits, loss = model(inputs, targets)
        valid = targets != -100
        torch.testing.assert_close(loss, F.cross_entropy(logits[valid], targets[valid]))
        loss.backward()
        self.assertTrue(torch.isfinite(model.token_embedding.weight.grad).all())

    def test_complete_pairs_are_not_truncated(self):
        data = TranslationData(self.pairs, self.tokenizer, 128)
        length = len(data.examples[0][0])
        shorter = TranslationData(self.pairs, self.tokenizer, length)
        self.assertEqual(len(shorter), 1)
        self.assertEqual(shorter.skipped, 1)
        with self.assertRaisesRegex(ValueError, "No complete"):
            TranslationData(self.pairs, self.tokenizer, 1)

    def test_source_groups_and_duplicates_stay_together(self):
        pairs = self.pairs + [self.pairs[0], {"english": " HELLO! ", "chinese": "您好！"}]
        train, validation = split_pairs(pairs)
        self.assertEqual(len(train) + len(validation), 3)
        self.assertFalse({source_key(p["english"]) for p in train} &
                         {source_key(p["english"]) for p in validation})
        self.assertEqual((train, validation), tuple(split_pairs(pairs)))

    def test_archive_preparation_preserves_attribution(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            archive = folder / "source.zip"
            with ZipFile(archive, "w") as zipped:
                zipped.writestr("cmn.txt", "Hello!\t你好！\tAuthor A\nGoodbye!\t再见！\tAuthor B\n")
                zipped.writestr("_about.txt", "License and attribution")
            output = folder / "enzh.jsonl"
            prepare_translation(output, archive)
            records = [json.loads(path.read_text()) for path in
                       (output, folder / "enzh-valid.jsonl")]
            self.assertEqual({p["attribution"] for p in records}, {"Author A", "Author B"})
            self.assertEqual((folder / "enzh-sources/cmn-eng.zip").read_bytes(), archive.read_bytes())
            self.assertEqual((folder / "enzh-sources/_about.txt").read_text(), "License and attribution")
            train, valid, tokenizer = load_translation_data(output, 128, 300,
                                                            folder / "enzh-valid.jsonl")
            self.assertEqual((len(train), len(valid)), (1, 1))
            self.assertIsNotNone(tokenizer.eos_token_id)
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_translation_data(output, 128, 300, output)

    def test_tokenizer_uses_training_only(self):
        with tempfile.TemporaryDirectory() as folder:
            train_path, valid_path = [Path(folder) / name for name in ("train.jsonl", "valid.jsonl")]
            train_path.write_text(json.dumps(self.pairs[0]) + "\n", encoding="utf-8")
            valid_path.write_text(json.dumps(self.pairs[1]) + "\n", encoding="utf-8")
            with patch("llm.translation.BPETokenizer.train", wraps=BPETokenizer.train) as trainer:
                load_translation_data(train_path, 128, 300, valid_path)
            training_text = trainer.call_args.args[0]
            self.assertIn(self.pairs[0]["chinese"], training_text)
            self.assertNotIn(self.pairs[1]["chinese"], training_text)

    def test_prompt_rejects_empty_or_reserved_token(self):
        for english in (" ", EOS_TOKEN):
            with self.assertRaises(ValueError):
                translation_prompt(english)

    def test_translation_cli_prints_only_response_and_uses_greedy_decoding(self):
        model = LanguageModel(ModelConfig(vocab_size=self.tokenizer.vocab_size,
                              context_size=128, n_layers=1, n_heads=2, n_embd=16, dropout=0))
        prefix = self.tokenizer.encode(translation_prompt("Hello!"))
        response = self.tokenizer.encode("你好！") + [self.tokenizer.eos_token_id]
        result = torch.tensor([prefix + response])
        with patch("sys.argv", ["generate", "--translate", "Hello!", "--device", "cpu"]), \
                patch("llm.generate.load_model", return_value=(model, self.tokenizer)), \
                patch.object(model, "generate", return_value=result) as generate, \
                patch("sys.stdout", new_callable=io.StringIO) as stdout:
            generate_main()
        self.assertEqual(stdout.getvalue(), "你好！\n")
        self.assertEqual(generate.call_args.args[2], 0)
        self.assertEqual(generate.call_args.kwargs["repetition_penalty"], 1)
        self.assertEqual(generate.call_args.args[0].tolist(), [prefix])


if __name__ == "__main__":
    unittest.main()
