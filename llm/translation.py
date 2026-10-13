"""Sentence-pair preparation and Chinese-only training targets."""

import csv
import hashlib
import io
import json
from pathlib import Path
import random
import unicodedata
from urllib.request import urlopen
from zipfile import ZipFile

import torch

from .tokenizer import BPETokenizer, EOS_TOKEN

SOURCE_URL = "https://www.manythings.org/anki/cmn-eng.zip"


def translation_prompt(english):
    if not english.strip() or EOS_TOKEN in english:
        raise ValueError("English must be nonempty and cannot contain the end token")
    return f"English: {english.strip()}\nChinese:\n"


def source_key(english):
    return " ".join(unicodedata.normalize("NFC", english).casefold().split())


def split_pairs(pairs, seed=42):
    groups = {}
    seen = set()
    for pair in pairs:
        identity = (pair["english"], pair["chinese"])
        if identity not in seen:
            groups.setdefault(source_key(pair["english"]), []).append(pair)
            seen.add(identity)
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError("Translation corpus needs at least two distinct English sentences")
    random.Random(seed).shuffle(keys)
    boundary = max(1, min(len(keys) - 1, int(len(keys) * 0.9)))
    return [[pair for key in split for pair in groups[key]]
            for split in (keys[:boundary], keys[boundary:])]


def prepare_translation(output, source=None):
    output = Path(output)
    if output.suffix != ".jsonl":
        raise ValueError("Translation output must have a .jsonl extension")
    output.parent.mkdir(parents=True, exist_ok=True)
    originals = output.parent / "enzh-sources"
    originals.mkdir(exist_ok=True)
    archive = originals / "cmn-eng.zip"
    if source is not None:
        raw = Path(source).read_bytes()
    elif archive.exists():
        raw = archive.read_bytes()
    else:
        print(f"Downloading {SOURCE_URL}", flush=True)
        with urlopen(SOURCE_URL, timeout=60) as response:
            raw = response.read()
    with ZipFile(io.BytesIO(raw)) as zipped:
        text = zipped.read("cmn.txt").decode("utf-8-sig")
        about = zipped.read("_about.txt")
    pairs = []
    for row in csv.reader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE):
        if len(row) != 3 or not all(field.strip() for field in row):
            raise ValueError("Expected English, Chinese, and attribution in every source row")
        english, chinese, attribution = (field.strip() for field in row)
        if EOS_TOKEN in english or EOS_TOKEN in chinese:
            raise ValueError("Source sentences cannot contain the reserved end token")
        pairs.append(dict(english=english, chinese=chinese, attribution=attribution))
    train, validation = split_pairs(pairs)
    archive.write_bytes(raw)
    (originals / "_about.txt").write_bytes(about)
    validation_path = output.with_name(output.stem + "-valid.jsonl")
    for path, records in ((output, train), (validation_path, validation)):
        path.write_text("".join(json.dumps(pair, ensure_ascii=False) + "\n" for pair in records),
                        encoding="utf-8")
        print(f"Saved {path}: {len(records):,} sentence pairs", flush=True)
    metadata = {"dataset": "ManyThings / Tatoeba English-Mandarin", "source": SOURCE_URL,
                "license": "CC-BY 2.0 France; per-sentence attribution retained in JSONL",
                "sha256": hashlib.sha256(raw).hexdigest(), "train_pairs": len(train),
                "validation_pairs": len(validation), "split_seed": 42,
                "split": "90/10 by normalized English sentence; exact pairs deduplicated",
                "scripts": "Simplified and traditional Chinese, as provided by the source"}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def read_pairs(path):
    pairs = []
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            pair = json.loads(line)
            for field in ("english", "chinese"):
                value = pair.get(field)
                if not isinstance(value, str) or not value.strip() or EOS_TOKEN in value:
                    raise ValueError(f"Invalid {field} sentence in {path}")
            pairs.append(pair)
    if not pairs:
        raise ValueError(f"Empty translation corpus: {path}")
    return pairs


class TranslationData:
    def __init__(self, pairs, tokenizer, context_size):
        self.examples = []
        self.skipped = 0
        if tokenizer.eos_token_id is None:
            raise ValueError("Translation tokenizer needs an end token")
        self.pad_id = tokenizer.eos_token_id
        for pair in pairs:
            prompt = tokenizer.encode(translation_prompt(pair["english"]))
            response = tokenizer.encode(pair["chinese"].strip()) + [self.pad_id]
            tokens = prompt + response
            if len(tokens) - 1 > context_size:
                self.skipped += 1
                continue
            # Shifted targets: the final prompt position predicts the first Chinese token.
            targets = [-100] * (len(prompt) - 1) + response
            self.examples.append((torch.tensor(tokens[:-1], dtype=torch.long),
                                  torch.tensor(targets, dtype=torch.long)))
        if not self.examples:
            raise ValueError("No complete translation pairs fit; increase --context-size")
        self.token_count = sum(len(x) + 1 for x, _ in self.examples)

    def __len__(self):
        return len(self.examples)

    def batch(self, batch_size, device, generator=None):
        indices = torch.randint(len(self), (batch_size,), generator=generator).tolist()
        examples = [self.examples[index] for index in indices]
        length = max(len(x) for x, _ in examples)
        inputs = torch.full((batch_size, length), self.pad_id, dtype=torch.long)
        targets = torch.full_like(inputs, -100)
        for index, (x, y) in enumerate(examples):
            inputs[index, :len(x)] = x
            targets[index, :len(y)] = y
        return inputs.to(device), targets.to(device)


def load_translation_data(path, context_size, vocab_size=8192, validation_path=None):
    train = read_pairs(path)
    if validation_path is None:
        train, validation = split_pairs(train)
    else:
        validation = read_pairs(validation_path)
        if {source_key(pair["english"]) for pair in train} & {
                source_key(pair["english"]) for pair in validation}:
            raise ValueError("English sentences overlap between training and validation")
    text = "".join(translation_prompt(pair["english"]) + pair["chinese"].strip()
                   + EOS_TOKEN + "\n" for pair in train)
    tokenizer = BPETokenizer.train(text, vocab_size)
    datasets = [TranslationData(pairs, tokenizer, context_size) for pairs in (train, validation)]
    for split, data in zip(("train", "validation"), datasets):
        print(f"{split}_pairs={len(data):,} skipped_overlength={data.skipped:,}", flush=True)
    return *datasets, tokenizer
