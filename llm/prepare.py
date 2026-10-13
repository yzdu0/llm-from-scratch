import argparse
import json
from pathlib import Path
import re
from urllib.request import urlopen

from .tokenizer import EOS_TOKEN


SOURCE_URL = "https://www.gutenberg.org/ebooks/100.txt.utf-8"
STORIES_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"


def read_stories(lines, count):
    """Keep complete records and stop reading once the requested subset is ready."""
    if count < 1:
        raise ValueError("Story count must be positive")
    stories, current = [], []
    for line in lines:
        text = line.decode("utf-8")
        if text.strip() == EOS_TOKEN:
            story = "".join(current).strip()
            if story:
                stories.append(story + "\n" + EOS_TOKEN + "\n")
            current = []
            if len(stories) == count:
                return "\n".join(stories)
        else:
            current.append(text)
    raise ValueError(f"Expected {count} complete stories, found {len(stories)}")


def prepare_stories(output, train_count, validation_count):
    texts = []
    urls = []
    for split, count in (("train", train_count), ("valid", validation_count)):
        url = f"{STORIES_URL}/TinyStories-{split}.txt"
        print(f"Reading {count:,} stories from {url}", flush=True)
        with urlopen(url, timeout=60) as response:
            texts.append(read_stories(response, count))
        urls.append(url)
    output.parent.mkdir(parents=True, exist_ok=True)
    validation = output.with_name(output.stem + "-valid.txt")
    for path, text in zip((output, validation), texts):
        path.write_text(text, encoding="utf-8")
        print(f"Saved {path}: {len(text):,} characters", flush=True)
    metadata = {"dataset": "roneneldan/TinyStories", "license": "CDLA-Sharing-1.0",
                "dataset_card": "https://huggingface.co/datasets/roneneldan/TinyStories",
                "sources": urls, "train_stories": train_count,
                "validation_stories": validation_count,
                "selection": "First complete stories from each official split"}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def clean_gutenberg(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    start = re.search(r"^\*\*\* START OF THE PROJECT GUTENBERG EBOOK .*?\*\*\*[^\n]*\n",
                      text, re.MULTILINE)
    end = re.search(r"^\*\*\* END OF THE PROJECT GUTENBERG EBOOK .*?\*\*\*",
                    text, re.MULTILINE)
    if start is None or end is None or start.end() >= end.start():
        raise ValueError("Missing or invalid Gutenberg boundaries; refusing to prepare corpus")
    return text[start.end():end.start()].strip() + "\n"


def main():
    parser = argparse.ArgumentParser(description="Prepare Shakespeare or modern-English TinyStories")
    parser.add_argument("--corpus", choices=["shakespeare", "tinystories"], default="shakespeare")
    parser.add_argument("--source", help="Use an existing Gutenberg UTF-8 file instead of downloading")
    parser.add_argument("--out", help="Defaults to data/<corpus>.txt")
    parser.add_argument("--stories", type=int, default=10000)
    parser.add_argument("--validation-stories", type=int, default=1000)
    args = parser.parse_args()
    output = Path(args.out or f"data/{args.corpus}.txt")
    if args.corpus == "tinystories":
        if args.source or min(args.stories, args.validation_stories) < 1:
            parser.error("TinyStories needs positive story counts and does not use --source")
        prepare_stories(output, args.stories, args.validation_stories)
        return
    source = output.with_name(output.stem + "-source.txt")
    if args.source:
        raw = Path(args.source).read_bytes()
    else:
        print(f"Downloading {SOURCE_URL}", flush=True)
        with urlopen(SOURCE_URL, timeout=60) as response:
            raw = response.read()
    corpus = clean_gutenberg(raw.decode("utf-8-sig"))
    output.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(raw)
    output.write_text(corpus, encoding="utf-8")
    print(f"Saved {output}: {len(corpus):,} characters; original with license: {source}")


if __name__ == "__main__":
    main()
