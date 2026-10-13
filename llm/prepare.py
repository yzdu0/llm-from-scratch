import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import re
from urllib.request import urlopen

from .tokenizer import EOS_TOKEN
from .book_catalog import BALANCED_BOOKS, EXTRA_BOOKS


STORIES_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
DARK_BOOKS = (
    (345, "Dracula", "Bram Stoker", "train"),
    (84, "Frankenstein", "Mary Shelley", "train"),
    (43, "Dr. Jekyll and Mr. Hyde", "Robert Louis Stevenson", "train"),
    (2148, "The Works of Edgar Allan Poe, Volume 2", "Edgar Allan Poe", "train"),
    (1952, "The Yellow Wallpaper", "Charlotte Perkins Gilman", "train"),
    (12122, "The Monkey's Paw", "W. W. Jacobs", "train"),
    (209, "The Turn of the Screw", "Henry James", "validation"),
)

BOOKS = DARK_BOOKS + (
    (10007, "Carmilla", "Joseph Sheridan Le Fanu", "train"),
    (174, "The Picture of Dorian Gray", "Oscar Wilde", "train"),
    (768, "Wuthering Heights", "Emily Bronte", "train"),
    (1260, "Jane Eyre", "Charlotte Bronte", "train"),
    (583, "The Woman in White", "Wilkie Collins", "train"),
    (155, "The Moonstone", "Wilkie Collins", "train"),
    (219, "Heart of Darkness", "Joseph Conrad", "train"),
    (159, "The Island of Doctor Moreau", "H. G. Wells", "train"),
    (35, "The Time Machine", "H. G. Wells", "train"),
    (36, "The War of the Worlds", "H. G. Wells", "train"),
    (1400, "Great Expectations", "Charles Dickens", "train"),
    (730, "Oliver Twist", "Charles Dickens", "train"),
    (46, "A Christmas Carol", "Charles Dickens", "train"),
    (98, "A Tale of Two Cities", "Charles Dickens", "train"),
    (1023, "Bleak House", "Charles Dickens", "train"),
    (1342, "Pride and Prejudice", "Jane Austen", "train"),
    (161, "Sense and Sensibility", "Jane Austen", "train"),
    (158, "Emma", "Jane Austen", "train"),
    (120, "Treasure Island", "Robert Louis Stevenson", "train"),
    (421, "Kidnapped", "Robert Louis Stevenson", "train"),
    (2701, "Moby Dick", "Herman Melville", "train"),
    (215, "The Call of the Wild", "Jack London", "train"),
    (910, "White Fang", "Jack London", "train"),
    (11, "Alice's Adventures in Wonderland", "Lewis Carroll", "train"),
    (113, "The Secret Garden", "Frances Hodgson Burnett", "train"),
    (5230, "The Invisible Man", "H. G. Wells", "validation"),
    (105, "Persuasion", "Jane Austen", "validation"),
) + EXTRA_BOOKS


def prepare_books(output, catalog=None):
    """Build a books-only corpus, keeping whole held-out works out of training."""
    output = Path(output)
    catalog = BOOKS if catalog is None else catalog
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = {"train": output, "validation": output.with_name(output.stem + "-valid.txt")}
    if len({book[0] for book in catalog}) != len(catalog):
        raise ValueError("Book catalog must not contain duplicate editions")
    originals = output.parent / "books-sources"
    originals.mkdir(exist_ok=True)
    pending = {split: path.with_suffix(path.suffix + ".part") for split, path in paths.items()}
    records = []
    try:
        with ExitStack() as stack:
            streams = {split: stack.enter_context(path.open("w", encoding="utf-8"))
                       for split, path in pending.items()}
            for ebook, title, author, split in catalog:
                if split not in streams:
                    raise ValueError(f"Invalid book split: {split}")
                url = f"https://www.gutenberg.org/ebooks/{ebook}.txt.utf-8"
                source = originals / f"{ebook}.txt"
                if source.exists():
                    raw = source.read_bytes()
                else:
                    print(f"Downloading {title}: {url}", flush=True)
                    with urlopen(url, timeout=60) as response:
                        raw = response.read()
                cleaned = clean_gutenberg(raw.decode("utf-8-sig"))
                source.write_bytes(raw)
                streams[split].write(cleaned + EOS_TOKEN + "\n\n")
                records.append({"title": title, "author": author, "ebook": ebook,
                                "split": split, "url": url, "source": str(source),
                                "characters": len(cleaned), "sha256": hashlib.sha256(raw).hexdigest()})
        for split, path in paths.items():
            pending[split].replace(path)
            print(f"Saved {path}: {path.stat().st_size:,} bytes", flush=True)
    finally:
        for path in pending.values():
            path.unlink(missing_ok=True)
    metadata = {"dataset": "Classic books: Project Gutenberg", "books": records,
                "train_books": sum(book["split"] == "train" for book in records),
                "validation_books": sum(book["split"] == "validation" for book in records),
                "split": "Whole works; held out: " + ", ".join(book["title"] for book in records if book["split"] == "validation"),
                "rights": "Original Gutenberg files and licenses preserved in sources directory"}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


BALANCED_WEIGHTS = {"fiction": 0.80, "education": 0.10, "maths_books": 0.10}


def prepare_balanced(output):
    """Sample 80% fiction and 20% education, using general education and maths books."""
    output = Path(output)
    prepare_books(output.parent / "books.txt")
    sources = []
    for name, weight in BALANCED_WEIGHTS.items():
        path = output.parent / "books.txt" if name == "fiction" else output.with_name(output.stem + f"-{name}.txt")
        if name != "fiction":
            prepare_books(path, BALANCED_BOOKS[name])
        sources.append({"name": name, "weight": weight, "train": path.name,
                        "validation": path.with_name(path.stem + "-valid.txt").name,
                        "provenance": path.with_suffix(".json").name})
    metadata = {"dataset": "80% fiction / 20% education mixture", "sources": sources,
                "fiction_total": 0.80, "education_total": 0.20, "maths_total": 0.10,
                "sampling": "Fixed-length sequences selected by category weight, then uniformly by token position",
                "notes": ["All existing fiction works retained; complete works held out for validation.",
                          "General education and maths books each receive half of the education share.",
                          "Educational facts and perspectives reflect the historical sources."]}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved mixture manifest: {output.with_suffix('.json')}", flush=True)


def write_stories(lines, destination, count=None, allow_incomplete_tail=False):
    """Stream complete records to disk; None selects the entire split."""
    if count is not None and count < 1:
        raise ValueError("Story count must be positive or None")
    written, current = 0, []
    for line in lines:
        text = line.decode("utf-8")
        if text.strip() == EOS_TOKEN:
            story = "".join(current).strip()
            if story:
                if written:
                    destination.write("\n")
                destination.write(story + "\n" + EOS_TOKEN + "\n")
                written += 1
            current = []
            if count is not None and written == count:
                return written
        else:
            current.append(text)
    if any(part.strip() for part in current):
        if not allow_incomplete_tail:
            raise ValueError("Corpus ends with an incomplete story")
        print(f"Discarded unfinished final record ({sum(map(len, current)):,} characters)", flush=True)
    if not written or (count is not None and written < count):
        raise ValueError(f"Expected {count or 'some'} complete stories, found {written}")
    return written


def prepare_stories(output, train_count, validation_count, version="gpt4"):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    validation = output.with_name(output.stem + "-valid.txt")
    urls = []
    if version != "gpt4":
        raise ValueError("Only GPT-4 TinyStories is supported")
    prefix = "TinyStoriesV2-GPT4"
    counts, pending = [], []
    try:
        for split, count, path in (("train", train_count, output),
                                  ("valid", validation_count, validation)):
            url = f"{STORIES_URL}/{prefix}-{split}.txt"
            selection = "all" if count is None else f"{count:,}"
            print(f"Reading {selection} stories from {url}", flush=True)
            temporary = path.with_suffix(path.suffix + ".part")
            pending.append((temporary, path))
            with urlopen(url, timeout=60) as response, temporary.open("w", encoding="utf-8") as destination:
                def checked_lines():
                    headers = getattr(response, "headers", {})
                    expected = headers.get("Content-Length")
                    received = 0
                    for line in response:
                        received += len(line)
                        yield line
                    if expected is not None and received != int(expected):
                        raise OSError(f"Incomplete download: expected {expected} bytes, received {received}")

                counts.append(write_stories(checked_lines(), destination, count,
                                            allow_incomplete_tail=count is None))
            urls.append(url)
        for temporary, path in pending:
            temporary.replace(path)
            print(f"Saved {path}: {path.stat().st_size:,} bytes", flush=True)
    finally:
        for temporary, _ in pending:
            temporary.unlink(missing_ok=True)
    metadata = {"dataset": "roneneldan/TinyStories", "license": "CDLA-Sharing-1.0",
                "dataset_card": "https://huggingface.co/datasets/roneneldan/TinyStories",
                "sources": urls, "version": version, "train_stories": counts[0],
                "validation_stories": counts[1],
                "selection": "All complete stories" if train_count is None else "First complete stories from each official split"}
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
    parser = argparse.ArgumentParser(description="Prepare 80/20 books or GPT-4 TinyStories")
    parser.add_argument("--corpus", choices=["balanced", "tinystories"], default="balanced")
    parser.add_argument("--out", help="Output prefix; defaults to data/balanced.txt or data/tinystories-gpt4.txt")
    parser.add_argument("--story-version", choices=["gpt4"], default="gpt4")
    parser.add_argument("--stories", type=int, help="Optional training subset; default is the full split")
    parser.add_argument("--validation-stories", type=int, help="Optional validation subset")
    parser.add_argument("--all-stories", action="store_true", help="Use full GPT-4 splits (the default)")
    args = parser.parse_args()
    if args.corpus == "balanced":
        if args.stories is not None or args.validation_stories is not None or args.all_stories:
            parser.error("Story options require --corpus tinystories")
        prepare_balanced(Path(args.out or "data/balanced.txt"))
    else:
        if args.all_stories and (args.stories is not None or args.validation_stories is not None):
            parser.error("Choose full splits or subset counts")
        if any(n is not None and n < 1 for n in (args.stories, args.validation_stories)):
            parser.error("Story counts must be positive")
        prepare_stories(Path(args.out or "data/tinystories-gpt4.txt"), args.stories, args.validation_stories)


if __name__ == "__main__":
    main()
