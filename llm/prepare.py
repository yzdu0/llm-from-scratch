import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import re
from urllib.request import urlopen

from .tokenizer import EOS_TOKEN


SOURCE_URL = "https://www.gutenberg.org/ebooks/100.txt.utf-8"
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
)


def prepare_books(output):
    """Build a books-only corpus, keeping whole held-out works out of training."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = {"train": output, "validation": output.with_name(output.stem + "-valid.txt")}
    if len({book[0] for book in BOOKS}) != len(BOOKS):
        raise ValueError("Book catalog must not contain duplicate editions")
    originals = output.parent / "books-sources"
    originals.mkdir(exist_ok=True)
    pending = {split: path.with_suffix(path.suffix + ".part") for split, path in paths.items()}
    records = []
    try:
        with ExitStack() as stack:
            streams = {split: stack.enter_context(path.open("w", encoding="utf-8"))
                       for split, path in pending.items()}
            for ebook, title, author, split in BOOKS:
                if split not in streams:
                    raise ValueError(f"Invalid book split: {split}")
                url = f"https://www.gutenberg.org/ebooks/{ebook}.txt.utf-8"
                source = originals / f"{ebook}.txt"
                existing = output.parent / "darkfiction-sources" / f"{ebook}.txt"
                if source.exists():
                    raw = source.read_bytes()
                elif existing.exists():
                    raw = existing.read_bytes()
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
                "split": "Whole works; The Turn of the Screw, The Invisible Man, and Persuasion held out",
                "rights": "Original Gutenberg files and licenses preserved in sources directory"}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def prepare_darkfiction(output, mix_with=None, mix_validation_with=None):
    output = Path(output)
    validation = output.with_name(output.stem + "-valid.txt")
    if (mix_with is None) != (mix_validation_with is None):
        raise ValueError("Supply both training and validation files when mixing corpora")
    sources = [Path(path) for path in (mix_with, mix_validation_with) if path is not None]
    if len(sources) == 2 and sources[0].resolve() == sources[1].resolve():
        raise ValueError("Training and validation inputs must be distinct")
    if any(path.resolve() in {output.resolve(), validation.resolve()} for path in sources):
        raise ValueError("Output files must differ from the input corpora")
    parts = {"train": [], "validation": []}
    for split, path in zip(parts, sources):
        base = path.read_text(encoding="utf-8").rstrip()
        parts[split].append(base + ("\n" if base.endswith(EOS_TOKEN) else "\n" + EOS_TOKEN + "\n"))
    originals = output.parent / "darkfiction-sources"
    originals.mkdir(parents=True, exist_ok=True)
    records = []
    for ebook, title, author, split in DARK_BOOKS:
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
        parts[split].append(cleaned + EOS_TOKEN + "\n")
        records.append({"title": title, "author": author, "ebook": ebook, "split": split,
                        "url": url, "source": str(source), "characters": len(cleaned),
                        "sha256": hashlib.sha256(raw).hexdigest()})
    for path, split in ((output, "train"), (validation, "validation")):
        text = "\n".join(parts[split])
        path.write_text(text, encoding="utf-8")
        print(f"Saved {path}: {len(text):,} characters", flush=True)
    metadata = {"dataset": "Dark fiction: Project Gutenberg", "books": records,
                "split": "Whole works; The Turn of the Screw is held out",
                "mixed_with": [str(path) for path in sources],
                "rights": "Original Gutenberg files and licenses preserved in sources directory"}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


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


def prepare_stories(output, train_count, validation_count, version="original"):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    validation = output.with_name(output.stem + "-valid.txt")
    urls = []
    prefix = "TinyStoriesV2-GPT4" if version == "gpt4" else "TinyStories"
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
    parser = argparse.ArgumentParser(description="Prepare text or English-Chinese sentence pairs")
    parser.add_argument("--corpus", choices=["shakespeare", "tinystories", "darkfiction", "books", "enzh"], default="shakespeare")
    parser.add_argument("--source", help="Existing Gutenberg UTF-8 file or enzh ZIP archive")
    parser.add_argument("--out", help="Defaults to data/<corpus>.txt (enzh: .jsonl)")
    parser.add_argument("--stories", type=int, default=10000)
    parser.add_argument("--validation-stories", type=int, default=1000)
    parser.add_argument("--story-version", choices=["original", "gpt4"], default="original")
    parser.add_argument("--all-stories", action="store_true", help="Use entire TinyStories training and validation splits")
    parser.add_argument("--mix-with", help="Existing training corpus to combine with dark fiction")
    parser.add_argument("--mix-validation-with", help="Matching validation corpus to combine with dark fiction")
    args = parser.parse_args()
    extension = "jsonl" if args.corpus == "enzh" else "txt"
    corpus_name = "tinystories-gpt4" if args.corpus == "tinystories" and args.story_version == "gpt4" else args.corpus
    output = Path(args.out or f"data/{corpus_name}.{extension}")
    if args.corpus != "tinystories" and (args.all_stories or args.story_version != "original"):
        parser.error("all-stories and story-version apply only to TinyStories")
    if args.corpus == "books":
        if args.source or args.mix_with or args.mix_validation_with:
            parser.error("Books uses its own catalog and does not accept source or corpus mixing")
        prepare_books(output)
        return
    if args.corpus == "enzh":
        if args.mix_with or args.mix_validation_with:
            parser.error("Translation pairs cannot be mixed with unpaired text")
        from .translation import prepare_translation
        prepare_translation(output, args.source)
        return
    if args.corpus == "darkfiction":
        if args.source:
            parser.error("Dark fiction does not use --source; downloaded originals are cached")
        prepare_darkfiction(output, args.mix_with, args.mix_validation_with)
        return
    if args.mix_with or args.mix_validation_with:
        parser.error("Corpus mixing is only supported with --corpus darkfiction")
    if args.corpus == "tinystories":
        if args.source or min(args.stories, args.validation_stories) < 1:
            parser.error("TinyStories needs positive story counts and does not use --source")
        prepare_stories(output, None if args.all_stories else args.stories,
                        None if args.all_stories else args.validation_stories, args.story_version)
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
