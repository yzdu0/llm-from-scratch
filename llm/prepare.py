import argparse
from pathlib import Path
import re
from urllib.request import urlopen


SOURCE_URL = "https://www.gutenberg.org/ebooks/100.txt.utf-8"


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
    parser = argparse.ArgumentParser(description="Prepare Shakespeare's Complete Works")
    parser.add_argument("--source", help="Use an existing Gutenberg UTF-8 file instead of downloading")
    parser.add_argument("--out", default="data/shakespeare.txt")
    args = parser.parse_args()
    output = Path(args.out)
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
