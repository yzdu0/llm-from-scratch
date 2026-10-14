# Corpus

The prepared training pool contains **99,992,938 tokens**:

- Fiction: 79,993,024 tokens of English Gutenberg books.
- Education: 19,999,914 tokens of literature and criticism, humanities, history,
  science books, and educational web prose from FineWeb-Edu.

Validation contains a separate 5,133,965 tokens. Q&A uses short conversations
from Smol-SmolTalk: 6,485 training examples and 330 validation examples.

```sh
python -m llm.prepare
```

`expanded.json` describes the 80/20 training mixture. `expanded-fiction.txt` and
`expanded-education.txt` have matching `-valid.txt` files. `expanded-tokenizer.json`
holds the frozen vocabulary used to count and train on the pool.
`instruction.jsonl` and `instruction-valid.jsonl` contain Q&A pairs.

`books.txt` and `education-books.txt`, with their validation files, are preparation
inputs built from the seed catalogs in `llm/book_catalog.py`. Source downloads,
licenses and provenance are retained in the `*-sources/` directories and JSON
metadata. `token-cache/` contains disk-backed arrays. Generated data is ignored by Git.

Whole books and web documents are held out by content hash; seed-book splits are
specified in the catalog. Exact normalized document duplicates are omitted, and
normalized questions cannot cross the Q&A split. Training candidates alone teach
BPE merges; complete documents are selected for the 100M-token pool. Shared
quotations and overlapping editions can occur across otherwise distinct works.

Gutenberg files retain their source licenses. FineWeb-Edu has an ODC-BY database
license; source pages retain their own rights. Smol-SmolTalk uses Apache-2.0.
