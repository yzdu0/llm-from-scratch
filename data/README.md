# Corpus

The training pool contains **199,999,185 tokens**:

- Fiction: 139,999,249 tokens of English Gutenberg novels and short-story collections.
- Education: 59,999,936 tokens of essays, literature and criticism, humanities, history,
  science books, and educational web prose from FineWeb-Edu.

Validation contains 9,940,819 tokens in separate held-out documents. The training split retains 424 short-story collections and 105 essay books,
in addition to the seed catalogs. Q&A uses short conversations
from Smol-SmolTalk: 6,485 training examples and 330 validation examples.

```sh
python -m llm.prepare
```

`expanded.json` describes the 70/30 training mixture. `expanded-fiction.txt` and
`expanded-education.txt` have matching `-valid.txt` files. `expanded-tokenizer.json`
holds the frozen 16,384-token vocabulary used to count and train on the pool.
`instruction.jsonl` and `instruction-valid.jsonl` contain Q&A pairs.

`books.txt` and `education-books.txt`, with their validation files, are preparation
inputs built from the seed catalogs in `llm/book_catalog.py`.
`expanded-essays.txt` adds essay collections to the education candidates.
Fiction selection interleaves short-story collections with other fiction; education
includes an 12M-token candidate allocation for seed books and essays. Source downloads,
licenses and provenance are retained in the `*-sources/` directories and JSON
metadata. `token-cache/` contains disk-backed arrays. Generated data is ignored by Git.

Whole books and web documents are held out by content hash; seed-book splits are
specified in the catalog. Exact normalized document duplicates are omitted, and
normalized questions cannot cross the Q&A split. Training candidates alone teach
BPE merges; complete documents are selected for the 200M-token pool. Shared
quotations and overlapping editions can occur across otherwise distinct works.

Gutenberg files retain their source licenses. FineWeb-Edu has an ODC-BY database
license; source pages retain their own rights. Smol-SmolTalk uses Apache-2.0.
