# Corpora

Only two training recipes are maintained:

- **80% fiction / 20% education:** `python -m llm.prepare --corpus balanced`.
  `balanced.json` samples `books.txt` at 80%, `balanced-education.txt` at 10%,
  and `balanced-maths_books.txt` at 10%, each with its matching `-valid.txt`.
  All 115 fiction works remain included. Original Gutenberg files and licenses
  live in `books-sources/`; category JSON files retain provenance and whole-book splits.
- **GPT-4 TinyStories:** `python -m llm.prepare --corpus tinystories --all-stories`.
  `tinystories-gpt4.txt` and `tinystories-gpt4-valid.txt` use the official GPT-4-only
  splits. Metadata records URLs, CDLA-Sharing-1.0 licensing, and story counts.

Prepared text, metadata and token caches are ignored by Git. The shared BPE
vocabulary learns only from training files. `token-cache/` stores compact arrays
for disk-backed sampling. See [books](../docs/balanced.md) and
[GPT-4 TinyStories](../docs/tinystories.md) for training commands.
