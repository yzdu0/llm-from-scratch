# Shakespeare Corpus

The default training corpus is Project Gutenberg ebook 100,
[The Complete Works of William Shakespeare](https://www.gutenberg.org/ebooks/100).
This is the complete collection in that edition, not the tiny Shakespeare excerpt.

Run `python -m llm.prepare` to download the UTF-8 edition. Preparation removes
the Gutenberg header and footer and normalizes line endings. Titles, the table
of contents, stage directions, punctuation, and verse line breaks are retained.
The original download, including the Gutenberg license, is saved alongside it.

- `shakespeare.txt`: cleaned training corpus.
- `shakespeare-source.txt`: original ebook, attribution, and license.

These generated files are ignored by Git. To prepare an offline copy, use
`python -m llm.prepare --source /path/to/100.txt`.
