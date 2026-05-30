# Data Source

## Final corpus: Project Gutenberg
The model will ultimately be trained on **public-domain text from
[Project Gutenberg](https://www.gutenberg.org/)**. Project Gutenberg offers tens of
thousands of books whose copyright has expired, which makes them suitable for
training a model from scratch without licensing concerns.

The planned pipeline (not implemented yet):
1. Download one or more public-domain works into `data/raw/`.
2. Clean the text (e.g. strip the Project Gutenberg header/footer boilerplate,
   normalize whitespace).
3. Build the character vocabulary and a train/validation split into
   `data/processed/`.

Only public-domain works will be used, consistent with Project Gutenberg's
[Terms of Use](https://www.gutenberg.org/policy/permission_how_to.html).

## Smoke-test corpus: `sample_corpus.txt`
`data/raw/sample_corpus.txt` is a **tiny, original, public-domain-style** text used
only for **smoke tests** during development. Its purpose is to let the tokenizer,
model, and training loop run end-to-end quickly on a few paragraphs, so we can
verify the code works before downloading and training on a real corpus.

It is intentionally small and is **not** representative of the final training data.
Unlike the rest of `data/raw/` (which is gitignored), this file is committed so the
smoke tests are reproducible for anyone who clones the repo.
