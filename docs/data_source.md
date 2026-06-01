# Data Source

## Final corpus: Project Gutenberg
The model will ultimately be trained on **public-domain text from
[Project Gutenberg](https://www.gutenberg.org/)**. Project Gutenberg offers tens of
thousands of books whose copyright has expired, which makes them suitable for
training a model from scratch without licensing concerns.

The planned pipeline:
1. Download one or more public-domain works into `data/raw/`
   (see [`scripts/download_gutenberg.py`](../scripts/download_gutenberg.py)).
2. Clean the text (e.g. strip the Project Gutenberg header/footer boilerplate,
   normalize whitespace) via
   [`scripts/prepare_corpus.py`](../scripts/prepare_corpus.py).
3. Build the character vocabulary and a train/validation split into
   `data/processed/` (not implemented yet).

Only public-domain works will be used, consistent with Project Gutenberg's
[Terms of Use](https://www.gutenberg.org/policy/permission_how_to.html).

### Download helper
`scripts/download_gutenberg.py` downloads a small, fixed list of public-domain
plain-text books using only the Python standard library (`urllib`). The current
registry:

| Key      | Work                                          | eBook |
|----------|-----------------------------------------------|-------|
| `alice`  | *Alice's Adventures in Wonderland* (Carroll)  | 11    |
| `holmes` | *The Adventures of Sherlock Holmes* (Doyle)   | 1661  |
| `grimm`  | *Grimms' Fairy Tales* (Brothers Grimm)        | 2591  |
| `poe`    | *The Works of Edgar Allan Poe, Vol. 1*        | 2147  |

All four are in the public domain. Downloaded files land in `data/raw/` and are
**gitignored** (only the tiny `sample_corpus.txt` is committed). If a download
fails (e.g. no network), the script prints a helpful message and continues with the
remaining books instead of crashing.

```bash
# all books
uv run python scripts/download_gutenberg.py --output_dir data/raw

# a subset
uv run python scripts/download_gutenberg.py --books alice grimm
```

## Smoke-test corpus: `sample_corpus.txt`
`data/raw/sample_corpus.txt` is a **tiny, original, public-domain-style** text used
only for **smoke tests** during development. Its purpose is to let the tokenizer,
model, and training loop run end-to-end quickly on a few paragraphs, so we can
verify the code works before downloading and training on a real corpus.

It is intentionally small and is **not** representative of the final training data.
Unlike the rest of `data/raw/` (which is gitignored), this file is committed so the
smoke tests are reproducible for anyone who clones the repo.
