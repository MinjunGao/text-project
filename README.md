# Char-Level GPT From Scratch

A small character-level, decoder-only Transformer (a "tiny GPT") trained **from
scratch** in PyTorch on public-domain text from Project Gutenberg. Built for the
MSAI 495 GenAI class project.

The focus is a clean implementation, a clear explanation, a documented training
process, generated samples, metrics tracking, and a simple Streamlit chatbot GUI —
not state-of-the-art text quality.

No pretrained models are used; all weights are randomly initialized and trained
from scratch.

See [`docs/project_plan.md`](docs/project_plan.md) for the full plan.

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/) for environment and dependency
management. Install `uv` first (see the uv docs), then:

```bash
# Create the virtual environment (pinned to Python 3.12 via .python-version)
uv venv

# Install project dependencies from pyproject.toml / uv.lock
uv sync
```

Run any command inside the environment with `uv run`, for example:

```bash
uv run python -c "import torch; print(torch.__version__)"
```

Add new dependencies with `uv add <package>` (do not use bare `pip`).

### Dependencies
- `torch` — model and training (from scratch, no pretrained weights)
- `numpy` — array utilities
- `pyyaml` — load configs (e.g. `configs/tiny_transformer.yaml`)
- `matplotlib` — loss curves / plots
- `streamlit` — chatbot GUI

### Configuration
Training/model hyperparameters live in [`configs/tiny_transformer.yaml`](configs/tiny_transformer.yaml).

### Data
The final model will be trained on public-domain text from Project Gutenberg. A tiny
committed sample, [`data/raw/sample_corpus.txt`](data/raw/sample_corpus.txt), is
included only for smoke tests so the pipeline can be run end-to-end quickly. See
[`docs/data_source.md`](docs/data_source.md) for details.

### Download public-domain text (optional)
Fetch a small set of public-domain books from Project Gutenberg into `data/raw/`
(standard library only; downloaded files are gitignored):

```bash
# all books (alice, holmes, grimm, poe)
uv run python scripts/download_gutenberg.py --output_dir data/raw

# or a subset
uv run python scripts/download_gutenberg.py --books alice grimm
```

If a download fails, the script reports it and keeps going. You can also skip this
step and just use the committed `data/raw/sample_corpus.txt` for smoke tests.

### Prepare the corpus
Concatenate and clean the raw `.txt` files in `data/raw/` into a single corpus:

```bash
uv run python scripts/prepare_corpus.py \
    --input_dir data/raw \
    --output_path data/processed/corpus.txt
```

This normalizes line endings and removes excessive whitespace. The output lands in
`data/processed/corpus.txt` (gitignored).

## Testing
Run the lightweight tests with:

```bash
uv run pytest
```

## Status
Project scaffolding and dependencies are set up, with the configuration file in
place. The tokenizer, model, and training loop are **not implemented yet**.
