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

## Status
Project scaffolding and dependencies are set up, with the configuration file in
place. The tokenizer, model, and training loop are **not implemented yet**.
