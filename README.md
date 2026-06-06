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

### Tokenizer
A simple character-level tokenizer lives in
[`src/tokenizer.py`](src/tokenizer.py). `CharTokenizer` builds a deterministic,
sorted character vocabulary from the training text (`fit`), converts between text
and integer ids (`encode` / `decode`), persists to JSON (`save` / `load`), and
maps unseen characters to an `<unk>` id.

### Train the model
Train the from-scratch Transformer using a YAML config:

```bash
uv run python -m src.train \
    --config configs/tiny_transformer.yaml \
    --corpus_path data/processed/corpus.txt
```

During training it logs train/val loss, writes metrics to
`outputs/metrics/losses.csv`, checkpoints to `outputs/checkpoints/`, and generated
samples to `outputs/samples/` (all gitignored).

**Quick smoke test** (only 20 iterations, runs on the tiny sample corpus):

```bash
python -m src.train --config configs/tiny_transformer.yaml --corpus_path data/processed/corpus.txt --max_iters_override 20
```

(Prefix with `uv run` to use the project environment, and run
`uv run python scripts/prepare_corpus.py` first if `data/processed/corpus.txt`
does not exist yet.)

### Plot the loss curve
After training, plot the train/validation loss curves from the metrics CSV:

```bash
uv run python -m src.plot_metrics \
    --metrics_path outputs/metrics/losses.csv \
    --output_path outputs/metrics/loss_curve.png
```

This saves a PNG to `outputs/metrics/loss_curve.png` (gitignored).

### Generate text
Generate text from a trained checkpoint with a prompt and sampling settings:

```bash
uv run python -m src.generate \
    --checkpoint outputs/checkpoints/ckpt.pt \
    --prompt "The " \
    --max_new_tokens 400 --temperature 0.8 --top_k 40 \
    --output_path outputs/samples/generated.txt
```

The tokenizer is loaded from `tokenizer.json` next to the checkpoint by default
(override with `--tokenizer`). `--output_path` is optional; without it the text is
only printed. (Text quality depends on how long the model was trained.)

### Chatbot GUI
Launch the simple Streamlit chatbot to generate text interactively. Run it from
the project root:

```bash
uv run streamlit run app/chatbot.py
```

In the app you can choose a checkpoint path, enter a prompt, and set
`max_new_tokens`, `temperature`, and `top_k`. It shows a clear error if the
checkpoint or tokenizer files are missing.

## Testing
Run the lightweight tests with:

```bash
uv run pytest
```

## Status
Project scaffolding and dependencies are set up, with the configuration file in
place. The tokenizer, model, and training loop are **not implemented yet**.
