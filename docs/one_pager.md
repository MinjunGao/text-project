# Text Generation Project — One-Pager

## 1. GitHub URL
https://github.com/MinjunGao/text-project

## 2. Project Name & Overview
**Char-Level GPT From Scratch** — a small decoder-only Transformer (a "tiny GPT")
trained from scratch in PyTorch to generate text one character at a time.

- **No pretrained models.** All weights are randomly initialized and trained from
  scratch; I do not fine-tune GPT-2, LLaMA, or any Hugging Face checkpoint.
- **Tokenization:** simple character level (each unique character is a token;
  vocabulary built deterministically from the training corpus).
- **Model:** token + positional embeddings → stacked Transformer blocks with
  explicit causal multi-head self-attention, feed-forward MLP, residual
  connections, layer norm, and dropout → linear language-modeling head. The
  attention block is implemented by hand (no `torch.nn.Transformer`) so it can be
  explained component by component.
- **Data:** public-domain books from Project Gutenberg (Alice in Wonderland,
  The Adventures of Sherlock Holmes, Grimms' Fairy Tales, Works of Edgar Allan Poe)
  — ~1.83M characters after cleaning.

**Result (real, see `docs/experiment_log.md`):** trained 8,000 iterations on a
Colab GPU; validation loss fell from ~4.76 to ~1.45. Generated text is *locally*
plausible — correctly spelled common words, capitalization, punctuation, and
paragraph structure in a Gutenberg-ish style — but **not globally coherent**, which
is the expected ceiling for an ~0.84M-parameter character-level model. The goal was
a clean, explainable implementation rather than state-of-the-art text quality.

## 3. Extra Criteria pursued
I implemented **two** of the listed options (only one was required):

1. **Chatbot GUI** — a Streamlit app (`app/chatbot.py`) where the user picks a
   checkpoint, enters a prompt, and sets `max_new_tokens`, `temperature`, and
   `top_k`, then sees generated text. It shows clear errors if files are missing.
2. **MLOps** — reproducible config (`configs/tiny_transformer.yaml`), saved
   checkpoints, metrics logged to CSV, an auto-generated loss curve
   (`src/plot_metrics.py`), and generated samples saved during/after training.

## 4. Difficulties faced and how I solved them
- **Corpus far too small at first.** Early runs used a ~1.3 KB smoke-test sample,
  which a model this size simply memorizes. I added a Project Gutenberg download
  helper (`scripts/download_gutenberg.py`) and a cleaning step
  (`scripts/prepare_corpus.py`) to build a ~1.8 MB real corpus.
- **CPU training was slow.** I trained on a Google Colab GPU. I wrote a runbook
  (`docs/colab.md`) and used a zip + Google Drive workflow to move the project to
  Colab, since the model auto-selects CUDA via `device: auto`.
- **Knowing when "good enough" is good enough.** The loss curve flattened well
  before 8,000 iterations. I learned that for a character-level model this size,
  the quality ceiling is set by **model capacity**, not more iterations, so I
  stopped rather than overspending compute and documented this trade-off honestly.
- **Reproducibility & honesty.** I fixed a seed, saved configs/metrics, and only
  recorded numbers from runs that actually executed (no fabricated results).
