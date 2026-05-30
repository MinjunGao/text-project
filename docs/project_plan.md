# Project Plan: Char-Level GPT From Scratch

## 1. Project idea
Train a small text generation model **from scratch** for the MSAI 495 GenAI class.
The model is a character-level, decoder-only Transformer (a "tiny GPT"). It learns
to predict the next character given the preceding characters, and can then generate
new text one character at a time.

The goal is **not** state-of-the-art text quality. The goals are:
- A clean, readable implementation.
- A clear explanation of how it works.
- A documented training process.
- Generated samples saved over the course of training.
- Metrics tracking (loss curves, etc.).
- A simple chatbot GUI for interacting with the trained model.

## 2. Constraints
- No pretrained models and no fine-tuning of any external language model
  (GPT-2, LLaMA, Hugging Face checkpoints, etc.).
- Weights are randomly initialized and trained from scratch.
- PyTorch only.
- Simple character-level tokenization (each unique character is a token).
- Small and reliable; built incrementally with a clean commit history.
- No faked results. Only document numbers from commands that actually ran.

## 3. Model
A decoder-only Transformer in the style of a minimal GPT:
- **Tokenization:** character level. Build a vocabulary from the unique characters
  in the training corpus; map char <-> integer id.
- **Embeddings:** token embedding table + learned positional embedding.
- **Blocks:** stacked Transformer decoder blocks, each with masked (causal)
  multi-head self-attention + a feed-forward MLP, with residual connections and
  layer norm.
- **Head:** a linear layer projecting back to vocabulary logits.
- **Objective:** next-character prediction with cross-entropy loss.
- **Generation:** autoregressive sampling with controls like temperature and top-k.

Model size (number of layers, heads, embedding dim, context length) will be kept
small so it trains reliably on modest hardware. Exact hyperparameters will live in
a config file under `configs/`.

## 4. Data source
Public-domain text from **Project Gutenberg**. Raw downloaded text goes in
`data/raw/`. A preprocessing step (cleaning, optional header/footer stripping,
building the vocabulary, train/val split) produces files in `data/processed/`.

Only public-domain works will be used, consistent with Project Gutenberg's terms.

## 5. Extra criteria
### MLOps
- **Configs** saved in `configs/` so runs are reproducible.
- **Checkpoints** saved in `outputs/checkpoints/`.
- **Metrics** (e.g. train/val loss per step) saved in `outputs/metrics/`.
- **Loss curves** generated from the metrics.
- **Generated samples** saved in `outputs/samples/`.
- **Logs** saved in `outputs/logs/`.

### Chatbot GUI
A simple **Streamlit** app in `app/` where the user can:
- Enter a prompt.
- Adjust generation settings (e.g. max new tokens, temperature, top-k).
- See the generated text.

## 6. Planned build steps (incremental)
The project will be built in small, testable steps with separate commits. Roughly:
1. Project scaffolding: rules, plan, README, folder structure. *(this step)*
2. Data download + preprocessing + char tokenizer, with a smoke test.
3. Model definition (tiny GPT), with a forward-pass smoke test.
4. Training loop with config, checkpointing, and metrics logging.
5. Sampling / generation script.
6. Metrics + loss-curve plotting.
7. Streamlit chatbot GUI.

Each step will include a lightweight test or smoke test, and results will only be
documented if the corresponding command actually ran.
