"""A simple Streamlit chatbot GUI for the from-scratch character-level GPT.

The user picks a trained checkpoint, enters a prompt, sets sampling controls
(max new tokens, temperature, top-k), and gets generated text. The model is the
project's own ``TinyGPT`` trained from scratch -- no pretrained models are used.

Run from the project root:
    streamlit run app/chatbot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is importable (streamlit puts app/ on sys.path, not root).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.generate import generate_text, load_model
from src.tokenizer import CharTokenizer
from src.train import resolve_device

DEFAULT_CHECKPOINT = "outputs/checkpoints/ckpt.pt"


@st.cache_resource(show_spinner=False)
def _load(checkpoint_path: str, tokenizer_path: str):
    """Load and cache the tokenizer + model for a given pair of paths.

    Caching avoids reloading the checkpoint on every interaction. The cache key is
    the path pair, so changing either path loads a fresh model.

    Returns:
        A ``(model, tokenizer, device)`` tuple.
    """
    device = resolve_device("auto")
    tokenizer = CharTokenizer.load(tokenizer_path)
    model = load_model(Path(checkpoint_path), device)
    return model, tokenizer, device


def main() -> None:
    """Render the Streamlit app."""
    st.set_page_config(page_title="Tiny GPT Chatbot", page_icon="*")
    st.title("Tiny GPT Chatbot")
    st.caption(
        "Character-level Transformer trained from scratch (no pretrained models)."
    )

    with st.sidebar:
        st.header("Model")
        checkpoint_path = st.text_input("Checkpoint path", value=DEFAULT_CHECKPOINT)
        tokenizer_path = st.text_input(
            "Tokenizer path (blank = next to checkpoint)", value=""
        )
        st.header("Generation settings")
        max_new_tokens = st.slider("max_new_tokens", 10, 1000, 400, step=10)
        temperature = st.slider("temperature", 0.1, 2.0, 0.8, step=0.1)
        top_k = st.slider("top_k (0 disables)", 0, 100, 40, step=1)

    prompt = st.text_area("Prompt", value="The ", height=120)

    if not st.button("Generate", type="primary"):
        return

    # Resolve the tokenizer path (default: tokenizer.json beside the checkpoint).
    resolved_tokenizer = (
        tokenizer_path.strip()
        or str(Path(checkpoint_path).parent / "tokenizer.json")
    )

    # Validate files before attempting to load, with clear messages.
    if not Path(checkpoint_path).is_file():
        st.error(f"Checkpoint not found: `{checkpoint_path}`. Train a model first.")
        return
    if not Path(resolved_tokenizer).is_file():
        st.error(f"Tokenizer not found: `{resolved_tokenizer}`.")
        return

    try:
        with st.spinner("Loading model..."):
            model, tokenizer, device = _load(checkpoint_path, resolved_tokenizer)
    except Exception as exc:  # noqa: BLE001 - surface any load error to the user
        st.error(f"Failed to load model: {exc}")
        return

    try:
        with st.spinner("Generating..."):
            text = generate_text(
                model,
                tokenizer,
                prompt=prompt,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_k=int(top_k) if top_k > 0 else None,
                device=device,
            )
    except Exception as exc:  # noqa: BLE001 - surface any generation error
        st.error(f"Generation failed: {exc}")
        return

    st.caption(f"Device: {device}")
    st.subheader("Generated text")
    st.text(text)


main()
