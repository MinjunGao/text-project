"""Generate text from a trained checkpoint.

Loads a checkpoint saved by ``src/train.py`` plus its character tokenizer,
conditions on a text prompt, and autoregressively generates new characters with
configurable ``temperature``, ``top_k``, and ``max_new_tokens``. The result is
printed and can optionally be saved to ``outputs/samples/``.

Example:
    python -m src.generate \\
        --checkpoint outputs/checkpoints/ckpt.pt \\
        --prompt "The " --max_new_tokens 200 --temperature 0.8 --top_k 40
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.model import GPTConfig, TinyGPT
from src.tokenizer import CharTokenizer
from src.train import resolve_device

DEFAULT_CHECKPOINT = Path("outputs/checkpoints/ckpt.pt")


def load_model(checkpoint_path: Path, device: str) -> TinyGPT:
    """Load a :class:`TinyGPT` model from a training checkpoint.

    Args:
        checkpoint_path: Path to a checkpoint produced by ``src/train.py``.
        device: Device to load the model onto.

    Returns:
        The model in eval mode with weights restored.

    Raises:
        FileNotFoundError: If ``checkpoint_path`` does not exist.
    """
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train a model first."
        )
    # The checkpoint is produced by this project and is therefore trusted.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model_config = GPTConfig(
        vocab_size=int(ckpt["vocab_size"]),
        block_size=int(cfg["block_size"]),
        n_embd=int(cfg["n_embd"]),
        n_head=int(cfg["n_head"]),
        n_layer=int(cfg["n_layer"]),
        dropout=float(cfg["dropout"]),
    )
    model = TinyGPT(model_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def generate_text(
    model: TinyGPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    device: str,
) -> str:
    """Generate text conditioned on ``prompt``.

    Args:
        model: A loaded :class:`TinyGPT`.
        tokenizer: The matching character tokenizer.
        prompt: Conditioning text. If empty, a newline is used as the seed.
        max_new_tokens: Number of characters to generate.
        temperature: Softmax temperature (>0).
        top_k: Restrict sampling to the ``top_k`` most likely characters, or
            ``None`` to disable.
        device: Device to run generation on.

    Returns:
        The full text: the prompt followed by the generated characters.
    """
    seed = prompt if prompt else "\n"
    start_ids = tokenizer.encode(seed) or [0]
    context = torch.tensor([start_ids], dtype=torch.long, device=device)
    generated = model.generate(
        context,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    return tokenizer.decode(generated[0].tolist())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to the trained checkpoint (default: outputs/checkpoints/ckpt.pt).",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="Path to the tokenizer JSON (default: tokenizer.json next to checkpoint).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Prompt text to condition generation on (default: empty).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=400,
        help="Number of characters to generate (default: 400).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature; lower is more conservative (default: 0.8).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Restrict sampling to the top-k characters; <=0 disables (default: 40).",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="Optional path to save the generated text (e.g. outputs/samples/gen.txt).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: load a checkpoint, generate text, print and optionally save."""
    args = parse_args(argv)
    device = resolve_device("auto")

    tokenizer_path = args.tokenizer or args.checkpoint.parent / "tokenizer.json"
    if not Path(tokenizer_path).is_file():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    tokenizer = CharTokenizer.load(tokenizer_path)
    model = load_model(args.checkpoint, device)

    top_k = args.top_k if args.top_k and args.top_k > 0 else None
    text = generate_text(
        model,
        tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=top_k,
        device=device,
    )

    print("=" * 60)
    print(text)
    print("=" * 60)

    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(text, encoding="utf-8")
        print(f"Saved generated text to {args.output_path}")


if __name__ == "__main__":
    main()
