"""Train the from-scratch character-level Transformer language model.

Pipeline:
    load YAML config -> load processed corpus -> fit/load char tokenizer ->
    encode + train/val split -> build TinyGPT (random init) -> train with
    cross-entropy next-character prediction -> periodically evaluate, checkpoint,
    and sample.

Artifacts (MLOps):
    * metrics  -> outputs/metrics/losses.csv
    * tokenizer + checkpoints -> outputs/checkpoints/
    * generated samples -> outputs/samples/

Example (quick smoke test on the tiny sample corpus, 20 iterations):
    python -m src.train --config configs/tiny_transformer.yaml \\
        --corpus_path data/processed/corpus.txt --max_iters_override 20
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
import yaml

from src.dataset import get_batch, load_text, make_data_tensor, train_val_split
from src.model import GPTConfig, TinyGPT
from src.tokenizer import CharTokenizer

# Default output locations.
CHECKPOINT_DIR = Path("outputs/checkpoints")
METRICS_PATH = Path("outputs/metrics/losses.csv")
SAMPLES_DIR = Path("outputs/samples")
TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer.json"
CHECKPOINT_PATH = CHECKPOINT_DIR / "ckpt.pt"


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dictionary.

    Args:
        path: Path to the YAML config.

    Returns:
        The parsed configuration as a dict.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(device: str) -> str:
    """Resolve the ``device`` config value to a concrete torch device string.

    Args:
        device: ``"auto"``, ``"cuda"``, ``"mps"``, or ``"cpu"``.

    Returns:
        A concrete device string (``"cuda"``, ``"mps"``, or ``"cpu"``).
    """
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_tokenizer(text: str, tokenizer_path: Path) -> CharTokenizer:
    """Load an existing tokenizer if present, otherwise fit one and save it.

    Args:
        text: Corpus text used to fit a new tokenizer if none exists.
        tokenizer_path: Where the tokenizer vocabulary is stored.

    Returns:
        A fitted :class:`CharTokenizer`.
    """
    if tokenizer_path.is_file():
        print(f"Loading tokenizer from {tokenizer_path}")
        return CharTokenizer.load(tokenizer_path)
    print(f"Fitting new tokenizer and saving to {tokenizer_path}")
    tokenizer = CharTokenizer().fit(text)
    tokenizer.save(tokenizer_path)
    return tokenizer


@torch.no_grad()
def estimate_loss(
    model: TinyGPT,
    splits: dict[str, torch.Tensor],
    batch_size: int,
    block_size: int,
    eval_iters: int,
    device: str,
) -> dict[str, float]:
    """Estimate mean loss on each split by averaging over ``eval_iters`` batches.

    Args:
        model: The model to evaluate.
        splits: Mapping of split name -> token id tensor (e.g. ``{"train": ...}``).
        batch_size: Batch size used for evaluation.
        block_size: Context length.
        eval_iters: Number of batches to average per split.
        device: Device to run on.

    Returns:
        Mapping of split name -> mean loss (float).
    """
    model.eval()
    out: dict[str, float] = {}
    for split, data in splits.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, batch_size, block_size, device=device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def save_checkpoint(
    model: TinyGPT, config: dict[str, Any], iteration: int, path: Path
) -> None:
    """Save model weights and metadata to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "iteration": iteration,
            "vocab_size": model.config.vocab_size,
        },
        path,
    )


def save_sample(
    model: TinyGPT,
    tokenizer: CharTokenizer,
    iteration: int,
    config: dict[str, Any],
    device: str,
) -> Path:
    """Generate a text sample and write it to ``outputs/samples/``.

    Returns:
        The path of the written sample file.
    """
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    # Seed the context with a newline if available, else the first vocab id.
    start_ids = tokenizer.encode("\n") or [0]
    context = torch.tensor([start_ids], dtype=torch.long, device=device)
    generated = model.generate(
        context,
        max_new_tokens=int(config["max_new_tokens"]),
        temperature=float(config["temperature"]),
        top_k=int(config["top_k"]),
    )
    text = tokenizer.decode(generated[0].tolist())
    out_path = SAMPLES_DIR / f"sample_iter_{iteration:06d}.txt"
    out_path.write_text(text, encoding="utf-8")
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tiny_transformer.yaml"),
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--corpus_path",
        type=Path,
        default=Path("data/processed/corpus.txt"),
        help="Path to the processed corpus text.",
    )
    parser.add_argument(
        "--max_iters_override",
        type=int,
        default=None,
        help="Override max_iters from the config (useful for quick smoke tests).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the training loop end to end."""
    args = parse_args(argv)
    config = load_config(args.config)

    if args.max_iters_override is not None:
        config["max_iters"] = args.max_iters_override

    seed = int(config["seed"])
    torch.manual_seed(seed)

    device = resolve_device(str(config["device"]))
    batch_size = int(config["batch_size"])
    block_size = int(config["block_size"])
    max_iters = int(config["max_iters"])
    eval_interval = int(config["eval_interval"])
    eval_iters = int(config["eval_iters"])
    checkpoint_interval = int(config["checkpoint_interval"])
    sample_interval = int(config["sample_interval"])

    print(f"Device: {device} | seed: {seed} | max_iters: {max_iters}")

    # Data + tokenizer.
    text = load_text(args.corpus_path)
    tokenizer = build_tokenizer(text, TOKENIZER_PATH)
    data = make_data_tensor(tokenizer.encode(text))
    train_data, val_data = train_val_split(data, float(config["train_split"]))
    splits = {"train": train_data, "val": val_data}
    print(
        f"Corpus chars: {len(text)} | vocab_size: {tokenizer.vocab_size} | "
        f"train tokens: {len(train_data)} | val tokens: {len(val_data)}"
    )

    # Model (randomly initialized, trained from scratch).
    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_embd=int(config["n_embd"]),
        n_head=int(config["n_head"]),
        n_layer=int(config["n_layer"]),
        dropout=float(config["dropout"]),
    )
    model = TinyGPT(model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))

    # Prepare metrics CSV.
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["iter", "train_loss", "val_loss"])

    def log_metrics(iteration: int) -> dict[str, float]:
        losses = estimate_loss(
            model, splits, batch_size, block_size, eval_iters, device
        )
        with METRICS_PATH.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [iteration, f"{losses['train']:.6f}", f"{losses['val']:.6f}"]
            )
        print(
            f"iter {iteration:6d} | train loss {losses['train']:.4f} | "
            f"val loss {losses['val']:.4f}"
        )
        return losses

    # Training loop.
    for iteration in range(max_iters):
        if iteration % eval_interval == 0:
            log_metrics(iteration)
        if iteration > 0 and iteration % checkpoint_interval == 0:
            save_checkpoint(model, config, iteration, CHECKPOINT_PATH)
        if iteration > 0 and iteration % sample_interval == 0:
            save_sample(model, tokenizer, iteration, config, device)

        x, y = get_batch(train_data, batch_size, block_size, device=device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Final evaluation, checkpoint, and sample so every run produces artifacts.
    final_iter = max_iters
    log_metrics(final_iter)
    save_checkpoint(model, config, final_iter, CHECKPOINT_PATH)
    sample_path = save_sample(model, tokenizer, final_iter, config, device)

    print("\nTraining complete.")
    print(f"Metrics:    {METRICS_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Sample:     {sample_path}")


if __name__ == "__main__":
    main()
