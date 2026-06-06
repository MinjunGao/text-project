"""Plot training/validation loss curves from a metrics CSV.

Reads a CSV produced by ``src/train.py`` (columns: ``iter``, ``train_loss``,
``val_loss``) and saves a loss-curve PNG. Both curves are drawn when present.

Example:
    python -m src.plot_metrics \\
        --metrics_path outputs/metrics/losses.csv \\
        --output_path outputs/metrics/loss_curve.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

# Use a non-interactive backend so the script runs headless (e.g. on a server).
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow backend selection)


def read_metrics(
    metrics_path: str | Path,
) -> tuple[list[int], list[float | None], list[float | None]]:
    """Read iterations and loss columns from a metrics CSV.

    Args:
        metrics_path: Path to the CSV file with ``iter``, ``train_loss``, and
            optionally ``val_loss`` columns.

    Returns:
        A tuple ``(iters, train_losses, val_losses)``. Missing/blank cells become
        ``None`` so gaps are not plotted.

    Raises:
        FileNotFoundError: If ``metrics_path`` does not exist.
        ValueError: If the file has no ``iter`` column or no data rows.
    """
    path = Path(metrics_path)
    if not path.is_file():
        raise FileNotFoundError(f"Metrics file not found: {path}")

    iters: list[int] = []
    train_losses: list[float | None] = []
    val_losses: list[float | None] = []

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "iter" not in reader.fieldnames:
            raise ValueError("Metrics CSV must have an 'iter' column.")
        for row in reader:
            iters.append(int(row["iter"]))
            train_losses.append(_to_float(row.get("train_loss")))
            val_losses.append(_to_float(row.get("val_loss")))

    if not iters:
        raise ValueError(f"No data rows found in {path}.")
    return iters, train_losses, val_losses


def _to_float(value: str | None) -> float | None:
    """Parse a CSV cell into a float, returning ``None`` for blank/missing values."""
    if value is None or value.strip() == "":
        return None
    return float(value)


def plot_losses(
    metrics_path: str | Path, output_path: str | Path
) -> Path:
    """Plot train/val loss curves and save them to ``output_path``.

    Args:
        metrics_path: Path to the metrics CSV.
        output_path: Destination PNG path. Parent directories are created.

    Returns:
        The path to the written image.
    """
    iters, train_losses, val_losses = read_metrics(metrics_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    if any(v is not None for v in train_losses):
        ax.plot(iters, train_losses, marker="o", label="train loss")
    if any(v is not None for v in val_losses):
        ax.plot(iters, val_losses, marker="o", label="val loss")

    ax.set_xlabel("iteration")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Training loss curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics_path",
        type=Path,
        default=Path("outputs/metrics/losses.csv"),
        help="Path to the metrics CSV (default: outputs/metrics/losses.csv).",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("outputs/metrics/loss_curve.png"),
        help="Path to save the loss curve PNG (default: outputs/metrics/loss_curve.png).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: read metrics and write the loss-curve image."""
    args = parse_args(argv)
    out = plot_losses(args.metrics_path, args.output_path)
    print(f"Saved loss curve to {out}")


if __name__ == "__main__":
    main()
