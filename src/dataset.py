"""Dataset utilities for character-level next-character prediction.

This module turns a cleaned corpus into integer-id tensors and serves random
mini-batches suitable for language-model training. A batch is a pair ``(x, y)``
where ``y`` is ``x`` shifted one character into the future, so the model learns to
predict the next character at every position.

No model code lives here; these are plain data helpers built on PyTorch tensors.
"""

from __future__ import annotations

from pathlib import Path

import torch

DEFAULT_CORPUS_PATH = Path("data/processed/corpus.txt")


def load_text(path: str | Path = DEFAULT_CORPUS_PATH) -> str:
    """Load the processed corpus text from ``path``.

    Args:
        path: Path to the cleaned corpus (default: ``data/processed/corpus.txt``).

    Returns:
        The corpus contents as a string.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Corpus not found at {path}. Run scripts/prepare_corpus.py first."
        )
    return path.read_text(encoding="utf-8")


def make_data_tensor(ids: list[int]) -> torch.Tensor:
    """Convert a list of token ids into a 1-D ``torch.long`` tensor.

    Args:
        ids: Encoded token ids (e.g. from ``CharTokenizer.encode``).

    Returns:
        A 1-D tensor of dtype ``torch.long``.
    """
    return torch.tensor(ids, dtype=torch.long)


def train_val_split(
    data: torch.Tensor, train_split: float = 0.9
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a 1-D tensor of token ids into contiguous train/validation parts.

    The split is contiguous (the first ``train_split`` fraction is training data),
    which is the standard approach for character-level language modeling.

    Args:
        data: 1-D tensor of token ids.
        train_split: Fraction of the data used for training, in ``(0, 1)``.

    Returns:
        A ``(train_data, val_data)`` tuple of 1-D tensors.

    Raises:
        ValueError: If ``data`` is not 1-D or ``train_split`` is out of range.
    """
    if data.dim() != 1:
        raise ValueError(f"Expected a 1-D tensor, got shape {tuple(data.shape)}.")
    if not 0.0 < train_split < 1.0:
        raise ValueError(f"train_split must be in (0, 1), got {train_split}.")
    n_train = int(len(data) * train_split)
    return data[:n_train], data[n_train:]


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of ``(x, y)`` sequences for language modeling.

    For each of ``batch_size`` random start positions ``i``, ``x`` is the slice
    ``data[i : i + block_size]`` and ``y`` is the same slice shifted by one,
    ``data[i + 1 : i + 1 + block_size]``. Thus ``y[:, t]`` is the next character
    after ``x[:, t]``.

    Args:
        data: 1-D tensor of token ids to sample from.
        batch_size: Number of sequences in the batch.
        block_size: Context length (sequence length) of each sample.
        device: Device to move the returned tensors to.
        generator: Optional ``torch.Generator`` for reproducible sampling.

    Returns:
        A tuple ``(x, y)`` of tensors, each of shape ``[batch_size, block_size]``
        and dtype ``torch.long``.

    Raises:
        ValueError: If ``data`` is too short to produce a ``block_size`` sample
            (it must contain more than ``block_size`` tokens).
    """
    if data.dim() != 1:
        raise ValueError(f"Expected a 1-D tensor, got shape {tuple(data.shape)}.")
    if len(data) <= block_size:
        raise ValueError(
            f"Data length ({len(data)}) must be greater than block_size "
            f"({block_size}) to form (x, y) pairs."
        )

    high = len(data) - block_size
    ix = torch.randint(high, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)
