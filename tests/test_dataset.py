"""Lightweight tests for src/dataset.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Make the project's src/ directory importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset import get_batch, load_text, make_data_tensor, train_val_split


def test_load_text(tmp_path: Path) -> None:
    """load_text reads file contents; missing file raises FileNotFoundError."""
    p = tmp_path / "corpus.txt"
    p.write_text("hello", encoding="utf-8")
    assert load_text(p) == "hello"
    with pytest.raises(FileNotFoundError):
        load_text(tmp_path / "missing.txt")


def test_train_val_split_sizes_and_contiguity() -> None:
    """Split fractions are correct and the parts are contiguous halves."""
    data = make_data_tensor(list(range(100)))
    train, val = train_val_split(data, train_split=0.9)
    assert len(train) == 90
    assert len(val) == 10
    assert torch.equal(train, torch.arange(0, 90))
    assert torch.equal(val, torch.arange(90, 100))


def test_train_val_split_validates_input() -> None:
    """Invalid train_split or non-1-D input raises ValueError."""
    data = make_data_tensor(list(range(10)))
    with pytest.raises(ValueError):
        train_val_split(data, train_split=1.5)
    with pytest.raises(ValueError):
        train_val_split(data.view(2, 5), train_split=0.9)


def test_get_batch_shapes() -> None:
    """x and y have shape [batch_size, block_size] and long dtype."""
    data = make_data_tensor(list(range(200)))
    x, y = get_batch(data, batch_size=8, block_size=16)
    assert x.shape == (8, 16)
    assert y.shape == (8, 16)
    assert x.dtype == torch.long
    assert y.dtype == torch.long


def test_get_batch_y_is_x_shifted_by_one() -> None:
    """y equals x shifted one position into the future for every sample."""
    # Sequential data makes the shift relationship easy to verify.
    data = make_data_tensor(list(range(500)))
    gen = torch.Generator().manual_seed(0)
    x, y = get_batch(data, batch_size=16, block_size=32, generator=gen)
    # For sequential ids, the next char is always value + 1.
    assert torch.equal(y, x + 1)
    # And y[:, :-1] should match x[:, 1:] (overlapping window shifted by one).
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_get_batch_reproducible_with_generator() -> None:
    """Same seed -> same batch."""
    data = make_data_tensor(list(range(500)))
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    x1, y1 = get_batch(data, batch_size=4, block_size=8, generator=g1)
    x2, y2 = get_batch(data, batch_size=4, block_size=8, generator=g2)
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)


def test_get_batch_rejects_short_data() -> None:
    """Data not longer than block_size raises ValueError."""
    data = make_data_tensor(list(range(8)))
    with pytest.raises(ValueError):
        get_batch(data, batch_size=2, block_size=8)
