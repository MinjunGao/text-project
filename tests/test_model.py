"""Lightweight tests for src/model.py (TinyGPT)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Make the project's src/ directory importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model import CausalSelfAttention, GPTConfig, TinyGPT


def make_config(**overrides: int) -> GPTConfig:
    """Small config for fast tests."""
    params: dict = dict(
        vocab_size=17, block_size=16, n_embd=32, n_head=4, n_layer=2, dropout=0.0
    )
    params.update(overrides)
    return GPTConfig(**params)


def test_forward_pass_shape() -> None:
    """Logits have shape [B, T, vocab_size]."""
    config = make_config()
    model = TinyGPT(config)
    B, T = 3, 8
    idx = torch.randint(0, config.vocab_size, (B, T))
    logits, loss = model(idx)
    assert logits.shape == (B, T, config.vocab_size)
    assert loss is None


def test_loss_is_computed_with_targets() -> None:
    """Providing targets yields a scalar cross-entropy loss that can backprop."""
    config = make_config()
    model = TinyGPT(config)
    B, T = 4, 8
    idx = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))
    logits, loss = model(idx, targets)
    assert loss is not None
    assert loss.ndim == 0  # scalar
    assert loss.item() > 0
    # Loss should be roughly -ln(1/vocab_size) at init; just sanity-check finite.
    assert torch.isfinite(loss)
    loss.backward()  # gradients flow without error


def test_generate_returns_expected_length() -> None:
    """generate appends exactly max_new_tokens tokens."""
    config = make_config()
    model = TinyGPT(config)
    B, T = 2, 4
    idx = torch.randint(0, config.vocab_size, (B, T))
    out = model.generate(idx, max_new_tokens=10)
    assert out.shape == (B, T + 10)
    # All generated ids are valid vocabulary indices.
    assert int(out.min()) >= 0
    assert int(out.max()) < config.vocab_size


def test_generate_respects_block_size() -> None:
    """Generation works even when context already exceeds block_size."""
    config = make_config(block_size=8)
    model = TinyGPT(config)
    idx = torch.randint(0, config.vocab_size, (1, 8))
    out = model.generate(idx, max_new_tokens=5, top_k=3)
    assert out.shape == (1, 13)


def test_forward_rejects_too_long_sequence() -> None:
    """A sequence longer than block_size raises ValueError."""
    config = make_config(block_size=8)
    model = TinyGPT(config)
    idx = torch.randint(0, config.vocab_size, (1, 9))
    with pytest.raises(ValueError):
        model(idx)


def test_causal_mask_blocks_future_tokens() -> None:
    """Changing a future token must not change earlier positions' outputs."""
    config = make_config(dropout=0.0)
    model = TinyGPT(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (1, 8))
    with torch.no_grad():
        logits_a, _ = model(idx)
        # Modify only the last token.
        idx_mod = idx.clone()
        idx_mod[0, -1] = (idx_mod[0, -1] + 1) % config.vocab_size
        logits_b, _ = model(idx_mod)
    # Outputs at all positions before the last must be identical (no leakage).
    assert torch.allclose(logits_a[:, :-1, :], logits_b[:, :-1, :], atol=1e-5)


def test_attention_head_divisibility_validated() -> None:
    """n_embd not divisible by n_head raises ValueError."""
    with pytest.raises(ValueError):
        CausalSelfAttention(make_config(n_embd=30, n_head=4))
