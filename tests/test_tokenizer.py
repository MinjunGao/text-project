"""Lightweight tests for src/tokenizer.py (CharTokenizer)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the project's src/ directory importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokenizer import CharTokenizer

SAMPLE = "the quick brown fox jumps over the lazy dog.\nHELLO, World!"


def test_fit_builds_sorted_vocab() -> None:
    """fit() builds a deterministic, sorted character vocabulary."""
    tok = CharTokenizer().fit(SAMPLE)
    expected_chars = sorted(set(SAMPLE))
    # vocab_size includes the extra <unk> id.
    assert tok.vocab_size == len(expected_chars) + 1
    # The lowest ids correspond to the sorted unique characters.
    assert tok.decode(list(range(len(expected_chars)))) == "".join(expected_chars)


def test_encode_decode_roundtrip() -> None:
    """encode then decode returns the original in-vocabulary text."""
    tok = CharTokenizer().fit(SAMPLE)
    ids = tok.encode(SAMPLE)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert tok.decode(ids) == SAMPLE


def test_unknown_character_maps_to_unk() -> None:
    """Characters unseen during fit encode to the <unk> id and decode to unk_token."""
    tok = CharTokenizer().fit("abc")
    ids = tok.encode("az")  # 'z' is unknown
    assert ids == [tok.encode("a")[0], tok.vocab_size - 1]
    assert tok.decode(ids) == "a" + tok.unk_token


def test_save_and_load_preserves_behavior(tmp_path: Path) -> None:
    """save() then load() preserves vocab size and encode/decode behavior."""
    tok = CharTokenizer().fit(SAMPLE)
    path = tmp_path / "vocab.json"
    tok.save(path)

    loaded = CharTokenizer.load(path)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.encode(SAMPLE) == tok.encode(SAMPLE)
    assert loaded.decode(loaded.encode(SAMPLE)) == SAMPLE


def test_fit_empty_text_raises() -> None:
    """Fitting on empty text raises ValueError."""
    with pytest.raises(ValueError):
        CharTokenizer().fit("")


def test_encode_before_fit_raises() -> None:
    """Using the tokenizer before fitting raises RuntimeError."""
    with pytest.raises(RuntimeError):
        CharTokenizer().encode("abc")
