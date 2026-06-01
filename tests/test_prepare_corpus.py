"""Lightweight tests for scripts/prepare_corpus.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the project's scripts/ directory importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_corpus import find_text_files, normalize_text, prepare_corpus


def test_normalize_text_line_endings() -> None:
    """CRLF and CR line endings are converted to LF."""
    assert normalize_text("a\r\nb\rc") == "a\nb\nc\n"


def test_normalize_text_collapses_whitespace() -> None:
    """Repeated spaces/tabs collapse and trailing spaces are stripped."""
    assert normalize_text("hello    world   ") == "hello world\n"


def test_normalize_text_collapses_blank_lines() -> None:
    """Three or more newlines collapse to a single blank line."""
    assert normalize_text("a\n\n\n\n\nb") == "a\n\nb\n"


def test_prepare_corpus_concatenates_sorted(tmp_path: Path) -> None:
    """Files are read in sorted order, cleaned, and written to the output path."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "b.txt").write_text("second\r\n", encoding="utf-8")
    (raw / "a.txt").write_text("first   line", encoding="utf-8")
    out = tmp_path / "processed" / "corpus.txt"

    cleaned = prepare_corpus(raw, out)

    assert out.exists()
    # a.txt sorts before b.txt, and whitespace is normalized.
    assert cleaned == "first line\n\nsecond\n"
    assert out.read_text(encoding="utf-8") == cleaned


def test_find_text_files_only_txt(tmp_path: Path) -> None:
    """Only .txt files are discovered."""
    (tmp_path / "keep.txt").write_text("x", encoding="utf-8")
    (tmp_path / "skip.md").write_text("y", encoding="utf-8")
    found = find_text_files(tmp_path)
    assert [p.name for p in found] == ["keep.txt"]


def test_prepare_corpus_raises_without_files(tmp_path: Path) -> None:
    """An empty input directory raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        prepare_corpus(tmp_path, tmp_path / "out.txt")
