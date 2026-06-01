"""Prepare a cleaned training corpus from raw text files.

This script reads one or more ``.txt`` files from an input directory, concatenates
them, normalizes line endings, removes excessive whitespace, and writes the result
to a single output file.

It does NOT do tokenization or any model-related work; it only produces a clean
plain-text corpus that later steps can consume.

Example:
    uv run python scripts/prepare_corpus.py \\
        --input_dir data/raw \\
        --output_path data/processed/corpus.txt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def find_text_files(input_dir: Path) -> list[Path]:
    """Return a sorted list of ``.txt`` files in ``input_dir``.

    Args:
        input_dir: Directory to search (non-recursively) for ``.txt`` files.

    Returns:
        Sorted list of paths to ``.txt`` files. Sorting keeps concatenation order
        deterministic across runs and machines.

    Raises:
        FileNotFoundError: If ``input_dir`` does not exist or is not a directory.
    """
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    return sorted(p for p in input_dir.glob("*.txt") if p.is_file())


def normalize_text(text: str) -> str:
    """Normalize line endings and strip excessive whitespace from ``text``.

    The cleaning steps are:
      * Convert Windows (``\\r\\n``) and old Mac (``\\r``) line endings to ``\\n``.
      * Strip trailing spaces/tabs from each line.
      * Collapse runs of spaces/tabs into a single space.
      * Collapse 3 or more consecutive newlines into exactly 2 (one blank line),
        preserving paragraph breaks.
      * Strip leading/trailing whitespace from the whole document and end with a
        single trailing newline.

    Args:
        text: Raw input text.

    Returns:
        The cleaned text.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def prepare_corpus(input_dir: Path, output_path: Path) -> str:
    """Build a cleaned corpus from all ``.txt`` files in ``input_dir``.

    Reads every ``.txt`` file in ``input_dir`` (sorted by name), concatenates them
    with a blank line between files, normalizes the combined text, writes it to
    ``output_path``, and returns the cleaned text.

    Args:
        input_dir: Directory containing raw ``.txt`` files.
        output_path: Destination file for the cleaned corpus. Parent directories
            are created if needed.

    Returns:
        The cleaned corpus text that was written to ``output_path``.

    Raises:
        FileNotFoundError: If ``input_dir`` has no ``.txt`` files.
    """
    files = find_text_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No .txt files found in: {input_dir}")

    parts = [path.read_text(encoding="utf-8", errors="replace") for path in files]
    combined = "\n\n".join(parts)
    cleaned = normalize_text(combined)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(cleaned, encoding="utf-8")
    return cleaned


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``). Useful for tests.

    Returns:
        Parsed arguments with ``input_dir`` and ``output_path`` as ``Path`` objects.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw .txt files (default: data/raw).",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("data/processed/corpus.txt"),
        help="Output path for the cleaned corpus (default: data/processed/corpus.txt).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: prepare the corpus and print a short summary."""
    args = parse_args(argv)
    cleaned = prepare_corpus(args.input_dir, args.output_path)
    files = find_text_files(args.input_dir)
    print(f"Read {len(files)} file(s) from {args.input_dir}")
    print(f"Wrote {len(cleaned)} characters to {args.output_path}")


if __name__ == "__main__":
    main()
