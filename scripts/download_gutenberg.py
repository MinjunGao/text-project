"""Download public-domain plain-text books from Project Gutenberg.

All books referenced here are in the **public domain** (their copyright has
expired), distributed by Project Gutenberg (https://www.gutenberg.org/). They are
suitable for training a model from scratch without licensing concerns. Please be
considerate of Project Gutenberg's servers: only download what you need.

This helper uses the Python standard library only (``urllib``) so it has no extra
dependencies. Downloaded files are saved as UTF-8 ``.txt`` into the output
directory (default: ``data/raw``).

Example:
    # Download the default set (alice, holmes, grimm, poe)
    uv run python scripts/download_gutenberg.py --output_dir data/raw

    # Download a subset
    uv run python scripts/download_gutenberg.py --books alice grimm
"""

from __future__ import annotations

import argparse
import urllib.error
import urllib.request
from pathlib import Path

# Registry of known public-domain books -> (Project Gutenberg plain-text URL,
# output filename). The "cache/epub/<id>/pg<id>.txt" form is the canonical UTF-8
# plain-text version of each work.
BOOKS: dict[str, tuple[str, str]] = {
    # Alice's Adventures in Wonderland by Lewis Carroll (eBook #11)
    "alice": ("https://www.gutenberg.org/cache/epub/11/pg11.txt", "alice.txt"),
    # The Adventures of Sherlock Holmes by Arthur Conan Doyle (eBook #1661)
    "holmes": ("https://www.gutenberg.org/cache/epub/1661/pg1661.txt", "holmes.txt"),
    # Grimms' Fairy Tales by Jacob & Wilhelm Grimm (eBook #2591)
    "grimm": ("https://www.gutenberg.org/cache/epub/2591/pg2591.txt", "grimm.txt"),
    # The Works of Edgar Allan Poe, Volume 1 (eBook #2147)
    "poe": ("https://www.gutenberg.org/cache/epub/2147/pg2147.txt", "poe.txt"),
}

# A descriptive User-Agent; some servers reject requests without one.
USER_AGENT = "text-project-gutenberg-downloader/0.1 (educational; from-scratch GPT)"

# Network timeout in seconds for each download.
TIMEOUT_SECONDS = 30


def download_book(key: str, output_dir: Path) -> bool:
    """Download a single book by its registry ``key`` into ``output_dir``.

    Args:
        key: A key from :data:`BOOKS` (e.g. ``"alice"``).
        output_dir: Directory to write the downloaded ``.txt`` file into.

    Returns:
        ``True`` if the file was downloaded and saved, ``False`` otherwise. Network
        and file errors are caught and reported rather than raised, so a single
        failure does not abort a multi-book download.
    """
    if key not in BOOKS:
        print(f"[skip] Unknown book '{key}'. Known books: {', '.join(sorted(BOOKS))}")
        return False

    url, filename = BOOKS[key]
    dest = output_dir / filename
    print(f"[get ] {key}: {url}")

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        print(f"[fail] {key}: HTTP error {exc.code} ({exc.reason}). Skipping.")
        return False
    except urllib.error.URLError as exc:
        print(f"[fail] {key}: network error ({exc.reason}). Check your connection.")
        return False
    except TimeoutError:
        print(f"[fail] {key}: timed out after {TIMEOUT_SECONDS}s. Skipping.")
        return False

    text = raw.decode("utf-8", errors="replace")

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
    except OSError as exc:
        print(f"[fail] {key}: could not write {dest} ({exc}). Skipping.")
        return False

    print(f"[ok  ] {key}: saved {len(text)} characters to {dest}")
    return True


def download_books(keys: list[str], output_dir: Path) -> tuple[int, int]:
    """Download several books, continuing past individual failures.

    Args:
        keys: Registry keys to download.
        output_dir: Directory to save files into.

    Returns:
        A ``(succeeded, failed)`` count tuple.
    """
    succeeded = 0
    for key in keys:
        if download_book(key, output_dir):
            succeeded += 1
    return succeeded, len(keys) - succeeded


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``). Useful for tests.

    Returns:
        Parsed arguments with ``output_dir`` (``Path``) and ``books`` (list[str]).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to save downloaded .txt files (default: data/raw).",
    )
    parser.add_argument(
        "--books",
        nargs="+",
        choices=sorted(BOOKS),
        default=sorted(BOOKS),
        metavar="BOOK",
        help=(
            "Which books to download. Choices: "
            f"{', '.join(sorted(BOOKS))}. Default: all."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (0 on full success)."""
    args = parse_args(argv)
    print(
        "Downloading public-domain texts from Project Gutenberg "
        f"({len(args.books)} book(s)) into {args.output_dir}\n"
    )
    succeeded, failed = download_books(args.books, args.output_dir)
    print(f"\nDone. {succeeded} succeeded, {failed} failed.")
    if failed:
        print("Some downloads failed; see messages above. You can re-run to retry.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
