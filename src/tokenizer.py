"""A simple, deterministic character-level tokenizer.

The tokenizer builds its vocabulary from the unique characters in a training text,
sorted so the mapping is reproducible across runs and machines. Each character maps
to an integer id; an extra ``<unk>`` id handles characters not seen during ``fit``.

This module has no model dependencies; it only turns text into integer ids and back.
"""

from __future__ import annotations

import json
from pathlib import Path

# Default placeholder used for characters not present in the vocabulary. The
# Unicode replacement character is used because it is extremely unlikely to appear
# in ordinary public-domain text, so it will not collide with real characters.
DEFAULT_UNK_TOKEN = "\ufffd"


class CharTokenizer:
    """Character-level tokenizer with a deterministic, sorted vocabulary.

    Known characters are assigned ids ``0 .. n-1`` in sorted order. A single
    ``<unk>`` id (``n``) represents any character not seen during :meth:`fit`, so
    :attr:`vocab_size` is ``n + 1``.

    Attributes:
        unk_token: The string used to represent unknown ids when decoding.
    """

    def __init__(self, unk_token: str = DEFAULT_UNK_TOKEN) -> None:
        self.unk_token = unk_token
        self._chars: list[str] = []
        self._stoi: dict[str, int] = {}
        self._itos: dict[int, str] = {}

    @property
    def _unk_id(self) -> int:
        """Id reserved for unknown characters (one past the known characters)."""
        return len(self._chars)

    @property
    def vocab_size(self) -> int:
        """Number of ids the tokenizer can emit, including the ``<unk>`` id."""
        return len(self._chars) + 1

    @property
    def is_fitted(self) -> bool:
        """Whether the tokenizer has been fitted (or loaded) with a vocabulary."""
        return len(self._chars) > 0

    def fit(self, text: str) -> "CharTokenizer":
        """Build the vocabulary from the unique characters in ``text``.

        Args:
            text: Training text to derive the character vocabulary from.

        Returns:
            ``self``, to allow chaining (e.g. ``CharTokenizer().fit(text)``).

        Raises:
            ValueError: If ``text`` is empty (no characters to build a vocab from).
        """
        if not text:
            raise ValueError("Cannot fit CharTokenizer on empty text.")
        self._chars = sorted(set(text))
        self._stoi = {ch: i for i, ch in enumerate(self._chars)}
        self._itos = {i: ch for i, ch in enumerate(self._chars)}
        return self

    def encode(self, text: str) -> list[int]:
        """Encode ``text`` into a list of integer ids.

        Characters not in the vocabulary are mapped to the ``<unk>`` id.

        Args:
            text: Text to encode.

        Returns:
            List of integer ids.

        Raises:
            RuntimeError: If the tokenizer has not been fitted or loaded.
        """
        self._require_fitted()
        return [self._stoi.get(ch, self._unk_id) for ch in text]

    def decode(self, ids: list[int]) -> str:
        """Decode a list of integer ids back into text.

        Unknown ids (the ``<unk>`` id or any out-of-range id) decode to
        :attr:`unk_token`.

        Args:
            ids: Integer ids to decode.

        Returns:
            The decoded string.

        Raises:
            RuntimeError: If the tokenizer has not been fitted or loaded.
        """
        self._require_fitted()
        return "".join(self._itos.get(i, self.unk_token) for i in ids)

    def save(self, path: str | Path) -> None:
        """Save the vocabulary to a JSON file.

        Args:
            path: Destination file path. Parent directories are created if needed.

        Raises:
            RuntimeError: If the tokenizer has not been fitted or loaded.
        """
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"chars": self._chars, "unk_token": self.unk_token}
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        """Load a tokenizer previously written by :meth:`save`.

        Args:
            path: Path to a JSON file produced by :meth:`save`.

        Returns:
            A fitted :class:`CharTokenizer` instance.
        """
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        tokenizer = cls(unk_token=payload.get("unk_token", DEFAULT_UNK_TOKEN))
        tokenizer._chars = list(payload["chars"])
        tokenizer._stoi = {ch: i for i, ch in enumerate(tokenizer._chars)}
        tokenizer._itos = {i: ch for i, ch in enumerate(tokenizer._chars)}
        return tokenizer

    def _require_fitted(self) -> None:
        """Raise if the tokenizer has no vocabulary yet."""
        if not self.is_fitted:
            raise RuntimeError("CharTokenizer is not fitted. Call fit() or load() first.")
