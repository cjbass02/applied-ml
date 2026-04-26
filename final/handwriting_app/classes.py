"""EMNIST byclass class list and small iteration helpers.

EMNIST ``byclass`` exposes 62 classes in this exact order:
  - Digits ``0`` through ``9`` (indices 0..9)
  - Uppercase letters ``A`` through ``Z`` (indices 10..35)
  - Lowercase letters ``a`` through ``z`` (indices 36..61)

The on-disk folder names match these strings, e.g. ``data/personal/train/A``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

DIGITS: List[str] = [str(i) for i in range(10)]
UPPERCASE: List[str] = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
LOWERCASE: List[str] = [chr(c) for c in range(ord("a"), ord("z") + 1)]

EMNIST_BYCLASS: List[str] = DIGITS + UPPERCASE + LOWERCASE
assert len(EMNIST_BYCLASS) == 62


def class_index(name: str) -> int:
    """Return the index of a class name in EMNIST byclass order."""
    return EMNIST_BYCLASS.index(name)


def existing_count(root: Path, split: str, class_name: str) -> int:
    """Return how many image files already exist for a (split, class)."""
    folder = root / split / class_name
    if not folder.is_dir():
        return 0
    n = 0
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        n += sum(1 for _ in folder.glob(pattern))
    return n
