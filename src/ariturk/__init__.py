"""arı-türk: Turkish Text Cleaning & Normalization Library.

Features:
    - Turkish-correct case handling (I→ı, İ→i)
    - Unicode NFC normalization
    - Diacritics restoration
    - Quality tier assignment (gold/silver/bronze)
    - Morpheme boundary extraction

Example::

    >>> from ariturk import TextCleaner
    >>> cleaner = TextCleaner()
    >>> cleaner.clean("  TÜRKÇE   metİn  ")
    'türkçe metin'
"""
from __future__ import annotations

__version__ = "0.1.0"
__all__ = [
    "TextCleaner",
    "QualityChecker",
    "BoundaryExtractor",
    "normalize_surface",
    "turkish_lower",
    "turkish_upper",
    "is_valid_turkish",
]

from ariturk.normalize import (
    normalize_surface,
    turkish_lower,
    turkish_upper,
    is_valid_turkish,
)
from ariturk.cleaner import TextCleaner
from ariturk.quality import QualityChecker
from ariturk.boundaries import BoundaryExtractor
