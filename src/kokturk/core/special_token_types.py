"""Frozen dataclass for special-token preprocessor results."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SpecialTokenResult:
    """Output of :func:`kokturk.core.special_tokens.preprocess_special_token`.

    Attributes:
        token_type: ``"abbreviation"``, ``"numeric"``, or ``"reduplication"``.
        base: the base form (NATO, 1990, mavi, …).
        suffix_part: the Turkish suffix attached to the base (without the
            apostrophe), or empty string if no suffix.
        harmony_vowel: the vowel that should drive vowel harmony for any
            attached suffix (one of ``a/e/ı/i/o/ö/u/ü``), or empty string
            if not determinable.
    """

    token_type: str
    base: str
    suffix_part: str
    harmony_vowel: str
