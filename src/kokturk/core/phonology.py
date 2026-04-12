"""Turkish vowel harmony and phonological normalization.

Turkish has two-dimensional vowel harmony:
- Front/back: e,i,ö,ü (front) vs a,ı,o,u (back)
- Rounded/unrounded

Suffix allomorphs are selected based on the last vowel of the stem.
This module normalizes surface allomorphs to canonical forms:
e.g., -da/-de/-ta/-te → +LOC
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HarmonyResult:
    """Result of a vowel harmony check."""

    ok: bool
    severity: str  # "none" | "warning" | "error"


FRONT_VOWELS = frozenset("eiöü")
BACK_VOWELS = frozenset("aıou")
ALL_VOWELS = FRONT_VOWELS | BACK_VOWELS

FRONT_UNROUNDED = frozenset("ei")
FRONT_ROUNDED = frozenset("öü")
BACK_UNROUNDED = frozenset("aı")
BACK_ROUNDED = frozenset("ou")

# Voicing alternation: stem-final consonant changes before vowel-initial suffix
VOICING_MAP: dict[str, str] = {
    "p": "b",
    "ç": "c",
    "t": "d",
    "k": "ğ",
}


def last_vowel(word: str) -> str | None:
    """Return the last vowel in a Turkish word.

    Scans the word right-to-left (case-insensitive) and returns the
    first vowel found. Used to determine vowel harmony class.

    Args:
        word: A Turkish word (any case).

    Returns:
        The last vowel character (lowercase), or None if the word
        contains no vowels.
    """
    for char in reversed(word.lower()):
        if char in ALL_VOWELS:
            return char
    return None


def is_front(word: str) -> bool:
    """Check if a word's last vowel is front (e, i, ö, ü).

    Front-vowel stems take front-vowel suffix variants:
    ``ev`` + LOC → ``evde`` (front ``e``), not ``evda``.

    Args:
        word: A Turkish word.

    Returns:
        True if the last vowel is front, False if back or no vowels.
    """
    v = last_vowel(word)
    return v is not None and v in FRONT_VOWELS


def is_back(word: str) -> bool:
    """Check if a word's last vowel is back (a, ı, o, u).

    Back-vowel stems take back-vowel suffix variants:
    ``araba`` + LOC → ``arabada`` (back ``a``), not ``arabade``.

    Args:
        word: A Turkish word.

    Returns:
        True if the last vowel is back, False if front or no vowels.
    """
    v = last_vowel(word)
    return v is not None and v in BACK_VOWELS


def is_rounded(word: str) -> bool:
    """Check if a word's last vowel is rounded (ö, ü, o, u).

    Rounded stems select rounded suffix allomorphs in four-way harmony:
    ``göz`` + DAT → ``göze`` (front rounded → front unrounded for DAT).

    Args:
        word: A Turkish word.

    Returns:
        True if the last vowel is rounded, False otherwise.
    """
    v = last_vowel(word)
    return v is not None and v in (FRONT_ROUNDED | BACK_ROUNDED)


def check_vowel_harmony(word: str) -> HarmonyResult:
    """Check 4-way Turkish vowel harmony.  Returns severity level.

    Two-way (front/back) violation is almost never valid in native Turkish
    words and is reported as ``"error"``.  Four-way (rounded/unrounded)
    violations occur in loanwords and are reported as ``"warning"``.

    Args:
        word: A Turkish word.

    Returns:
        ``HarmonyResult(ok, severity)`` where *severity* is one of
        ``"none"`` (harmony holds), ``"warning"`` (4-way only violation),
        or ``"error"`` (2-way front/back violation).
    """
    vowels = [ch for ch in word.lower() if ch in ALL_VOWELS]
    if len(vowels) <= 1:
        return HarmonyResult(ok=True, severity="none")

    # 2-way check: all vowels must share the same front/back class
    first_front = vowels[0] in FRONT_VOWELS
    for v in vowels[1:]:
        if (v in FRONT_VOWELS) != first_front:
            return HarmonyResult(ok=False, severity="error")

    # 4-way check: rounded/unrounded consistency
    rounded_set = FRONT_ROUNDED | BACK_ROUNDED
    first_rounded = vowels[0] in rounded_set
    for v in vowels[1:]:
        if (v in rounded_set) != first_rounded:
            return HarmonyResult(ok=False, severity="warning")

    return HarmonyResult(ok=True, severity="none")
