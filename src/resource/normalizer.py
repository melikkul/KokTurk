"""Turkish text normalization for surface forms and canonical tags."""
from __future__ import annotations
import unicodedata

# Turkish-specific Unicode normalization: homoglyphs that look similar but differ
_CHAR_NORM: dict[str, str] = {
    "â": "a",   # circumflex a → plain a (some Ottoman-era words)
    "î": "i",   # circumflex i
    "û": "u",   # circumflex u
    "\u0131": "ı",  # already correct dotless i, keep
}

def normalize_surface(s: str) -> str:
    """Normalize a Turkish surface form.

    Applies NFC Unicode normalization and lowercases.
    Handles Turkish-specific characters (î→i, â→a, û→u).

    Args:
        s: Raw surface form string.

    Returns:
        Normalized lowercase string.
    """
    s = unicodedata.normalize("NFC", s)
    # Replace specific homoglyphs before lowercasing
    for src, dst in _CHAR_NORM.items():
        s = s.replace(src, dst)
    return s.lower()


def normalize_canonical(tags: str) -> str:
    """Normalize a canonical tag string.

    Strips leading/trailing whitespace and normalizes internal whitespace
    to single spaces. Does NOT re-order tags (Turkish morphotactics are
    non-commutative).

    Args:
        tags: Canonical tag string like "ev +PLU +POSS.3SG +ABL".

    Returns:
        Whitespace-normalized string.
    """
    return " ".join(tags.split())
