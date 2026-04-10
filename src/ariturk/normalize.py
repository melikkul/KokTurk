"""Turkish text normalization utilities."""
from __future__ import annotations

import re
import unicodedata


def normalize_surface(text: str) -> str:
    """Full normalization pipeline for a surface form.

    Applies NFC normalization, strips whitespace, and collapses internal runs.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def turkish_lower(text: str) -> str:
    """Lowercase with Turkish rules: I→ı, İ→i."""
    result: list[str] = []
    for ch in text:
        if ch == "I":
            result.append("ı")
        elif ch == "\u0130":  # İ
            result.append("i")
        else:
            result.append(ch.lower())
    return "".join(result)


def turkish_upper(text: str) -> str:
    """Uppercase with Turkish rules: i→İ, ı→I."""
    result: list[str] = []
    for ch in text:
        if ch == "i":
            result.append("\u0130")  # İ
        elif ch == "ı":
            result.append("I")
        else:
            result.append(ch.upper())
    return "".join(result)


_TURKISH_CHARS = frozenset(
    "abcçdefgğhıijklmnoöprsştuüvyz"
    "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ"
    "0123456789'-. "
)


def is_valid_turkish(text: str) -> bool:
    """Check if text contains only valid Turkish characters."""
    return all(ch in _TURKISH_CHARS for ch in text)


# Common ASCII→diacritics fixes
_DIACRITIC_FIXES: dict[str, str] = {
    "turkce": "türkçe",
    "turk": "türk",
    "turkiye": "türkiye",
    "ogrenci": "öğrenci",
    "ogretmen": "öğretmen",
    "universite": "üniversite",
    "guzel": "güzel",
    "buyuk": "büyük",
    "kucuk": "küçük",
    "calisma": "çalışma",
    "islem": "işlem",
}


def restore_diacritics(text: str) -> str:
    """Attempt to restore missing Turkish diacritics using a lookup table."""
    words = text.split()
    return " ".join(_DIACRITIC_FIXES.get(w.lower(), w) for w in words)
