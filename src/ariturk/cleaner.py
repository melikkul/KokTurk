"""High-level Turkish text cleaning interface."""
from __future__ import annotations

from ariturk.normalize import (
    is_valid_turkish,
    normalize_surface,
    restore_diacritics,
    turkish_lower,
)


class TextCleaner:
    """Turkish text cleaning pipeline.

    Args:
        lowercase: Apply Turkish-correct lowercasing.
        fix_diacritics: Attempt to restore missing ç,ğ,ı,ö,ş,ü.
        remove_punctuation: Strip all non-alphanumeric characters.
        min_word_length: Drop words shorter than this.
    """

    def __init__(
        self,
        lowercase: bool = True,
        fix_diacritics: bool = False,
        remove_punctuation: bool = False,
        min_word_length: int = 1,
    ) -> None:
        self.lowercase = lowercase
        self._fix_diacritics = fix_diacritics
        self.remove_punctuation = remove_punctuation
        self.min_word_length = min_word_length

    def clean(self, text: str) -> str:
        """Clean a single text string."""
        text = normalize_surface(text)
        if self.lowercase:
            text = turkish_lower(text)
        if self._fix_diacritics:
            text = restore_diacritics(text)
        if self.remove_punctuation:
            text = "".join(ch for ch in text if ch.isalnum() or ch.isspace())
        if self.min_word_length > 1:
            words = text.split()
            text = " ".join(w for w in words if len(w) >= self.min_word_length)
        return text.strip()

    def clean_batch(self, texts: list[str]) -> list[str]:
        """Clean a list of texts."""
        return [self.clean(t) for t in texts]

    def is_clean(self, text: str) -> bool:
        """Check if text is already clean Turkish."""
        return is_valid_turkish(text) and text == normalize_surface(text)
