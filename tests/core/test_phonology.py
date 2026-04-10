"""Tests for Turkish phonology utilities."""

from __future__ import annotations

from kokturk.core.phonology import is_back, is_front, is_rounded, last_vowel


class TestLastVowel:
    def test_simple(self) -> None:
        assert last_vowel("ev") == "e"
        assert last_vowel("araba") == "a"

    def test_no_vowels(self) -> None:
        assert last_vowel("brç") is None

    def test_turkish_vowels(self) -> None:
        assert last_vowel("gül") == "ü"
        assert last_vowel("kız") == "ı"
        assert last_vowel("göz") == "ö"


class TestVowelHarmony:
    def test_front(self) -> None:
        assert is_front("ev") is True
        assert is_front("gül") is True

    def test_back(self) -> None:
        assert is_back("araba") is True
        assert is_back("kız") is True

    def test_rounded(self) -> None:
        assert is_rounded("göz") is True
        assert is_rounded("kul") is True
        assert is_rounded("ev") is False
