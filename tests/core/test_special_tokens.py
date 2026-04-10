"""Tests for the special-token preprocessor (Cat B Task 2)."""

from __future__ import annotations

from kokturk.core.special_token_types import SpecialTokenResult
from kokturk.core.special_tokens import (
    decompose_reduplication,
    get_abbreviation_final_vowel,
    get_numeric_final_vowel,
    preprocess_special_token,
    split_abbreviation_suffix,
    split_numeric_suffix,
)


# ----------------------- Abbreviations ------------------------------------


class TestAbbreviations:
    def test_acronym_word_pronunciation(self):
        # NATO is read /nato/ → final vowel 'o'.
        assert get_abbreviation_final_vowel("NATO") == "o"

    def test_letter_pronunciation_for_initialism(self):
        # TDK: T-D-K → 'ke' → final vowel 'e'.
        assert get_abbreviation_final_vowel("TDK") == "e"

    def test_unknown_letter_returns_none_or_vowel(self):
        # Just ensure it doesn't crash and returns a vowel or None.
        result = get_abbreviation_final_vowel("XYZ")
        assert result is None or result in "aeıioöuü"

    def test_split_apostrophe(self):
        assert split_abbreviation_suffix("NATO'nun") == ("NATO", "nun")
        assert split_abbreviation_suffix("TDK'ye") == ("TDK", "ye")

    def test_split_period_ending(self):
        assert split_abbreviation_suffix("Alm.lar") == ("Alm.", "lar")

    def test_split_missing_apostrophe(self):
        assert split_abbreviation_suffix("NATOnun") == ("NATO", "nun")

    def test_dispatch_natonu(self):
        result = preprocess_special_token("NATO'nun")
        assert isinstance(result, SpecialTokenResult)
        assert result.token_type == "abbreviation"
        assert result.base == "NATO"
        assert result.suffix_part == "nun"
        assert result.harmony_vowel == "o"

    def test_dispatch_tdkye(self):
        result = preprocess_special_token("TDK'ye")
        assert result is not None
        assert result.token_type == "abbreviation"
        assert result.base == "TDK"
        assert result.harmony_vowel == "e"


# ----------------------- Numerics -----------------------------------------


class TestNumerics:
    def test_1990_doksan_back_vowel(self):
        # 1990 → "doksan" → final 'a'.
        assert get_numeric_final_vowel("1990") == "a"

    def test_3_uc_front_rounded(self):
        assert get_numeric_final_vowel("3") == "ü"

    def test_split_1990larda(self):
        assert split_numeric_suffix("1990'larda") == ("1990", "larda")

    def test_split_3un(self):
        assert split_numeric_suffix("3'ün") == ("3", "ün")

    def test_dispatch_1990larda(self):
        result = preprocess_special_token("1990'larda")
        assert result is not None
        assert result.token_type == "numeric"
        assert result.base == "1990"
        assert result.suffix_part == "larda"
        assert result.harmony_vowel == "a"

    def test_dispatch_bare_2024(self):
        result = preprocess_special_token("2024")
        assert result is not None
        assert result.token_type == "numeric"
        assert result.base == "2024"
        # 2024 ends in 4 → "dört" → 'ö'
        assert result.harmony_vowel == "ö"


# ----------------------- Reduplication ------------------------------------


class TestReduplication:
    def test_lexicon_masmavi(self):
        result = decompose_reduplication("masmavi")
        assert result == ("mavi", "s")

    def test_lexicon_tertemiz(self):
        result = decompose_reduplication("tertemiz")
        assert result == ("temiz", "r")

    def test_non_redup_returns_none(self):
        # "evlerinden" is just a regular noun, not a reduplication.
        assert decompose_reduplication("evlerinden") is None

    def test_dispatch_masmavi(self):
        result = preprocess_special_token("masmavi")
        assert result is not None
        assert result.token_type == "reduplication"
        assert result.base == "mavi"
        assert result.harmony_vowel == "i"


# ----------------------- Dispatcher fallthrough ---------------------------


class TestDispatcherFallthrough:
    def test_normal_word_returns_none(self):
        assert preprocess_special_token("evlerinden") is None
        assert preprocess_special_token("geliyor") is None

    def test_empty_returns_none(self):
        assert preprocess_special_token("") is None


# ----------------------- Analyzer integration -----------------------------


def test_analyzer_handle_special_tokens_flag(monkeypatch):
    from kokturk.core import analyzer as analyzer_module
    from kokturk.core.datatypes import MorphologicalAnalysis

    class FakeBackend:
        def analyze(self, word):
            return [
                MorphologicalAnalysis(
                    surface=word,
                    root=word,
                    tags=("+OOV",),
                    morphemes=(),
                    source="fake",
                    score=1.0,
                )
            ]

        def close(self):
            pass

    monkeypatch.setitem(
        analyzer_module._BACKEND_REGISTRY, "fake", FakeBackend
    )
    a = analyzer_module.MorphoAnalyzer(backends=["fake"])

    # Default off → backend handles the token opaquely.
    default = a.analyze("NATO'nun")
    assert default.analyses[0].root == "NATO'nun"

    # Opt-in → preprocessor takes over.
    special = a.analyze("NATO'nun", handle_special_tokens=True)
    assert special.analyses[0].root == "NATO"
    assert special.analyses[0].tags[0] == "+ABBREVIATION"
