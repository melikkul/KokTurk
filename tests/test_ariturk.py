"""Tests for the arı-türk (ariturk) library."""
from __future__ import annotations


def test_imports():
    from ariturk import (  # noqa: F401
        TextCleaner,
        QualityChecker,
        BoundaryExtractor,
        normalize_surface,
        turkish_lower,
        turkish_upper,
        is_valid_turkish,
    )


def test_version():
    import ariturk
    assert ariturk.__version__ == "0.1.0"


# --- normalize ---


class TestTurkishLower:
    def test_dotless_i(self):
        from ariturk import turkish_lower
        assert turkish_lower("I") == "ı"

    def test_dotted_i(self):
        from ariturk import turkish_lower
        assert turkish_lower("İ") == "i"

    def test_turkce(self):
        from ariturk import turkish_lower
        assert turkish_lower("TÜRKÇE") == "türkçe"

    def test_mixed(self):
        from ariturk import turkish_lower
        assert turkish_lower("İSTANBUL") == "istanbul"


class TestTurkishUpper:
    def test_dotted_i(self):
        from ariturk import turkish_upper
        assert turkish_upper("i") == "İ"

    def test_dotless_i(self):
        from ariturk import turkish_upper
        assert turkish_upper("ı") == "I"


class TestNormalize:
    def test_strip_whitespace(self):
        from ariturk import normalize_surface
        assert normalize_surface("  hello  ") == "hello"

    def test_collapse_internal_spaces(self):
        from ariturk import normalize_surface
        assert normalize_surface("a   b   c") == "a b c"


class TestIsValidTurkish:
    def test_valid(self):
        from ariturk import is_valid_turkish
        assert is_valid_turkish("Türkçe metin")

    def test_invalid_special_chars(self):
        from ariturk import is_valid_turkish
        assert not is_valid_turkish("text with @#$ symbols")


# --- TextCleaner ---


class TestTextCleaner:
    def test_basic_clean(self):
        from ariturk import TextCleaner
        c = TextCleaner()
        assert c.clean("  TÜRKÇE   metİn  ") == "türkçe metin"

    def test_no_lowercase(self):
        from ariturk import TextCleaner
        c = TextCleaner(lowercase=False)
        result = c.clean("  TÜRKÇE  ")
        assert result == "TÜRKÇE"

    def test_remove_punctuation(self):
        from ariturk import TextCleaner
        c = TextCleaner(remove_punctuation=True)
        result = c.clean("merhaba, dünya!")
        assert result == "merhaba dünya"

    def test_min_word_length(self):
        from ariturk import TextCleaner
        c = TextCleaner(min_word_length=3)
        result = c.clean("bu bir test")
        assert result == "bir test"

    def test_clean_batch(self):
        from ariturk import TextCleaner
        c = TextCleaner()
        results = c.clean_batch(["  A  ", "  B  "])
        assert results == ["a", "b"]

    def test_is_clean(self):
        from ariturk import TextCleaner
        c = TextCleaner()
        assert c.is_clean("türkçe metin")


# --- QualityChecker ---


class TestQualityChecker:
    def test_gold_from_boun(self):
        from ariturk import QualityChecker
        qc = QualityChecker()
        assert qc.assign_tier(["boun", "zeyrek"], tags_agree=True) == "gold"

    def test_gold_from_imst(self):
        from ariturk import QualityChecker
        qc = QualityChecker()
        assert qc.assign_tier(["imst"], tags_agree=False) == "gold"

    def test_silver(self):
        from ariturk import QualityChecker
        qc = QualityChecker()
        assert qc.assign_tier(["unimorph", "zeyrek"], tags_agree=True) == "silver"

    def test_bronze_single_source(self):
        from ariturk import QualityChecker
        qc = QualityChecker()
        assert qc.assign_tier(["zeyrek"], tags_agree=False) == "bronze"

    def test_bronze_disagree(self):
        from ariturk import QualityChecker
        qc = QualityChecker()
        assert qc.assign_tier(["unimorph", "zeyrek"], tags_agree=False) == "bronze"

    def test_validate_ok(self):
        from ariturk import QualityChecker
        qc = QualityChecker()
        errors = qc.validate_entry("ev", "ev +NOM", "NOUN")
        assert errors == []

    def test_validate_missing_prefix(self):
        from ariturk import QualityChecker
        qc = QualityChecker()
        errors = qc.validate_entry("ev", "ev NOM", "NOUN")
        assert len(errors) == 1
        assert "missing + prefix" in errors[0]

    def test_validate_empty(self):
        from ariturk import QualityChecker
        qc = QualityChecker()
        errors = qc.validate_entry("", "", "")
        assert len(errors) == 2


# --- BoundaryExtractor ---


class TestBoundaryExtractor:
    def test_simple_plural(self):
        from ariturk import BoundaryExtractor
        ext = BoundaryExtractor()
        result = ext.extract("evler", "ev +PLU")
        assert result == "ev|ler"

    def test_possessive_ablative(self):
        from ariturk import BoundaryExtractor
        ext = BoundaryExtractor()
        result = ext.extract("evlerinden", "ev +PLU +POSS.3SG +ABL")
        assert "|" in result
        assert result.startswith("ev|")

    def test_no_suffix(self):
        from ariturk import BoundaryExtractor
        ext = BoundaryExtractor()
        result = ext.extract("ev", "ev")
        assert result == "ev"

    def test_extract_batch(self):
        from ariturk import BoundaryExtractor
        ext = BoundaryExtractor()
        results = ext.extract_batch([("evler", "ev +PLU"), ("ev", "ev")])
        assert len(results) == 2
        assert results[0] == "ev|ler"
