"""Tests for code-switch preprocessing."""

from __future__ import annotations

import pytest

from kokturk.core.analyzer import MorphoAnalyzer
from kokturk.core.code_switch import (
    analyze_foreign_suffixes,
    classify_foreign_root,
    detect_code_switch,
    split_foreign_suffix,
)
from kokturk.core.datatypes import TokenAnalyses

# -----------------------------------------------------------------------
# split_foreign_suffix
# -----------------------------------------------------------------------


class TestSplitForeignSuffix:
    def test_ascii_apostrophe(self) -> None:
        assert split_foreign_suffix("Google'ladım") == ("Google", "ladım")

    def test_unicode_right_quote(self) -> None:
        assert split_foreign_suffix("Google\u2019ladım") == ("Google", "ladım")

    def test_unicode_modifier_apostrophe(self) -> None:
        assert split_foreign_suffix("Google\u02BCladım") == ("Google", "ladım")

    def test_unicode_left_quote(self) -> None:
        assert split_foreign_suffix("Google\u2018ladım") == ("Google", "ladım")

    def test_no_apostrophe(self) -> None:
        assert split_foreign_suffix("evlerinden") is None

    def test_empty(self) -> None:
        assert split_foreign_suffix("") is None

    def test_apostrophe_at_start(self) -> None:
        assert split_foreign_suffix("'ladım") is None

    def test_apostrophe_at_end(self) -> None:
        assert split_foreign_suffix("Google'") is None

    def test_iphone(self) -> None:
        assert split_foreign_suffix("iPhone'un") == ("iPhone", "un")

    def test_biden(self) -> None:
        assert split_foreign_suffix("Biden'a") == ("Biden", "a")

    def test_netflix(self) -> None:
        assert split_foreign_suffix("Netflix'ten") == ("Netflix", "ten")


# -----------------------------------------------------------------------
# classify_foreign_root
# -----------------------------------------------------------------------


class TestClassifyForeignRoot:
    def test_verb_stem_la(self) -> None:
        assert classify_foreign_root("Google", "ladım") == "verb_stem"

    def test_verb_stem_le(self) -> None:
        assert classify_foreign_root("tweet", "ledim") == "verb_stem"

    def test_proper_noun_uppercase(self) -> None:
        assert classify_foreign_root("Biden", "a") == "proper_noun"

    def test_proper_noun_genitive(self) -> None:
        assert classify_foreign_root("Twitter", "ın") == "proper_noun"

    def test_noun_lowercase(self) -> None:
        assert classify_foreign_root("tweet", "in") == "noun"

    def test_noun_default(self) -> None:
        assert classify_foreign_root("email", "ler") == "noun"


# -----------------------------------------------------------------------
# detect_code_switch
# -----------------------------------------------------------------------


class TestDetectCodeSwitch:
    def test_google_ladim(self) -> None:
        result = detect_code_switch("Google'ladım")
        assert result is not None
        assert result.is_code_switched is True
        assert result.foreign_root == "Google"
        assert result.suffix_part == "ladım"
        assert result.root_type == "verb_stem"

    def test_iphone_possessive(self) -> None:
        result = detect_code_switch("iPhone'un")
        assert result is not None
        assert result.foreign_root == "iPhone"
        assert result.suffix_part == "un"

    def test_biden_dative(self) -> None:
        result = detect_code_switch("Biden'a")
        assert result is not None
        assert result.foreign_root == "Biden"
        assert result.suffix_part == "a"
        assert result.root_type == "proper_noun"

    def test_netflix_ablative(self) -> None:
        result = detect_code_switch("Netflix'ten")
        assert result is not None
        assert result.foreign_root == "Netflix"

    def test_tweet_verb(self) -> None:
        result = detect_code_switch("tweet'ledim")
        assert result is not None
        assert result.root_type == "verb_stem"

    def test_abbreviation_defers(self) -> None:
        """NATO'nun should NOT be code-switch — it's an abbreviation."""
        assert detect_code_switch("NATO'nun") is None

    def test_abbreviation_ab_defers(self) -> None:
        assert detect_code_switch("AB'nin") is None

    def test_no_apostrophe(self) -> None:
        assert detect_code_switch("evlerinden") is None

    def test_empty(self) -> None:
        assert detect_code_switch("") is None

    def test_turkish_root_defers(self) -> None:
        """Root with Turkish-specific chars defers to standard backends."""
        assert detect_code_switch("Gümüş'ten") is None

    def test_original_preserved(self) -> None:
        result = detect_code_switch("Google'ladım")
        assert result is not None
        assert result.original == "Google'ladım"


# -----------------------------------------------------------------------
# analyze_foreign_suffixes
# -----------------------------------------------------------------------


class TestAnalyzeForeignSuffixes:
    def test_ladim(self) -> None:
        tags = analyze_foreign_suffixes("ladım", "e")
        assert tags == ["+VERB.LA", "+PAST", "+1SG"]

    def test_ledim(self) -> None:
        tags = analyze_foreign_suffixes("ledim", "e")
        assert tags == ["+VERB.LA", "+PAST", "+1SG"]

    def test_genitive_un(self) -> None:
        tags = analyze_foreign_suffixes("un", "e")
        assert "+GEN" in tags

    def test_genitive_in(self) -> None:
        tags = analyze_foreign_suffixes("ın", "a")
        assert "+GEN" in tags

    def test_dative_a(self) -> None:
        tags = analyze_foreign_suffixes("a", "e")
        assert "+DAT" in tags

    def test_dative_e(self) -> None:
        tags = analyze_foreign_suffixes("e", "a")
        assert "+DAT" in tags

    def test_ablative_ten(self) -> None:
        tags = analyze_foreign_suffixes("ten", "e")
        assert "+ABL" in tags

    def test_ablative_dan(self) -> None:
        tags = analyze_foreign_suffixes("dan", "a")
        assert "+ABL" in tags

    def test_plural_ler(self) -> None:
        tags = analyze_foreign_suffixes("ler", "e")
        assert "+PLU" in tags

    def test_plural_lar(self) -> None:
        tags = analyze_foreign_suffixes("lar", "a")
        assert "+PLU" in tags

    def test_empty_suffix(self) -> None:
        assert analyze_foreign_suffixes("", "e") == []

    def test_none_vowel(self) -> None:
        tags = analyze_foreign_suffixes("un", None)
        assert "+GEN" in tags

    def test_relaxed_harmony_default(self) -> None:
        """Back-vowel suffix on front-vowel root should still parse."""
        tags = analyze_foreign_suffixes("ladım", "e", relaxed_harmony=True)
        assert "+VERB.LA" in tags

    def test_locative_da(self) -> None:
        tags = analyze_foreign_suffixes("da", "a")
        assert "+LOC" in tags

    def test_verb_prog(self) -> None:
        tags = analyze_foreign_suffixes("lıyor", "a")
        assert "+VERB.LA" in tags or "+PROG" in tags


# -----------------------------------------------------------------------
# Analyzer integration
# -----------------------------------------------------------------------


class TestAnalyzerCodeSwitchIntegration:
    """Integration tests using a mock backend."""

    @staticmethod
    def _make_analyzer() -> MorphoAnalyzer:
        """Create an analyzer with no real backends."""
        analyzer = MorphoAnalyzer.__new__(MorphoAnalyzer)
        from kokturk.core.cache import AnalysisCache

        analyzer._backends = []
        analyzer._cache = AnalysisCache(capacity=100)
        return analyzer

    def test_disabled_by_default(self) -> None:
        analyzer = self._make_analyzer()
        result = analyzer.analyze("Google'ladım")
        # Without the flag, no code-switch detection — empty analyses
        assert isinstance(result, TokenAnalyses)
        # With no backends, should be empty
        assert result.parse_count == 0

    def test_enabled_detects_foreign(self) -> None:
        analyzer = self._make_analyzer()
        result = analyzer.analyze("Google'ladım", handle_code_switch=True)
        assert result.parse_count == 1
        analysis = result.analyses[0]
        assert analysis.root == "Google"
        assert analysis.source == "code_switch"
        assert "+FOREIGN" in analysis.tags
        assert "+VERB.LA" in analysis.tags

    def test_cache_isolation(self) -> None:
        analyzer = self._make_analyzer()
        r1 = analyzer.analyze("Google'ladım", handle_code_switch=False)
        r2 = analyzer.analyze("Google'ladım", handle_code_switch=True)
        # Different cache entries → different results
        assert r1.parse_count != r2.parse_count

    def test_score_is_0_9(self) -> None:
        analyzer = self._make_analyzer()
        result = analyzer.analyze("Google'ladım", handle_code_switch=True)
        assert result.analyses[0].score == pytest.approx(0.9)

    def test_non_code_switch_falls_through(self) -> None:
        """A normal Turkish word with code_switch enabled still goes to backends."""
        analyzer = self._make_analyzer()
        result = analyzer.analyze("evlerinden", handle_code_switch=True)
        # No backends and not code-switched → empty
        assert result.parse_count == 0

    def test_abbreviation_not_captured(self) -> None:
        """NATO'nun should not trigger code-switch even with flag on."""
        analyzer = self._make_analyzer()
        result = analyzer.analyze("NATO'nun", handle_code_switch=True)
        # detect_code_switch returns None for abbreviations → falls to backends
        assert result.parse_count == 0
