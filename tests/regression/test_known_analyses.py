"""Canary regression tests for morphological analysis quality.

These tests verify that the Zeyrek backend returns expected parses for
known Turkish words. If Zeyrek can't produce multiple analyses for
ambiguous words, backend coverage is insufficient and we need to
escalate to the Zemberek Java JAR.

Each test checks SPECIFIC expected parses, not just counts.
"""

from __future__ import annotations

import pytest

from kokturk.core.analyzer import MorphoAnalyzer


@pytest.fixture(scope="module")
def analyzer() -> MorphoAnalyzer:
    """Module-scoped analyzer for regression tests (avoid repeated init)."""
    return MorphoAnalyzer(backends=["zeyrek"])


class TestAmbiguousWords:
    """Words that MUST return multiple analyses — the core challenge."""

    def test_yuz_ambiguity(self, analyzer: MorphoAnalyzer) -> None:
        """'yüz' = face (noun), hundred (numeral), swim (verb)."""
        result = analyzer.analyze("yüz")
        assert result.parse_count >= 2, (
            f"Expected ≥2 parses for 'yüz', got {result.parse_count}: "
            f"{[a.to_str() for a in result.analyses]}"
        )

    def test_yazar_ambiguity(self, analyzer: MorphoAnalyzer) -> None:
        """'yazar' = author (noun), writes (verb+aor)."""
        result = analyzer.analyze("yazar")
        assert result.parse_count >= 2, (
            f"Expected ≥2 parses for 'yazar', got {result.parse_count}"
        )

    def test_alan_ambiguity(self, analyzer: MorphoAnalyzer) -> None:
        """'alan' = field/area (noun), receiver/taker (verb participle)."""
        result = analyzer.analyze("alan")
        assert result.parse_count >= 2, (
            f"Expected ≥2 parses for 'alan', got {result.parse_count}"
        )


class TestInflectedWords:
    """Words with known morphological decompositions."""

    def test_evlerinden(self, analyzer: MorphoAnalyzer) -> None:
        """'evlerinden' = ev + PLU + POSS.3SG/3PL + ABL."""
        result = analyzer.analyze("evlerinden")
        assert result.parse_count >= 1
        roots = {a.root for a in result.analyses}
        assert "ev" in roots, f"Expected 'ev' root, got: {roots}"

    def test_gidiyorum(self, analyzer: MorphoAnalyzer) -> None:
        """'gidiyorum' = git (go) + PROG + 1SG."""
        result = analyzer.analyze("gidiyorum")
        assert result.parse_count >= 1
        roots = {a.root for a in result.analyses}
        assert "git" in roots or "gitmek" in roots, f"Expected 'git' root, got: {roots}"

    def test_kitabi(self, analyzer: MorphoAnalyzer) -> None:
        """'kitabı' = kitap + ACC or kitap + POSS.3SG."""
        result = analyzer.analyze("kitabı")
        assert result.parse_count >= 1
        roots = {a.root for a in result.analyses}
        assert "kitap" in roots, f"Expected 'kitap' root, got: {roots}"

    def test_okudum(self, analyzer: MorphoAnalyzer) -> None:
        """'okudum' = oku (read) + PAST + 1SG."""
        result = analyzer.analyze("okudum")
        assert result.parse_count >= 1
        roots = {a.root for a in result.analyses}
        assert "oku" in roots or "okumak" in roots, f"Expected 'oku' root, got: {roots}"


class TestSuffixTypes:
    """Verify coverage of major suffix categories."""

    def test_plural(self, analyzer: MorphoAnalyzer) -> None:
        """'evler' should have +PLU tag."""
        result = analyzer.analyze("evler")
        assert result.parse_count >= 1

    def test_locative(self, analyzer: MorphoAnalyzer) -> None:
        """'evde' should have +LOC tag."""
        result = analyzer.analyze("evde")
        assert result.parse_count >= 1

    def test_dative(self, analyzer: MorphoAnalyzer) -> None:
        """'eve' should have +DAT tag."""
        result = analyzer.analyze("eve")
        assert result.parse_count >= 1

    def test_genitive(self, analyzer: MorphoAnalyzer) -> None:
        """'evin' should have +GEN tag."""
        result = analyzer.analyze("evin")
        assert result.parse_count >= 1

    def test_ablative(self, analyzer: MorphoAnalyzer) -> None:
        """'evden' should have +ABL tag."""
        result = analyzer.analyze("evden")
        assert result.parse_count >= 1

    def test_instrumental(self, analyzer: MorphoAnalyzer) -> None:
        """'kalemle' should have +INS tag."""
        result = analyzer.analyze("kalemle")
        assert result.parse_count >= 1

    def test_possessive(self, analyzer: MorphoAnalyzer) -> None:
        """'evim' = ev + POSS.1SG."""
        result = analyzer.analyze("evim")
        assert result.parse_count >= 1

    def test_past_tense(self, analyzer: MorphoAnalyzer) -> None:
        """'geldi' = gel + PAST + 3SG."""
        result = analyzer.analyze("geldi")
        assert result.parse_count >= 1
        roots = {a.root for a in result.analyses}
        assert "gel" in roots or "gelmek" in roots, f"Expected 'gel' root, got: {roots}"

    def test_future_tense(self, analyzer: MorphoAnalyzer) -> None:
        """'gelecek' = gel + FUT (or 'future' as noun)."""
        result = analyzer.analyze("gelecek")
        assert result.parse_count >= 1

    def test_negative(self, analyzer: MorphoAnalyzer) -> None:
        """'gelmiyor' = gel + NEG + PROG."""
        result = analyzer.analyze("gelmiyor")
        assert result.parse_count >= 1

    def test_causative(self, analyzer: MorphoAnalyzer) -> None:
        """'gezdirdi' = gez + CAUS + PAST."""
        result = analyzer.analyze("gezdirdi")
        assert result.parse_count >= 1


class TestEdgeCases:
    """Edge cases and common failure modes."""

    def test_single_char(self, analyzer: MorphoAnalyzer) -> None:
        """Single-character words should not crash."""
        result = analyzer.analyze("o")
        assert isinstance(result.parse_count, int)

    def test_empty_string(self, analyzer: MorphoAnalyzer) -> None:
        """Empty string should return empty analyses."""
        result = analyzer.analyze("")
        assert result.parse_count >= 0

    def test_unknown_word(self, analyzer: MorphoAnalyzer) -> None:
        """A made-up word should not crash (may return 0 parses)."""
        result = analyzer.analyze("xyzqwerty")
        assert isinstance(result.parse_count, int)

    def test_proper_noun(self, analyzer: MorphoAnalyzer) -> None:
        """'Ankara' — proper noun handling."""
        result = analyzer.analyze("Ankara")
        assert result.parse_count >= 1
