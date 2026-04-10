"""Tests for core data types."""

from __future__ import annotations

import pytest

from kokturk.core.datatypes import Morpheme, MorphologicalAnalysis, TokenAnalyses


class TestMorpheme:
    def test_creation(self) -> None:
        m = Morpheme(surface="ler", canonical="+PLU", category="inflectional")
        assert m.surface == "ler"
        assert m.canonical == "+PLU"
        assert m.category == "inflectional"

    def test_frozen(self) -> None:
        m = Morpheme(surface="ler", canonical="+PLU", category="inflectional")
        with pytest.raises(AttributeError):
            m.surface = "den"  # type: ignore[misc]

    def test_equality(self) -> None:
        m1 = Morpheme(surface="ler", canonical="+PLU", category="inflectional")
        m2 = Morpheme(surface="ler", canonical="+PLU", category="inflectional")
        assert m1 == m2

    def test_hashable(self) -> None:
        m = Morpheme(surface="ler", canonical="+PLU", category="inflectional")
        assert hash(m) == hash(m)
        s = {m}
        assert len(s) == 1


class TestMorphologicalAnalysis:
    def test_creation(self, sample_analysis: MorphologicalAnalysis) -> None:
        assert sample_analysis.surface == "evlerinden"
        assert sample_analysis.root == "ev"
        assert sample_analysis.tags == ("+PLU", "+POSS.3SG", "+ABL")
        assert sample_analysis.source == "zeyrek"

    def test_frozen(self, sample_analysis: MorphologicalAnalysis) -> None:
        with pytest.raises(AttributeError):
            sample_analysis.root = "araba"  # type: ignore[misc]

    def test_lemma(self, sample_analysis: MorphologicalAnalysis) -> None:
        assert sample_analysis.lemma == "ev"

    def test_to_str(self, sample_analysis: MorphologicalAnalysis) -> None:
        assert sample_analysis.to_str() == "ev +PLU +POSS.3SG +ABL"

    def test_to_str_root_only(self) -> None:
        a = MorphologicalAnalysis(
            surface="ev", root="ev", tags=(), morphemes=(),
            source="zeyrek", score=1.0,
        )
        assert a.to_str() == "ev"

    def test_tag_order_preserved(self) -> None:
        """Tag order is semantically meaningful — must not be sorted or shuffled."""
        a = MorphologicalAnalysis(
            surface="test", root="test",
            tags=("+PAST", "+PLU", "+COP"),
            morphemes=(), source="zeyrek", score=1.0,
        )
        assert a.tags == ("+PAST", "+PLU", "+COP")
        assert a.tags != ("+PLU", "+PAST", "+COP")

    def test_to_conllu(self, sample_analysis: MorphologicalAnalysis) -> None:
        conllu = sample_analysis.to_conllu()
        assert "Case=Abl" in conllu
        assert "Number=Plur" in conllu
        assert "|" in conllu

    def test_parse_identity(self) -> None:
        a1 = MorphologicalAnalysis(
            surface="ev", root="ev", tags=("+PLU",), morphemes=(),
            source="zeyrek", score=0.5,
        )
        a2 = MorphologicalAnalysis(
            surface="ev", root="ev", tags=("+PLU",), morphemes=(),
            source="trmorph", score=0.8,
        )
        assert a1.parse_identity() == a2.parse_identity()

    def test_different_parses_different_identity(self) -> None:
        a1 = MorphologicalAnalysis(
            surface="ev", root="ev", tags=("+PLU",), morphemes=(),
            source="zeyrek", score=0.5,
        )
        a2 = MorphologicalAnalysis(
            surface="ev", root="ev", tags=("+ACC",), morphemes=(),
            source="zeyrek", score=0.5,
        )
        assert a1.parse_identity() != a2.parse_identity()


class TestTokenAnalyses:
    def test_ambiguous(self, sample_token_analyses: TokenAnalyses) -> None:
        assert sample_token_analyses.is_ambiguous is True
        assert sample_token_analyses.parse_count == 2

    def test_unambiguous(self, sample_analysis: MorphologicalAnalysis) -> None:
        ta = TokenAnalyses(surface="ev", analyses=(sample_analysis,))
        assert ta.is_ambiguous is False
        assert ta.parse_count == 1

    def test_empty(self) -> None:
        ta = TokenAnalyses(surface="xyz", analyses=())
        assert ta.is_ambiguous is False
        assert ta.parse_count == 0
        assert ta.best is None

    def test_best(self, sample_token_analyses: TokenAnalyses) -> None:
        best = sample_token_analyses.best
        assert best is not None
        assert best.score == 0.5  # The one with higher score
