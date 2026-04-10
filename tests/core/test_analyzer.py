"""Tests for MorphoAnalyzer and backends."""

from __future__ import annotations

import pytest

from kokturk.core.analyzer import MorphoAnalyzer
from kokturk.core.datatypes import TokenAnalyses


class TestMorphoAnalyzerInit:
    def test_default_backend(self) -> None:
        analyzer = MorphoAnalyzer()
        assert analyzer is not None

    def test_explicit_zeyrek(self) -> None:
        analyzer = MorphoAnalyzer(backends=["zeyrek"])
        assert analyzer is not None

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            MorphoAnalyzer(backends=["nonexistent"])


class TestMorphoAnalyzerAnalyze:
    def test_returns_token_analyses(self, analyzer: MorphoAnalyzer) -> None:
        result = analyzer.analyze("ev")
        assert isinstance(result, TokenAnalyses)
        assert result.surface == "ev"

    def test_parses_simple_word(self, analyzer: MorphoAnalyzer) -> None:
        result = analyzer.analyze("ev")
        assert result.parse_count >= 1

    def test_parses_inflected_word(self, analyzer: MorphoAnalyzer) -> None:
        result = analyzer.analyze("evlerinden")
        assert result.parse_count >= 1
        # At least one parse should have "ev" as root
        roots = {a.root for a in result.analyses}
        assert "ev" in roots, f"Expected 'ev' root, got: {roots}"

    def test_deduplicates_across_backends(self, analyzer: MorphoAnalyzer) -> None:
        """Same parse from different backends should be deduplicated."""
        result = analyzer.analyze("ev")
        identities = [a.parse_identity() for a in result.analyses]
        assert len(identities) == len(set(identities)), "Duplicate parses found"


class TestMorphoAnalyzerCache:
    def test_cache_hit(self, analyzer: MorphoAnalyzer) -> None:
        analyzer.analyze("ev")
        analyzer.analyze("ev")
        assert analyzer.cache.hits >= 1

    def test_cache_miss(self, analyzer: MorphoAnalyzer) -> None:
        analyzer.analyze("ev")
        assert analyzer.cache.misses >= 1  # First call is always a miss

    def test_cache_returns_same_result(self, analyzer: MorphoAnalyzer) -> None:
        r1 = analyzer.analyze("ev")
        r2 = analyzer.analyze("ev")
        assert r1 == r2


class TestMorphoAnalyzerPipe:
    def test_pipe_basic(self, analyzer: MorphoAnalyzer) -> None:
        words = ["ev", "araba", "kitap"]
        results = list(analyzer.pipe(words))
        assert len(results) == 3
        assert all(isinstance(r, TokenAnalyses) for r in results)


class TestMorphoAnalyzerContextManager:
    def test_context_manager(self) -> None:
        with MorphoAnalyzer() as analyzer:
            result = analyzer.analyze("ev")
            assert result.parse_count >= 1
