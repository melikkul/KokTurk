"""Tests for MorphoAnalyzer cache convenience methods."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from kokturk.core.analyzer import MorphoAnalyzer
from kokturk.core.cache import AnalysisCache
from kokturk.core.datatypes import MorphologicalAnalysis, TokenAnalyses


def _make_token(word: str) -> TokenAnalyses:
    return TokenAnalyses(
        surface=word,
        analyses=(
            MorphologicalAnalysis(
                surface=word, root=word, tags=(), morphemes=(),
                source="test", score=1.0,
            ),
        ),
    )


class TestEnableCache:
    @patch("kokturk.core.analyzer._BACKEND_REGISTRY", {"zeyrek": MagicMock})
    def test_enable_cache_replaces(self) -> None:
        analyzer = MorphoAnalyzer(backends=["zeyrek"])
        old_cache = analyzer.cache
        analyzer.enable_cache(memory_size=200)
        new_cache = analyzer.cache
        assert new_cache is not old_cache
        assert new_cache.capacity == 200

    @patch("kokturk.core.analyzer._BACKEND_REGISTRY", {"zeyrek": MagicMock})
    def test_enable_cache_disk(self, tmp_path) -> None:
        analyzer = MorphoAnalyzer(backends=["zeyrek"])
        disk_path = str(tmp_path / "cache")
        analyzer.enable_cache(memory_size=100, disk_path=disk_path)
        assert analyzer.cache._disk is not None


class TestCacheStats:
    @patch("kokturk.core.analyzer._BACKEND_REGISTRY", {"zeyrek": MagicMock})
    def test_cache_stats_property(self) -> None:
        analyzer = MorphoAnalyzer(backends=["zeyrek"])
        stats = analyzer.cache_stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert "memory_entries" in stats
        assert "disk_entries" in stats

    @patch("kokturk.core.analyzer._BACKEND_REGISTRY", {"zeyrek": MagicMock})
    def test_cache_stats_after_usage(self) -> None:
        analyzer = MorphoAnalyzer(backends=["zeyrek"])
        # Manually put/get to test stats
        analyzer.cache.put("ev", _make_token("ev"))
        analyzer.cache.get("ev")
        analyzer.cache.get("missing")
        stats = analyzer.cache_stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["memory_entries"] == 1
