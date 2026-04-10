"""Tests for two-tier LRU analysis cache."""

from __future__ import annotations

import pytest

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


class TestAnalysisCache:
    def test_put_and_get(self) -> None:
        cache = AnalysisCache(capacity=10)
        token = _make_token("ev")
        cache.put("ev", token)
        assert cache.get("ev") == token

    def test_miss(self) -> None:
        cache = AnalysisCache(capacity=10)
        assert cache.get("nonexistent") is None

    def test_eviction(self) -> None:
        cache = AnalysisCache(capacity=2)
        cache.put("a", _make_token("a"))
        cache.put("b", _make_token("b"))
        cache.put("c", _make_token("c"))  # should evict "a"
        assert cache.get("a") is None
        assert cache.get("b") is not None
        assert cache.get("c") is not None

    def test_lru_order(self) -> None:
        cache = AnalysisCache(capacity=2)
        cache.put("a", _make_token("a"))
        cache.put("b", _make_token("b"))
        cache.get("a")  # access "a" to make it recently used
        cache.put("c", _make_token("c"))  # should evict "b" (least recently used)
        assert cache.get("a") is not None
        assert cache.get("b") is None

    def test_hit_rate(self) -> None:
        cache = AnalysisCache(capacity=10)
        cache.put("a", _make_token("a"))
        cache.get("a")  # hit
        cache.get("b")  # miss
        assert cache.hit_rate == 0.5

    def test_clear(self) -> None:
        cache = AnalysisCache(capacity=10)
        cache.put("a", _make_token("a"))
        cache.clear()
        assert len(cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0


class TestAnalysisCacheStats:
    def test_stats_keys(self) -> None:
        cache = AnalysisCache(capacity=10)
        cache.put("a", _make_token("a"))
        cache.get("a")
        cache.get("b")
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(0.5)
        assert stats["memory_entries"] == 1
        assert stats["disk_entries"] == 0

    def test_stats_empty(self) -> None:
        cache = AnalysisCache(capacity=10)
        stats = cache.stats
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["memory_entries"] == 0


class TestDiskCache:
    def test_disk_roundtrip(self, tmp_path) -> None:
        """Write via one cache instance, read back via another."""
        disk_path = str(tmp_path / "test_cache")
        token = _make_token("ev")

        # Write
        cache1 = AnalysisCache(capacity=10, disk_path=disk_path)
        cache1.put("ev", token)

        # New instance, empty memory, same disk
        cache2 = AnalysisCache(capacity=10, disk_path=disk_path)
        assert len(cache2) == 0  # memory is empty
        result = cache2.get("ev")  # should hit disk
        assert result is not None
        assert result.surface == "ev"
        assert cache2.hits == 1

    def test_disk_miss(self, tmp_path) -> None:
        disk_path = str(tmp_path / "test_cache")
        cache = AnalysisCache(capacity=10, disk_path=disk_path)
        assert cache.get("nonexistent") is None
        assert cache.misses == 1

    def test_disk_promotes_to_memory(self, tmp_path) -> None:
        disk_path = str(tmp_path / "test_cache")
        token = _make_token("ev")

        cache1 = AnalysisCache(capacity=10, disk_path=disk_path)
        cache1.put("ev", token)

        cache2 = AnalysisCache(capacity=10, disk_path=disk_path)
        cache2.get("ev")  # disk hit → promotes to memory
        assert len(cache2) == 1  # now in memory

    def test_disk_stats(self, tmp_path) -> None:
        disk_path = str(tmp_path / "test_cache")
        cache = AnalysisCache(capacity=10, disk_path=disk_path)
        cache.put("a", _make_token("a"))
        cache.put("b", _make_token("b"))
        assert cache.stats["disk_entries"] == 2
        assert cache.stats["memory_entries"] == 2

    def test_warm_from_disk(self, tmp_path) -> None:
        disk_path = str(tmp_path / "test_cache")
        cache1 = AnalysisCache(capacity=10, disk_path=disk_path)
        cache1.put("a", _make_token("a"))
        cache1.put("b", _make_token("b"))
        cache1.put("c", _make_token("c"))

        cache2 = AnalysisCache(capacity=10, disk_path=disk_path)
        assert len(cache2) == 0
        loaded = cache2.warm_from_disk(top_n=2)
        assert loaded == 2
        assert len(cache2) == 2

    def test_warm_from_disk_no_disk(self) -> None:
        cache = AnalysisCache(capacity=10)
        assert cache.warm_from_disk() == 0

    def test_clear_both_tiers(self, tmp_path) -> None:
        disk_path = str(tmp_path / "test_cache")
        cache = AnalysisCache(capacity=10, disk_path=disk_path)
        cache.put("a", _make_token("a"))
        cache.clear()
        assert len(cache) == 0
        assert cache.stats["disk_entries"] == 0


class TestBackwardCompat:
    """Ensure AnalysisCache() without disk_path works exactly as before."""

    def test_no_disk_path(self) -> None:
        cache = AnalysisCache(capacity=10)
        assert cache._disk is None
        token = _make_token("ev")
        cache.put("ev", token)
        assert cache.get("ev") == token
        assert cache.stats["disk_entries"] == 0
