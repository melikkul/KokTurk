"""Tests for corpus statistics and cache sizing."""

from __future__ import annotations

import json

import pytest

from benchmark.corpus_stats import CorpusStats, compute_corpus_stats, recommend_cache_size


def _write_corpus(tmp_path, words: list[str]) -> str:
    """Write a synthetic JSONL corpus and return the path."""
    corpus_path = tmp_path / "corpus.jsonl"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for w in words:
            f.write(json.dumps({"surface": w, "label": f"{w} +NOM"}) + "\n")
    return str(corpus_path)


class TestComputeCorpusStats:
    def test_basic_stats(self, tmp_path) -> None:
        words = ["ev"] * 5 + ["araba"] * 3 + ["kitap"] * 2 + ["kalem"]
        path = _write_corpus(tmp_path, words)
        stats = compute_corpus_stats(path)

        assert stats.total_tokens == 11
        assert stats.unique_types == 4
        assert stats.hapax_count == 1  # "kalem" appears once
        assert stats.hapax_ratio == pytest.approx(0.25)
        assert stats.type_token_ratio == pytest.approx(4 / 11)

    def test_coverage_at_k_monotonic(self, tmp_path) -> None:
        # Zipfian-ish distribution
        words = []
        for i in range(1, 51):
            words.extend([f"word_{i}"] * (100 // i))
        path = _write_corpus(tmp_path, words)
        stats = compute_corpus_stats(path, coverage_ks=[1, 5, 10, 50])

        coverages = [stats.coverage_at_k[k] for k in sorted(stats.coverage_at_k)]
        for i in range(len(coverages) - 1):
            assert coverages[i] <= coverages[i + 1]

    def test_zipf_alpha_positive(self, tmp_path) -> None:
        words = []
        for i in range(1, 101):
            words.extend([f"word_{i}"] * max(1, 1000 // i))
        path = _write_corpus(tmp_path, words)
        stats = compute_corpus_stats(path)
        assert stats.zipf_alpha > 0

    def test_empty_corpus(self, tmp_path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        stats = compute_corpus_stats(str(path))
        assert stats.total_tokens == 0
        assert stats.unique_types == 0
        assert stats.zipf_alpha == 0.0

    def test_frozen_dataclass(self, tmp_path) -> None:
        words = ["ev", "araba"]
        path = _write_corpus(tmp_path, words)
        stats = compute_corpus_stats(path)
        with pytest.raises(AttributeError):
            stats.total_tokens = 999  # type: ignore[misc]


class TestRecommendCacheSize:
    def test_returns_reasonable_int(self, tmp_path) -> None:
        words = []
        for i in range(1, 201):
            words.extend([f"word_{i}"] * max(1, 500 // i))
        path = _write_corpus(tmp_path, words)
        stats = compute_corpus_stats(path, coverage_ks=[10, 50, 100, 200])
        rec = recommend_cache_size(stats, target_hit_rate=0.80)
        assert isinstance(rec, int)
        assert rec > 0

    def test_high_target_returns_large(self, tmp_path) -> None:
        words = [f"word_{i}" for i in range(1000)]  # all hapax
        path = _write_corpus(tmp_path, words)
        stats = compute_corpus_stats(path, coverage_ks=[100, 500, 1000])
        rec = recommend_cache_size(stats, target_hit_rate=0.99)
        assert rec >= 100

    def test_empty_coverage(self) -> None:
        stats = CorpusStats(
            total_tokens=0, unique_types=0, type_token_ratio=0.0,
            hapax_count=0, hapax_ratio=0.0, zipf_alpha=0.0,
            coverage_at_k={},
        )
        rec = recommend_cache_size(stats)
        assert rec == 0

    def test_already_met(self, tmp_path) -> None:
        # One word repeated 100 times → 100% coverage at k=1
        words = ["ev"] * 100
        path = _write_corpus(tmp_path, words)
        stats = compute_corpus_stats(path, coverage_ks=[1, 10, 100])
        rec = recommend_cache_size(stats, target_hit_rate=0.90)
        assert rec == 1
