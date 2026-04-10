"""Corpus statistics for cache sizing and linguistic analysis.

Computes Heaps' Law and Zipf's Law parameters from a Turkish corpus.
Used to empirically determine optimal cache sizes.

Usage::

    python -m benchmark.corpus_stats --corpus data/splits/train.jsonl

"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CorpusStats:
    """Aggregate statistics for a token corpus."""

    total_tokens: int
    unique_types: int
    type_token_ratio: float
    hapax_count: int
    hapax_ratio: float
    zipf_alpha: float
    coverage_at_k: dict[int, float]


# ------------------------------------------------------------------
# Core computation
# ------------------------------------------------------------------

def compute_corpus_stats(
    corpus_path: str | Path,
    coverage_ks: list[int] | None = None,
) -> CorpusStats:
    """Analyse a corpus and return aggregate statistics.

    Args:
        corpus_path: Path to a JSONL file where each line has a
            ``"surface"`` field.
        coverage_ks: Cache sizes at which to measure token coverage.
            Defaults to ``[100, 1000, 10_000, 50_000, 100_000, 200_000]``.

    Returns:
        A :class:`CorpusStats` with Zipf exponent and coverage data.
    """
    if coverage_ks is None:
        coverage_ks = [100, 1_000, 10_000, 50_000, 100_000, 200_000]

    freq: Counter[str] = Counter()
    corpus_path = Path(corpus_path)

    with open(corpus_path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            surface = rec.get("surface", "")
            if surface:
                freq[surface] += 1

    total_tokens = sum(freq.values())
    unique_types = len(freq)

    if total_tokens == 0:
        return CorpusStats(
            total_tokens=0,
            unique_types=0,
            type_token_ratio=0.0,
            hapax_count=0,
            hapax_ratio=0.0,
            zipf_alpha=0.0,
            coverage_at_k={k: 0.0 for k in coverage_ks},
        )

    hapax_count = sum(1 for c in freq.values() if c == 1)

    # Coverage: top-k types cover what fraction of total tokens?
    sorted_counts = sorted(freq.values(), reverse=True)
    cumsum = np.cumsum(sorted_counts)
    coverage_at_k: dict[int, float] = {}
    for k in coverage_ks:
        idx = min(k, len(sorted_counts)) - 1
        coverage_at_k[k] = float(cumsum[idx] / total_tokens) if idx >= 0 else 0.0

    # Zipf exponent: fit log(freq) = -alpha * log(rank) + C
    zipf_alpha = _fit_zipf(sorted_counts)

    return CorpusStats(
        total_tokens=total_tokens,
        unique_types=unique_types,
        type_token_ratio=unique_types / total_tokens,
        hapax_count=hapax_count,
        hapax_ratio=hapax_count / unique_types if unique_types > 0 else 0.0,
        zipf_alpha=zipf_alpha,
        coverage_at_k=coverage_at_k,
    )


def _fit_zipf(sorted_counts: list[int]) -> float:
    """Fit Zipf's Law exponent via least-squares on log-log scale."""
    n = len(sorted_counts)
    if n < 2:
        return 0.0

    log_ranks = np.log(np.arange(1, n + 1, dtype=np.float64))
    log_freqs = np.log(np.array(sorted_counts, dtype=np.float64))

    # Simple linear regression: log_freq = -alpha * log_rank + C
    coeffs = np.polyfit(log_ranks, log_freqs, 1)
    return float(-coeffs[0])


# ------------------------------------------------------------------
# Cache sizing recommendation
# ------------------------------------------------------------------

def recommend_cache_size(
    stats: CorpusStats,
    target_hit_rate: float = 0.90,
) -> int:
    """Recommend minimum cache size to achieve a target hit rate.

    Uses linear interpolation over ``coverage_at_k``.

    Args:
        stats: Corpus statistics from :func:`compute_corpus_stats`.
        target_hit_rate: Desired fraction of token occurrences served
            from cache (default 0.90).

    Returns:
        Recommended cache capacity (number of unique types).
    """
    if not stats.coverage_at_k:
        return stats.unique_types

    ks = sorted(stats.coverage_at_k.keys())
    coverages = [stats.coverage_at_k[k] for k in ks]

    # If smallest k already exceeds target, return it
    if coverages[0] >= target_hit_rate:
        return ks[0]

    # Linear interpolation between successive k values
    for i in range(len(ks) - 1):
        if coverages[i] <= target_hit_rate <= coverages[i + 1]:
            frac = (target_hit_rate - coverages[i]) / (
                coverages[i + 1] - coverages[i]
            )
            return int(math.ceil(ks[i] + frac * (ks[i + 1] - ks[i])))

    # Target exceeds all measured coverages — return largest k or unique_types
    return min(ks[-1], stats.unique_types)


# ------------------------------------------------------------------
# CLI entry-point
# ------------------------------------------------------------------

def main() -> None:  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Corpus statistics for cache sizing")
    parser.add_argument(
        "--corpus", type=Path, required=True,
        help="Path to JSONL corpus with 'surface' fields",
    )
    parser.add_argument(
        "--target-hit-rate", type=float, default=0.90,
        help="Target cache hit rate (default 0.90)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    stats = compute_corpus_stats(args.corpus)

    print(f"Total tokens:     {stats.total_tokens:>12,}")
    print(f"Unique types:     {stats.unique_types:>12,}")
    print(f"Type/token ratio: {stats.type_token_ratio:>12.4f}")
    print(f"Hapax legomena:   {stats.hapax_count:>12,} ({stats.hapax_ratio:.1%})")
    print(f"Zipf alpha:       {stats.zipf_alpha:>12.3f}")
    print()
    print("Coverage at cache size K:")
    for k, cov in sorted(stats.coverage_at_k.items()):
        print(f"  K={k:>8,}  →  {cov:.2%}")
    print()
    rec = recommend_cache_size(stats, target_hit_rate=args.target_hit_rate)
    print(f"Recommended cache size for {args.target_hit_rate:.0%} hit rate: {rec:,}")


if __name__ == "__main__":
    main()
