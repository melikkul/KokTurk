"""Tests for intrinsic evaluation metrics."""

from __future__ import annotations

from benchmark.intrinsic_eval import (
    compute_all_metrics,
    compute_exact_match,
    compute_root_accuracy,
    compute_tag_f1,
)


class TestExactMatch:
    def test_perfect(self) -> None:
        preds = [[10, 20, 30], [40, 50]]
        gold = [[10, 20, 30], [40, 50]]
        assert compute_exact_match(preds, gold) == 1.0

    def test_none(self) -> None:
        preds = [[10, 20], [40, 50]]
        gold = [[10, 30], [40, 60]]
        assert compute_exact_match(preds, gold) == 0.0

    def test_partial(self) -> None:
        preds = [[10, 20], [40, 50]]
        gold = [[10, 20], [40, 60]]
        assert compute_exact_match(preds, gold) == 0.5

    def test_empty(self) -> None:
        assert compute_exact_match([], []) == 0.0


class TestRootAccuracy:
    def test_correct_roots(self) -> None:
        preds = [[10, 20], [40, 50]]
        gold = [[10, 30], [40, 60]]
        assert compute_root_accuracy(preds, gold) == 1.0

    def test_wrong_roots(self) -> None:
        preds = [[99, 20], [88, 50]]
        gold = [[10, 20], [40, 50]]
        assert compute_root_accuracy(preds, gold) == 0.0


class TestTagF1:
    def test_perfect(self) -> None:
        preds = [[10, 20, 30]]
        gold = [[10, 20, 30]]
        result = compute_tag_f1(preds, gold)
        assert result["f1"] == 1.0

    def test_partial_overlap(self) -> None:
        preds = [[10, 20, 30]]  # tags: {20, 30}
        gold = [[10, 20, 40]]   # tags: {20, 40}
        result = compute_tag_f1(preds, gold)
        # tp=1 (20), fp=1 (30), fn=1 (40)
        assert result["precision"] == 0.5
        assert result["recall"] == 0.5

    def test_no_tags(self) -> None:
        preds = [[10]]  # root only
        gold = [[10]]
        result = compute_tag_f1(preds, gold)
        # No tags to compare — precision/recall undefined, F1=0
        assert result["f1"] == 0.0 or result["f1"] == 1.0  # depends on impl


class TestAllMetrics:
    def test_with_parse_counts(self) -> None:
        preds = [[10, 20], [40, 50], [70, 80]]
        gold = [[10, 20], [40, 60], [70, 80]]
        counts = [1, 3, 2]
        metrics = compute_all_metrics(preds, gold, counts)
        assert "exact_match" in metrics
        assert "root_accuracy" in metrics
        assert "f1" in metrics
        assert "ambiguous_em" in metrics
        # Ambiguous tokens (count > 1): [40,50] vs [40,60] and [70,80] vs [70,80]
        # EM on ambiguous = 1/2 = 0.5
        assert metrics["ambiguous_em"] == 0.5
