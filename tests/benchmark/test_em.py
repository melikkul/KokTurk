"""Unit tests for src/aksu/benchmark/em.py."""
from __future__ import annotations

from aksu.benchmark.em import em_argmax, em_string, pred_index_to_strings


class TestEmArgmax:
    def test_perfect_match(self) -> None:
        assert em_argmax([0, 1, 2], [0, 1, 2]) == 1.0

    def test_no_match(self) -> None:
        assert em_argmax([1, 2, 0], [0, 1, 2]) == 0.0

    def test_partial_match(self) -> None:
        assert em_argmax([0, 2, 2], [0, 1, 2]) == pytest.approx(2 / 3)

    def test_empty_returns_zero(self) -> None:
        assert em_argmax([], []) == 0.0

    def test_single_correct(self) -> None:
        assert em_argmax([0], [0]) == 1.0

    def test_single_wrong(self) -> None:
        assert em_argmax([1], [0]) == 0.0


class TestEmString:
    def test_perfect_match(self) -> None:
        preds = ["ev +PLU +ABL", "git +PAST +1SG"]
        gold = ["ev +PLU +ABL", "git +PAST +1SG"]
        assert em_string(preds, gold) == 1.0

    def test_no_match(self) -> None:
        assert em_string(["ev +PLU"], ["ev +ABL"]) == 0.0

    def test_partial(self) -> None:
        preds = ["ev +PLU +ABL", "git +PAST +1SG", "kitap +PLU"]
        gold  = ["ev +PLU +ABL", "git +PAST +2SG", "kitap +PLU"]
        assert em_string(preds, gold) == pytest.approx(2 / 3)

    def test_empty_returns_zero(self) -> None:
        assert em_string([], []) == 0.0

    def test_case_sensitive(self) -> None:
        assert em_string(["ev +PLU"], ["EV +PLU"]) == 0.0


class TestPredIndexToStrings:
    def test_basic(self) -> None:
        candidates = [["ev +NOM", "ev +GEN"], ["git +PAST", "git +PROG"]]
        pred_indices = [0, 1]
        result = pred_index_to_strings(pred_indices, candidates)
        assert result == ["ev +NOM", "git +PROG"]

    def test_out_of_range_returns_empty(self) -> None:
        candidates = [["ev +NOM"]]
        result = pred_index_to_strings([5], candidates)
        assert result == [""]

    def test_negative_index_returns_empty(self) -> None:
        candidates = [["ev +NOM", "ev +GEN"]]
        result = pred_index_to_strings([-1], candidates)
        assert result == [""]

    def test_empty_inputs(self) -> None:
        assert pred_index_to_strings([], []) == []

    def test_index_zero(self) -> None:
        candidates = [["a", "b", "c"]]
        assert pred_index_to_strings([0], candidates) == ["a"]

    def test_last_valid_index(self) -> None:
        candidates = [["a", "b", "c"]]
        assert pred_index_to_strings([2], candidates) == ["c"]

    def test_one_past_end_returns_empty(self) -> None:
        candidates = [["a", "b", "c"]]
        assert pred_index_to_strings([3], candidates) == [""]


import pytest  # noqa: E402 — imported here so test class bodies above are readable
