"""Tests for training efficiency metrics."""
from __future__ import annotations

from benchmark.training_efficiency import (
    area_under_learning_curve,
    compute_all_efficiency_metrics,
    format_efficiency_table,
    relative_efficiency,
    time_to_threshold,
)


class TestTimeToThreshold:
    def test_basic(self):
        history = [0.1, 0.3, 0.5, 0.7, 0.8, 0.82]
        # 90% of 0.82 = 0.738. epoch 5 (0.8) >= 0.738
        assert time_to_threshold(history, 0.9) == 5

    def test_never_reached(self):
        history = [0.1, 0.2, 0.3]
        assert time_to_threshold(history, 0.95) == 3

    def test_empty_history(self):
        assert time_to_threshold([], 0.9) == 0

    def test_first_epoch(self):
        history = [0.9, 0.92, 0.95]
        assert time_to_threshold(history, 0.9) == 1

    def test_custom_final_em(self):
        history = [0.1, 0.5, 0.8]
        assert time_to_threshold(history, 0.9, final_em=1.0) == 3


class TestAULC:
    def test_constant(self):
        history = [0.5, 0.5, 0.5, 0.5]
        assert abs(area_under_learning_curve(history) - 0.5) < 0.01

    def test_increasing(self):
        history = [0.0, 0.5, 1.0]
        aulc = area_under_learning_curve(history)
        assert abs(aulc - 0.5) < 0.01

    def test_empty(self):
        assert area_under_learning_curve([]) == 0.0


class TestRelativeEfficiency:
    def test_faster(self):
        assert relative_efficiency(30, 15) == 2.0

    def test_same(self):
        assert relative_efficiency(20, 20) == 1.0

    def test_slower(self):
        assert relative_efficiency(10, 20) == 0.5


class TestComputeAll:
    def test_basic(self):
        history = [0.1, 0.5, 0.7, 0.8, 0.82]
        m = compute_all_efficiency_metrics(history)
        assert "ttt_90" in m
        assert "ttt_95" in m
        assert "aulc" in m
        assert "final_em" in m
        assert m["final_em"] == 0.82

    def test_with_baseline(self):
        method = [0.5, 0.7, 0.8, 0.82]
        baseline = [0.1, 0.3, 0.5, 0.7, 0.8, 0.82]
        m = compute_all_efficiency_metrics(method, baseline)
        assert "relative_efficiency_90" in m


class TestFormatTable:
    def test_output(self):
        results = {
            "TAAC": {"final_em": 0.82, "ttt_90": 15, "ttt_95": 25,
                     "aulc": 0.65, "relative_efficiency_90": 2.0},
            "Fixed": {"final_em": 0.82, "ttt_90": 30, "ttt_95": 40,
                      "aulc": 0.55},
        }
        table = format_efficiency_table(results)
        assert "TAAC" in table
        assert "2.0x" in table
        assert "Fixed" in table
