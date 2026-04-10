"""Focused tests for inverse-sigmoid scheduled sampling ratio."""

from __future__ import annotations

import pytest

from train.curriculum import scheduled_sampling_ratio


class TestScheduledSampling:
    def test_epoch_0_value(self) -> None:
        """At epoch 0: k/(k + exp(0/k)) = k/(k+1) ≈ 0.833 for k=5."""
        ratio = scheduled_sampling_ratio(0, k=5)
        expected = 5.0 / (5.0 + 1.0)
        assert ratio == pytest.approx(expected, abs=1e-6)

    def test_epoch_5_less_than_epoch_0(self) -> None:
        r0 = scheduled_sampling_ratio(0, k=5)
        r5 = scheduled_sampling_ratio(5, k=5)
        assert r5 < r0

    def test_epoch_20_near_zero(self) -> None:
        ratio = scheduled_sampling_ratio(20, k=5)
        assert ratio < 0.1

    def test_always_positive(self) -> None:
        for epoch in range(50):
            assert scheduled_sampling_ratio(epoch, k=5) > 0.0

    def test_monotonically_decreasing(self) -> None:
        ratios = [scheduled_sampling_ratio(e, k=5) for e in range(31)]
        for i in range(len(ratios) - 1):
            assert ratios[i] >= ratios[i + 1], (
                f"Not decreasing at epoch {i}: {ratios[i]} < {ratios[i+1]}"
            )
