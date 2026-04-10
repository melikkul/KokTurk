"""Tests for curriculum training schedule."""

from __future__ import annotations

import pytest
import torch

from train.curriculum import (
    compute_tier_weights,
    get_allowed_tiers,
    get_curriculum_phase,
    get_learning_rate,
    scheduled_sampling_ratio,
)


class TestGetCurriculumPhase:
    def test_epoch_1_gold_only(self) -> None:
        assert get_curriculum_phase(1) == "gold_only"

    def test_epoch_3_gold_only(self) -> None:
        assert get_curriculum_phase(3) == "gold_only"

    def test_epoch_4_gold_and_silver(self) -> None:
        assert get_curriculum_phase(4) == "gold_and_silver_auto"

    def test_epoch_12_gold_and_silver(self) -> None:
        assert get_curriculum_phase(12) == "gold_and_silver_auto"

    def test_epoch_13_all_tiers(self) -> None:
        assert get_curriculum_phase(13) == "all_tiers"

    def test_epoch_20_all_tiers(self) -> None:
        assert get_curriculum_phase(20) == "all_tiers"

    def test_epoch_21_gold_only(self) -> None:
        assert get_curriculum_phase(21) == "gold_only"

    def test_epoch_25_gold_only(self) -> None:
        assert get_curriculum_phase(25) == "gold_only"

    def test_epoch_26_all_tiers(self) -> None:
        assert get_curriculum_phase(26) == "all_tiers"

    def test_epoch_30_all_tiers(self) -> None:
        assert get_curriculum_phase(30) == "all_tiers"


class TestGetAllowedTiers:
    def test_gold_only(self) -> None:
        assert get_allowed_tiers("gold_only") == {0}

    def test_gold_and_silver_auto(self) -> None:
        assert get_allowed_tiers("gold_and_silver_auto") == {0, 1}

    def test_all_tiers(self) -> None:
        assert get_allowed_tiers("all_tiers") == {0, 1, 2}


class TestScheduledSamplingRatio:
    def test_epoch_0_near_one(self) -> None:
        ratio = scheduled_sampling_ratio(0, k=5.0)
        expected = 5.0 / (5.0 + 1.0)  # k / (k + exp(0/k)) = k/(k+1)
        assert ratio == pytest.approx(expected, abs=1e-6)

    def test_monotonically_decreasing(self) -> None:
        ratios = [scheduled_sampling_ratio(e, k=5.0) for e in range(31)]
        for i in range(len(ratios) - 1):
            assert ratios[i] >= ratios[i + 1]

    def test_approaches_zero(self) -> None:
        assert scheduled_sampling_ratio(100, k=5.0) < 0.01


class TestGetLearningRate:
    def test_epoch_1_warmup_start(self) -> None:
        assert get_learning_rate(1) == pytest.approx(1e-4, rel=1e-3)

    def test_epoch_3_warmup_end(self) -> None:
        assert get_learning_rate(3) == pytest.approx(5e-4, rel=1e-3)

    def test_epoch_25_constant(self) -> None:
        assert get_learning_rate(25) == pytest.approx(5e-5, rel=1e-3)

    def test_epoch_30_constant(self) -> None:
        assert get_learning_rate(30) == pytest.approx(1e-5, rel=1e-3)


class TestComputeTierWeights:
    def test_gold_weight(self) -> None:
        tiers = torch.tensor([0])
        weights = compute_tier_weights(tiers)
        assert weights[0].item() == pytest.approx(5.0)

    def test_silver_auto_weight(self) -> None:
        tiers = torch.tensor([1])
        weights = compute_tier_weights(tiers)
        assert weights[0].item() == pytest.approx(1.0)

    def test_silver_agreed_weight(self) -> None:
        tiers = torch.tensor([2])
        weights = compute_tier_weights(tiers)
        assert weights[0].item() == pytest.approx(0.7)

    def test_batch(self) -> None:
        tiers = torch.tensor([0, 1, 2, 0])
        weights = compute_tier_weights(tiers)
        assert weights.tolist() == pytest.approx([5.0, 1.0, 0.7, 5.0])
