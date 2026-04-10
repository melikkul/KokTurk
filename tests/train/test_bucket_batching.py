"""Tests for dynamic bucket batching."""

from __future__ import annotations

import torch
import pytest

from train.bucket_batching import (
    BucketBatchSampler,
    analyze_batching_efficiency,
    compute_pad_fraction,
    dynamic_pad_collate,
)


class TestBucketBatchSampler:
    def test_all_indices_covered(self) -> None:
        """Every dataset index must appear exactly once."""
        lengths = [3, 7, 2, 15, 5, 10, 1, 8, 4, 12]
        sampler = BucketBatchSampler(lengths, batch_size=3, shuffle=False)
        all_indices: list[int] = []
        for batch in sampler:
            all_indices.extend(batch)
        assert sorted(all_indices) == list(range(len(lengths)))

    def test_batch_size_respected(self) -> None:
        lengths = list(range(1, 21))  # 20 items
        sampler = BucketBatchSampler(lengths, batch_size=4, shuffle=False)
        for batch in sampler:
            assert len(batch) <= 4

    def test_drop_last(self) -> None:
        lengths = [5] * 7  # 7 items, batch_size=3 → 2 full + 1 partial
        sampler = BucketBatchSampler(
            lengths, batch_size=3, shuffle=False, drop_last=True,
        )
        for batch in sampler:
            assert len(batch) == 3

    def test_len(self) -> None:
        lengths = [5] * 10
        sampler = BucketBatchSampler(lengths, batch_size=3, shuffle=False)
        batches = list(sampler)
        assert len(sampler) == len(batches)

    def test_subset_indices(self) -> None:
        """Only specified indices should appear."""
        lengths = [5] * 10
        indices = [0, 2, 4, 6, 8]
        sampler = BucketBatchSampler(
            lengths, batch_size=2, indices=indices, shuffle=False,
        )
        all_indices: list[int] = []
        for batch in sampler:
            all_indices.extend(batch)
        assert sorted(all_indices) == indices

    def test_shuffle_changes_order(self) -> None:
        lengths = list(range(1, 101))
        sampler = BucketBatchSampler(
            lengths, batch_size=10, shuffle=True, seed=42,
        )
        sampler.set_epoch(0)
        order_0 = [idx for batch in sampler for idx in batch]
        sampler.set_epoch(1)
        order_1 = [idx for batch in sampler for idx in batch]
        # Same elements, potentially different order
        assert sorted(order_0) == sorted(order_1)
        assert order_0 != order_1  # very unlikely to be identical

    def test_auto_boundaries(self) -> None:
        lengths = list(range(1, 101))
        sampler = BucketBatchSampler(lengths, batch_size=10)
        assert len(sampler._boundaries) > 0
        assert sampler._boundaries == sorted(sampler._boundaries)


class TestComputePadFraction:
    def test_no_padding(self) -> None:
        assert compute_pad_fraction([3, 3, 3, 3]) == pytest.approx(0.0)

    def test_heavy_padding(self) -> None:
        # [1,1,1,10]: total=13, cells=40 → pad=1-13/40=0.675
        assert compute_pad_fraction([1, 1, 1, 10]) == pytest.approx(0.675)

    def test_empty(self) -> None:
        assert compute_pad_fraction([]) == 0.0

    def test_single_item(self) -> None:
        assert compute_pad_fraction([5]) == pytest.approx(0.0)


class TestDynamicPadCollate:
    def test_pads_to_batch_max(self) -> None:
        batch = [
            (torch.tensor([1, 2, 3]), torch.tensor([10, 20]), 0, 5),
            (torch.tensor([4, 5]), torch.tensor([30]), 1, 3),
        ]
        char_ids, tag_ids, tiers, roots = dynamic_pad_collate(batch)
        assert char_ids.shape == (2, 3)  # max char len = 3
        assert tag_ids.shape == (2, 2)   # max tag len = 2
        # Padding is 0
        assert char_ids[1, 2].item() == 0
        assert tag_ids[1, 1].item() == 0

    def test_preserves_values(self) -> None:
        batch = [
            (torch.tensor([1, 2]), torch.tensor([10]), 0, 5),
            (torch.tensor([3, 4]), torch.tensor([20]), 1, 3),
        ]
        char_ids, tag_ids, tiers, roots = dynamic_pad_collate(batch)
        assert char_ids[0].tolist() == [1, 2]
        assert char_ids[1].tolist() == [3, 4]
        assert tiers.tolist() == [0, 1]
        assert roots.tolist() == [5, 3]


class TestAnalyzeBatchingEfficiency:
    def test_bucket_better_than_naive(self) -> None:
        # Mix of short and long sequences
        lengths = [2] * 50 + [20] * 50
        result = analyze_batching_efficiency(lengths, batch_size=10)
        assert result["bucket_pad_fraction"] <= result["naive_pad_fraction"]
        assert result["speedup_estimate"] >= 1.0

    def test_uniform_lengths(self) -> None:
        # No benefit from bucketing when all same length
        lengths = [10] * 100
        result = analyze_batching_efficiency(lengths, batch_size=10)
        assert result["naive_pad_fraction"] == pytest.approx(0.0, abs=0.01)
        assert result["bucket_pad_fraction"] == pytest.approx(0.0, abs=0.01)

    def test_empty(self) -> None:
        result = analyze_batching_efficiency([], batch_size=10)
        assert result["speedup_estimate"] == 1.0
