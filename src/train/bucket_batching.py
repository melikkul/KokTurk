"""Dynamic bucket batching for efficient GRU training.

Groups sequences by length into buckets, then samples batches within each
bucket.  This minimizes padding waste and dramatically improves GPU
utilization for variable-length Turkish words.

Research: bucket batching can reduce pad fraction from 40-60% to <5%,
yielding 2-3x training speedup with zero accuracy impact.

Usage::

    lengths = dataset.char_lengths
    sampler = BucketBatchSampler(lengths, batch_size=64)
    loader = DataLoader(dataset, batch_sampler=sampler,
                        collate_fn=dynamic_pad_collate)
"""

from __future__ import annotations

import math
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Sampler


class BucketBatchSampler(Sampler[list[int]]):
    """Sampler that groups dataset indices by sequence length into buckets.

    Args:
        lengths: Sequence length for each dataset item.
        batch_size: Target batch size.
        bucket_boundaries: Length boundaries for buckets.  If ``None``,
            automatically computed from length percentiles.
        shuffle: Shuffle within buckets each epoch (default ``True``).
        drop_last: Drop incomplete final batch per bucket.
        indices: Optional subset of dataset indices to sample from
            (e.g. tier-filtered).  ``None`` means all indices.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        lengths: list[int],
        batch_size: int = 64,
        bucket_boundaries: list[int] | None = None,
        shuffle: bool = True,
        drop_last: bool = False,
        indices: list[int] | None = None,
        seed: int | None = None,
    ) -> None:
        self._lengths = lengths
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._seed = seed
        self._epoch = 0

        # Subset of indices to consider
        self._indices = indices if indices is not None else list(range(len(lengths)))

        if bucket_boundaries is None:
            subset_lengths = [lengths[i] for i in self._indices]
            bucket_boundaries = self._auto_boundaries(subset_lengths)
        self._boundaries = sorted(bucket_boundaries)

        # Assign indices to buckets
        self._buckets: list[list[int]] = [[] for _ in range(len(self._boundaries) + 1)]
        for idx in self._indices:
            bucket_id = self._assign_bucket(lengths[idx])
            self._buckets[bucket_id].append(idx)

    def _auto_boundaries(
        self, lengths: list[int], num_buckets: int = 10,
    ) -> list[int]:
        """Compute bucket boundaries from length percentiles."""
        if not lengths:
            return []
        percentiles = np.linspace(0, 100, num_buckets + 1)[1:-1]
        boundaries = np.percentile(lengths, percentiles).astype(int)
        return sorted(set(int(b) for b in boundaries))

    def _assign_bucket(self, length: int) -> int:
        """Return the bucket index for a given sequence length."""
        for i, boundary in enumerate(self._boundaries):
            if length <= boundary:
                return i
        return len(self._boundaries)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch number for deterministic shuffling."""
        self._epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(
            seed=(self._seed + self._epoch) if self._seed is not None else None,
        )

        all_batches: list[list[int]] = []
        for bucket in self._buckets:
            if not bucket:
                continue
            indices = list(bucket)
            if self._shuffle:
                rng.shuffle(indices)
            # Chunk into batches
            for start in range(0, len(indices), self._batch_size):
                batch = indices[start : start + self._batch_size]
                if self._drop_last and len(batch) < self._batch_size:
                    continue
                all_batches.append(batch)

        # Shuffle batch order across buckets
        if self._shuffle:
            rng.shuffle(all_batches)

        yield from all_batches

    def __len__(self) -> int:
        total = 0
        for bucket in self._buckets:
            if not bucket:
                continue
            n_batches = len(bucket) // self._batch_size
            if not self._drop_last and len(bucket) % self._batch_size > 0:
                n_batches += 1
            total += n_batches
        return total


# ------------------------------------------------------------------
# Collate function for dynamic padding
# ------------------------------------------------------------------

def dynamic_pad_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor, int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate that pads char_ids and tag_ids to per-batch max.

    Unlike the default collate (which requires equal-length tensors and
    thus needs pre-padding to a global max), this pads each sequence to
    the longest in the *current batch*.

    Expects each item to be ``(char_ids, tag_ids, tier, root_idx)``
    where ``char_ids`` and ``tag_ids`` may differ in length across items.

    Returns:
        ``(char_ids, tag_ids, tiers, root_idxs)`` — same shape contract
        as the default stacking, but with per-batch padding.
    """
    char_list, tag_list, tiers, roots = [], [], [], []
    for char_ids, tag_ids, tier, root_idx in batch:
        char_list.append(char_ids)
        tag_list.append(tag_ids)
        tiers.append(tier)
        roots.append(root_idx)

    char_ids_padded = _pad_sequences(char_list)
    tag_ids_padded = _pad_sequences(tag_list)

    return (
        char_ids_padded,
        tag_ids_padded,
        torch.tensor(tiers, dtype=torch.long),
        torch.tensor(roots, dtype=torch.long),
    )


def _pad_sequences(seqs: list[torch.Tensor], pad_value: int = 0) -> torch.Tensor:
    """Pad a list of 1-D tensors to the length of the longest."""
    max_len = max(s.size(0) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_value, dtype=seqs[0].dtype)
    for i, s in enumerate(seqs):
        out[i, : s.size(0)] = s
    return out


# ------------------------------------------------------------------
# Efficiency analysis
# ------------------------------------------------------------------

def compute_pad_fraction(batch_lengths: list[int]) -> float:
    """Compute the fraction of padding in a batch.

    ``pad_fraction = 1 - sum(lengths) / (max_length * batch_size)``

    A value > 0.30 suggests bucket batching is recommended.
    """
    if not batch_lengths:
        return 0.0
    max_len = max(batch_lengths)
    total_cells = max_len * len(batch_lengths)
    return 1.0 - sum(batch_lengths) / total_cells if total_cells > 0 else 0.0


def analyze_batching_efficiency(
    lengths: list[int],
    batch_size: int = 64,
) -> dict[str, float]:
    """Compare naive vs bucket batching pad fractions.

    Args:
        lengths: Sequence lengths for the entire dataset.
        batch_size: Target batch size.

    Returns:
        Dict with ``naive_pad_fraction``, ``bucket_pad_fraction``,
        and ``speedup_estimate``.
    """
    if not lengths:
        return {
            "naive_pad_fraction": 0.0,
            "bucket_pad_fraction": 0.0,
            "speedup_estimate": 1.0,
        }

    rng = np.random.default_rng(seed=42)

    # Naive: random batches
    shuffled = list(range(len(lengths)))
    rng.shuffle(shuffled)
    naive_fracs = []
    for start in range(0, len(shuffled), batch_size):
        batch_idx = shuffled[start : start + batch_size]
        batch_lens = [lengths[i] for i in batch_idx]
        naive_fracs.append(compute_pad_fraction(batch_lens))
    naive_avg = float(np.mean(naive_fracs)) if naive_fracs else 0.0

    # Bucket: sorted then batched
    sampler = BucketBatchSampler(lengths, batch_size=batch_size, shuffle=False)
    bucket_fracs = []
    for batch_indices in sampler:
        batch_lens = [lengths[i] for i in batch_indices]
        bucket_fracs.append(compute_pad_fraction(batch_lens))
    bucket_avg = float(np.mean(bucket_fracs)) if bucket_fracs else 0.0

    useful_naive = 1.0 - naive_avg
    useful_bucket = 1.0 - bucket_avg
    speedup = useful_bucket / useful_naive if useful_naive > 0 else 1.0

    return {
        "naive_pad_fraction": naive_avg,
        "bucket_pad_fraction": bucket_avg,
        "speedup_estimate": speedup,
    }
