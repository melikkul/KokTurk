"""Contrastive root head for polysemy resolution.

A standalone opt-in replacement for the standard :class:`RootHead` MLP. Kept
in its own module so it does NOT change ``DualHeadAtomizer.__init__``
defaults — existing v2 checkpoints still load with ``strict=True``.

Architecture
------------
1. Pool encoder outputs (mean pooling with mask).
2. Project the pooled vector to a shared embedding space.
3. Project each candidate root's embedding to the same space.
4. Loss: margin-based triplet (Euclidean) with hard prefix-sharing negatives.

Negative sampling
-----------------
A precomputed ``LongTensor(num_roots, max_negatives)`` with ``-1`` padding is
built once at construction time from ``root_vocab.json``. For every root we
store up to ``max_negatives`` other roots sharing at least the first two
characters. At train time the index is gathered on-device — no Python-level
sampling in the hot path. Callers are responsible for ensuring ≥3 hard
negatives per anchor (falling back to random same-POS when a root has no
prefix neighbours).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_prefix_groups(
    root_vocab: dict[str, int] | list[str],
    prefix_len: int = 2,
    max_negatives: int = 16,
) -> tuple[torch.Tensor, list[str]]:
    """Return a ``(num_roots, max_negatives)`` LongTensor of hard negatives.

    Padding value is ``-1``. Index ``i`` lists the (up to ``max_negatives``)
    root indices that share the first ``prefix_len`` characters with root
    ``i``, excluding ``i`` itself.
    """
    if isinstance(root_vocab, list):
        roots = list(root_vocab)
    else:
        # {token: idx} → sorted by idx
        roots = [t for t, _ in sorted(root_vocab.items(), key=lambda kv: kv[1])]

    prefix_to_indices: dict[str, list[int]] = {}
    for idx, r in enumerate(roots):
        key = r[:prefix_len]
        prefix_to_indices.setdefault(key, []).append(idx)

    neg_idx = torch.full((len(roots), max_negatives), -1, dtype=torch.long)
    for idx, r in enumerate(roots):
        key = r[:prefix_len]
        candidates = [j for j in prefix_to_indices.get(key, []) if j != idx]
        for k, j in enumerate(candidates[:max_negatives]):
            neg_idx[idx, k] = j
    return neg_idx, roots


class ContrastiveRootHead(nn.Module):
    """Euclidean margin-loss root classifier.

    Args:
        encoder_dim: Dimension of encoder output vectors.
        root_vocab_size: Number of roots in the vocabulary.
        embed_dim: Shared projection-space dimension.
        margin: Triplet margin (Euclidean). CLI flag is
            ``--contrastive-margin`` (default 1.0).
        negative_index: Precomputed ``(V, max_negatives)`` LongTensor of hard
            negatives. Should come from :func:`build_prefix_groups`.
        dropout: Dropout applied before the projection.
    """

    def __init__(
        self,
        encoder_dim: int,
        root_vocab_size: int,
        embed_dim: int = 128,
        margin: float = 1.0,
        negative_index: torch.Tensor | None = None,
        dropout: float = 0.3,
        min_hard_negatives: int = 3,
    ) -> None:
        super().__init__()
        self.margin = margin
        self.min_hard_negatives = min_hard_negatives
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(encoder_dim, embed_dim)
        self.root_embeddings = nn.Embedding(root_vocab_size, embed_dim)

        if negative_index is None:
            negative_index = torch.full(
                (root_vocab_size, max(min_hard_negatives, 1)), -1,
                dtype=torch.long,
            )
        elif negative_index.shape[0] != root_vocab_size:
            raise ValueError(
                f"negative_index first dim ({negative_index.shape[0]}) != "
                f"root_vocab_size ({root_vocab_size})"
            )
        # Stored as buffer so it moves with the module to GPU.
        self.register_buffer("negative_index", negative_index, persistent=False)

    # ----- Pooling -----
    def _pool(
        self,
        encoder_outputs: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if mask is None:
            return encoder_outputs.mean(dim=1)
        m = mask.float().unsqueeze(-1)
        summed = (encoder_outputs * m).sum(dim=1)
        denom = m.sum(dim=1).clamp(min=1.0)
        return summed / denom

    # ----- Forward -----
    def forward(
        self,
        encoder_outputs: torch.Tensor,
        mask: torch.Tensor | None = None,
        gold_root: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return ``{distances, pooled_embed, loss}``.

        ``distances`` has shape ``(B, V)`` — negative Euclidean distance from
        the anchor to every root embedding. Callers treat it as a logit
        (argmax = predicted root).

        ``loss`` is computed only when ``gold_root`` is provided.
        """
        pooled = self._pool(encoder_outputs, mask)
        pooled = self.dropout(pooled)
        anchor = self.proj(pooled)  # (B, E)

        all_embeds = self.root_embeddings.weight  # (V, E)
        # Distances to every root: shape (B, V).
        dist_full = torch.cdist(anchor.unsqueeze(0), all_embeds.unsqueeze(0)).squeeze(0)
        logits = -dist_full  # argmax = closest = predicted

        result = {"distances": dist_full, "pooled_embed": anchor, "logits": logits}

        if gold_root is None:
            return result

        pos_emb = self.root_embeddings(gold_root)  # (B, E)
        d_pos = F.pairwise_distance(anchor, pos_emb)  # (B,)

        # Gather hard negatives per anchor.
        neg_idx = self.negative_index[gold_root]  # (B, K)
        B, K = neg_idx.shape
        valid = neg_idx >= 0  # (B, K)
        # Clamp -1 to 0 so the gather is safe; mask those out below.
        safe = neg_idx.clamp(min=0)
        neg_emb = self.root_embeddings(safe)  # (B, K, E)
        d_neg = torch.norm(anchor.unsqueeze(1) - neg_emb, dim=-1)  # (B, K)
        # Mask invalid positions with +inf so they do not lower the min.
        d_neg = d_neg.masked_fill(~valid, float("inf"))

        # If a row has <min_hard_negatives valid entries, fall back to random
        # in-batch negatives drawn from the gold_root vector itself.
        n_valid = valid.sum(dim=1)
        need_fallback = n_valid < self.min_hard_negatives
        if need_fallback.any() and B > 1:
            perm = torch.randperm(B, device=anchor.device)
            rand_negs = self.root_embeddings(gold_root[perm])  # (B, E)
            d_rand = F.pairwise_distance(anchor, rand_negs)  # (B,)
            # Replace inf columns with the random-negative distance.
            d_rand_expand = d_rand.unsqueeze(1).expand(B, K)
            d_neg = torch.where(
                torch.isinf(d_neg) & need_fallback.unsqueeze(1),
                d_rand_expand, d_neg,
            )

        # Hardest negative (minimum distance).
        d_neg_hard, _ = d_neg.min(dim=1)
        # Replace any remaining inf (shouldn't happen for B>1) with d_pos +
        # margin so the triplet is satisfied for those rows.
        d_neg_hard = torch.where(
            torch.isinf(d_neg_hard),
            d_pos + self.margin,
            d_neg_hard,
        )

        triplet = torch.relu(d_pos - d_neg_hard + self.margin)
        result["loss"] = triplet.mean()
        return result


def load_prefix_negatives_from_file(
    root_vocab_path: Path | str,
    prefix_len: int = 2,
    max_negatives: int = 16,
) -> torch.Tensor:
    """Load a ``root_vocab.json`` and return the precomputed negative index."""
    data = json.loads(Path(root_vocab_path).read_text())
    neg, _ = build_prefix_groups(data, prefix_len=prefix_len, max_negatives=max_negatives)
    return neg
