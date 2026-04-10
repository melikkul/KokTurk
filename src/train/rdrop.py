"""R-Drop: Regularized Dropout for seq2seq models.

Liang et al., 2021 — "R-Drop: Regularized Dropout for Neural Networks".

Each sample is forwarded twice through the model with active dropout,
producing two output distributions P1 and P2. The training loss becomes::

    L = 0.5 * (CE(P1, target) + CE(P2, target)) + alpha * sym_KL(P1, P2)

where ``sym_KL(P1, P2) = 0.5 * (KL(P1 || P2) + KL(P2 || P1))``. Padding
positions (``targets == ignore_index``) contribute zero to the KL term so
they do not leak gradient into the consistency objective.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

__all__ = ["compute_rdrop_loss", "symmetric_kl"]


def symmetric_kl(
    logits_1: torch.Tensor,
    logits_2: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-position symmetric KL divergence between two logit tensors.

    Args:
        logits_1: ``(N, V)`` logits from pass 1.
        logits_2: ``(N, V)`` logits from pass 2.
        mask: optional ``(N,)`` float mask; masked positions contribute 0.

    Returns:
        Per-position KL of shape ``(N,)``.
    """
    log_p1 = F.log_softmax(logits_1, dim=-1)
    log_p2 = F.log_softmax(logits_2, dim=-1)
    p1 = log_p1.exp()
    p2 = log_p2.exp()
    kl_12 = (p1 * (log_p1 - log_p2)).sum(dim=-1)
    kl_21 = (p2 * (log_p2 - log_p1)).sum(dim=-1)
    kl = 0.5 * (kl_12 + kl_21)
    if mask is not None:
        kl = kl * mask.to(kl.dtype)
    return kl


def compute_rdrop_loss(
    logits_1: torch.Tensor,
    logits_2: torch.Tensor,
    targets: torch.Tensor,
    base_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    alpha: float = 5.0,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute R-Drop loss components.

    Both ``logits_1`` and ``logits_2`` can be either ``(B, L, V)`` or
    ``(N, V)``. They are flattened to ``(N, V)`` internally. Targets must
    be broadcastable to ``(N,)``.

    Args:
        logits_1: first forward-pass logits.
        logits_2: second forward-pass logits.
        targets: gold class indices.
        base_loss_fn: callable taking ``(logits (N,V), targets (N,))`` and
            returning per-element loss of shape ``(N,)``. Typically a
            ``FocalLoss`` / ``LabelSmoothingCE`` / ``SymmetricCrossEntropy``
            instance from :mod:`src.train.losses`.
        alpha: weight on the symmetric KL term.
        ignore_index: target index that is excluded from the KL term.

    Returns:
        ``(ce_per_elem, kl_per_elem, total_per_elem)`` — all shape ``(N,)``.
        The caller applies tier weights / pad masking to ``ce_per_elem``
        and averages, matching existing v4 training semantics. The KL term
        is returned unweighted (the caller multiplies by ``alpha`` to
        avoid double-scaling when composing with tier weights).
    """
    if logits_1.shape != logits_2.shape:
        raise ValueError(
            f"R-Drop logit shapes mismatch: {tuple(logits_1.shape)} vs "
            f"{tuple(logits_2.shape)}"
        )
    if logits_1.dim() == 3:
        n, v = logits_1.size(0) * logits_1.size(1), logits_1.size(2)
        logits_1 = logits_1.reshape(n, v)
        logits_2 = logits_2.reshape(n, v)
    if targets.dim() > 1:
        targets = targets.reshape(-1)

    ce_1 = base_loss_fn(logits_1, targets)
    ce_2 = base_loss_fn(logits_2, targets)
    ce = 0.5 * (ce_1 + ce_2)

    mask = (targets != ignore_index).to(logits_1.dtype)
    kl = symmetric_kl(logits_1, logits_2, mask=mask)

    total = ce + alpha * kl
    return ce, kl, total
