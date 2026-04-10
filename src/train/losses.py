"""Loss functions for the kokturk training pipeline.

All losses return un-reduced per-element loss of shape ``(N,)`` where
``N = B * L`` — matching the convention in
:mod:`src.train.train_v4_master` where logits are reshaped to ``(N, V)`` and
targets to ``(N,)`` before the loss call. Downstream code multiplies the
per-element loss by tier weights and a non-pad mask, then averages.

Numerical stability:
- :class:`FocalLoss` clamps ``p_t`` to ``[1e-7, 1 - 1e-7]`` before
  ``log``/``(1 - p_t) ** gamma``.
- :class:`SymmetricCrossEntropy` clamps the one-hot target distribution to
  ``[1e-4, 1.0]`` before computing the reverse CE (RCE) term — otherwise
  ``log(0)`` yields ``-inf``.

Invariants (unit-tested):
- ``FocalLoss(gamma=0, label_smoothing=0)`` is numerically equivalent to
  ``F.cross_entropy(..., reduction='none')`` within fp32 tolerance.
- ``LabelSmoothingCE(epsilon=0)`` is equivalent to standard CE.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "FocalLoss",
    "SymmetricCrossEntropy",
    "LabelSmoothingCE",
    "build_loss",
]


def _valid_mask(targets: torch.Tensor, ignore_index: int) -> torch.Tensor:
    return (targets != ignore_index).to(targets.device)


class FocalLoss(nn.Module):
    """Focal Loss with optional label smoothing.

    FL(p_t) = -alpha_t * (1 - p_t) ** gamma * log(p_t)

    Args:
        gamma: focusing parameter. ``gamma=0`` reduces to standard CE.
        alpha: per-class weight tensor of shape ``(num_classes,)`` or ``None``.
        ignore_index: target index to ignore (zero-loss, does not contribute).
        label_smoothing: epsilon in ``[0, 1)``. When > 0 the target one-hot is
            smoothed uniformly over the full vocabulary before computing the
            Focal term — research reports this compound form outperforms plain
            Focal on imbalanced morphology.

            **Warning**: for morphological seq2seq with tag vocabularies
            larger than ~1K, keep epsilon ≤ 0.01. Values ≥ 0.1 cause
            generation collapse where the decoder over-predicts
            high-frequency tags.

    Output shape: ``(N,)`` per-element loss, where ``N = B * L`` for
    sequence inputs reshaped by the caller.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if not (0.0 <= label_smoothing < 1.0):
            raise ValueError(
                f"label_smoothing must be in [0, 1), got {label_smoothing}"
            )
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None  # type: ignore[assignment]

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        # logits: (N, V), targets: (N,)
        if logits.dim() != 2:
            raise ValueError(
                f"FocalLoss expects (N, V) logits; got shape {tuple(logits.shape)}"
            )
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        n, num_classes = logits.shape
        mask = _valid_mask(targets, self.ignore_index)
        safe_targets = targets.clamp(min=0)  # ignored entries zeroed out at end

        # Gather correct-class probability p_t
        p_t = probs.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
        p_t = p_t.clamp(min=1e-7, max=1.0 - 1e-7)
        log_p_t = torch.log(p_t)

        # Standard focal term on the true class
        focal_weight = (1.0 - p_t).pow(self.gamma)
        loss_true = -focal_weight * log_p_t  # (N,)

        if self.label_smoothing > 0:
            # Smoothed target: (1 - eps) * one_hot + eps / K
            # Focal on the smoothed distribution: average over classes using
            # q = eps / K uniformly and (1 - eps) on the true class.
            eps = self.label_smoothing
            # Uniform component: -((1-p_k)^gamma) * log(p_k) averaged
            uniform_probs = probs.clamp(min=1e-7, max=1.0 - 1e-7)
            uniform_focal = (1.0 - uniform_probs).pow(self.gamma) * (-torch.log(uniform_probs))
            uniform_term = uniform_focal.mean(dim=-1)  # (N,)
            loss = (1.0 - eps) * loss_true + eps * uniform_term
        else:
            loss = loss_true

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[safe_targets]
            loss = loss * alpha_t

        loss = loss * mask.to(loss.dtype)
        return loss


class SymmetricCrossEntropy(nn.Module):
    """Symmetric Cross Entropy: SCE = alpha * CE + beta * RCE.

    RCE(p, q) = -sum_k p_k * log(q_k) where p is the model prediction and
    q is the (smoothed) target distribution. One-hot q must be clamped to
    ``[1e-4, 1.0]`` before the log — otherwise the log explodes to ``-inf``.

    Args:
        alpha: weight on the standard (forward) CE term.
        beta: weight on the reverse CE term.
        num_classes: vocabulary size (needed to construct q).
        ignore_index: target index to ignore.
    """

    def __init__(
        self,
        num_classes: int,
        alpha: float = 1.0,
        beta: float = 0.4,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError(f"num_classes must be > 1, got {num_classes}")
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if logits.dim() != 2:
            raise ValueError(
                f"SCE expects (N, V) logits; got shape {tuple(logits.shape)}"
            )
        mask = _valid_mask(targets, self.ignore_index)
        safe_targets = targets.clamp(min=0)

        # Forward CE term
        ce = F.cross_entropy(
            logits, safe_targets, ignore_index=self.ignore_index,
            reduction="none",
        )

        # Reverse CE term — needs clamped q (one-hot) and model probs p
        probs = F.softmax(logits, dim=-1).clamp(min=1e-7)
        q = F.one_hot(safe_targets, num_classes=self.num_classes).float()
        q = q.clamp(min=1e-4, max=1.0)
        log_q = torch.log(q)
        rce = -(probs * log_q).sum(dim=-1)  # (N,)

        loss = self.alpha * ce + self.beta * rce
        loss = loss * mask.to(loss.dtype)
        return loss


class LabelSmoothingCE(nn.Module):
    """Cross-Entropy with uniform label smoothing.

    ``epsilon=0`` is equivalent to standard cross-entropy.
    """

    def __init__(self, epsilon: float = 0.1, ignore_index: int = -100) -> None:
        super().__init__()
        if not (0.0 <= epsilon < 1.0):
            raise ValueError(f"epsilon must be in [0, 1), got {epsilon}")
        self.epsilon = epsilon
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if logits.dim() != 2:
            raise ValueError(
                f"LabelSmoothingCE expects (N, V) logits; got shape {tuple(logits.shape)}"
            )
        log_probs = F.log_softmax(logits, dim=-1)
        mask = _valid_mask(targets, self.ignore_index)
        safe_targets = targets.clamp(min=0)

        nll = -log_probs.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
        if self.epsilon > 0:
            smooth = -log_probs.mean(dim=-1)
            loss = (1.0 - self.epsilon) * nll + self.epsilon * smooth
        else:
            loss = nll
        loss = loss * mask.to(loss.dtype)
        return loss


def build_loss(
    name: str,
    *,
    num_classes: int,
    ignore_index: int = 0,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.1,
    sce_alpha: float = 1.0,
    sce_beta: float = 0.4,
) -> nn.Module:
    """Factory keyed by the ``--loss-fn`` CLI flag.

    Returns a module producing per-element loss of shape ``(N,)``. Callers are
    expected to multiply by tier weights and a non-pad mask before averaging.
    """
    name = name.lower()
    if name == "ce":
        # Equivalent to F.cross_entropy(..., reduction='none')
        return LabelSmoothingCE(epsilon=0.0, ignore_index=ignore_index)
    if name == "focal":
        return FocalLoss(
            gamma=focal_gamma,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )
    if name == "sce":
        return SymmetricCrossEntropy(
            num_classes=num_classes,
            alpha=sce_alpha,
            beta=sce_beta,
            ignore_index=ignore_index,
        )
    if name == "label_smooth":
        return LabelSmoothingCE(
            epsilon=label_smoothing, ignore_index=ignore_index,
        )
    raise ValueError(f"Unknown loss: {name!r}")
