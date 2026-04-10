"""Tests for src.train.losses.

Core invariants:
- FocalLoss(gamma=0, label_smoothing=0) ≡ F.cross_entropy(reduction='none').
- LabelSmoothingCE(epsilon=0) ≡ F.cross_entropy(reduction='none').
- All losses return shape (N,) per-element loss compatible with the tier-weight
  broadcast in train_v4_master.py.
- ignore_index positions contribute zero loss.
- No NaN/inf on zero-probability targets (numerical stability guard).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from train.losses import (
    FocalLoss,
    LabelSmoothingCE,
    SymmetricCrossEntropy,
    build_loss,
)


def _fake_logits(n: int = 8, v: int = 10, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    logits = torch.randn(n, v, requires_grad=True)
    targets = torch.randint(0, v, (n,))
    return logits, targets


def test_focal_gamma_zero_equals_ce() -> None:
    logits, targets = _fake_logits()
    loss_fn = FocalLoss(gamma=0.0, label_smoothing=0.0, ignore_index=-100)
    focal = loss_fn(logits, targets)
    ce = F.cross_entropy(logits, targets, reduction="none")
    assert focal.shape == ce.shape == (8,)
    assert torch.allclose(focal, ce, atol=1e-5)


def test_focal_gamma_positive_differs_from_ce() -> None:
    logits, targets = _fake_logits()
    focal = FocalLoss(gamma=2.0)(logits, targets)
    ce = F.cross_entropy(logits, targets, reduction="none")
    # Focal down-weights well-classified examples → strictly <= CE pointwise.
    assert (focal <= ce + 1e-6).all()


def test_focal_ignore_index_zero_contribution() -> None:
    logits, targets = _fake_logits()
    targets[0] = 0
    loss_fn = FocalLoss(gamma=2.0, ignore_index=0)
    out = loss_fn(logits, targets)
    assert out[0].item() == 0.0
    assert out[1:].sum().item() > 0.0


def test_focal_no_nan_on_extreme_logits() -> None:
    # Extreme logits that would blow up a naive log(0) implementation.
    logits = torch.tensor([[-1e4, 1e4], [1e4, -1e4]], dtype=torch.float32, requires_grad=True)
    targets = torch.tensor([0, 1])  # Both "correct" picks get p_t ≈ 0
    loss_fn = FocalLoss(gamma=2.0)
    out = loss_fn(logits, targets)
    assert torch.isfinite(out).all(), f"FocalLoss produced non-finite: {out}"
    out.sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_focal_label_smoothing_nonzero() -> None:
    logits, targets = _fake_logits()
    out_plain = FocalLoss(gamma=2.0, label_smoothing=0.0)(logits, targets)
    out_smooth = FocalLoss(gamma=2.0, label_smoothing=0.1)(logits, targets)
    # Smoothed form differs from plain
    assert not torch.allclose(out_plain, out_smooth)
    assert torch.isfinite(out_smooth).all()


def test_label_smoothing_eps_zero_equals_ce() -> None:
    logits, targets = _fake_logits()
    loss = LabelSmoothingCE(epsilon=0.0, ignore_index=-100)(logits, targets)
    ce = F.cross_entropy(logits, targets, reduction="none")
    assert torch.allclose(loss, ce, atol=1e-6)


def test_label_smoothing_ignore_index() -> None:
    logits, targets = _fake_logits()
    targets[0] = 0
    loss = LabelSmoothingCE(epsilon=0.1, ignore_index=0)(logits, targets)
    assert loss[0].item() == 0.0


def test_sce_shape_and_finiteness() -> None:
    logits, targets = _fake_logits(v=10)
    loss_fn = SymmetricCrossEntropy(num_classes=10, alpha=1.0, beta=0.4, ignore_index=-100)
    out = loss_fn(logits, targets)
    assert out.shape == (8,)
    assert torch.isfinite(out).all()


def test_sce_ignore_index() -> None:
    logits, targets = _fake_logits(v=10)
    targets[0] = 0
    out = SymmetricCrossEntropy(num_classes=10, ignore_index=0)(logits, targets)
    assert out[0].item() == 0.0


def test_sce_no_neg_inf_on_onehot_log() -> None:
    # Without q clamping, log(0) on the non-target classes would be -inf.
    logits = torch.randn(4, 5, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3])
    out = SymmetricCrossEntropy(num_classes=5)(logits, targets)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_tier_weight_integration_shape() -> None:
    # Losses must be broadcastable against the (B*L,) tier-weight tensor from
    # train_v4_master.py.
    B, L, V = 4, 6, 12
    logits = torch.randn(B * L, V)
    targets = torch.randint(1, V, (B * L,))
    for loss_fn in [
        FocalLoss(gamma=2.0),
        LabelSmoothingCE(epsilon=0.1),
        SymmetricCrossEntropy(num_classes=V),
    ]:
        out = loss_fn(logits, targets)
        assert out.shape == (B * L,)
        tier_weights = torch.rand(B).unsqueeze(1).expand(B, L).reshape(-1)
        weighted = (out * tier_weights).mean()
        assert torch.isfinite(weighted)


def test_build_loss_factory() -> None:
    for name in ("ce", "focal", "sce", "label_smooth"):
        fn = build_loss(name, num_classes=10, ignore_index=0)
        logits = torch.randn(4, 10)
        targets = torch.randint(1, 10, (4,))
        out = fn(logits, targets)
        assert out.shape == (4,)
        assert torch.isfinite(out).all()
