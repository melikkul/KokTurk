"""Tests for R-Drop regularization (Category C Task 1)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from train.losses import FocalLoss, LabelSmoothingCE
from train.rdrop import compute_rdrop_loss, symmetric_kl


def _ce_none(logits, targets, ignore_index=0):
    return F.cross_entropy(logits, targets, ignore_index=ignore_index, reduction="none")


def test_symmetric_kl_zero_when_identical():
    torch.manual_seed(0)
    logits = torch.randn(6, 5)
    kl = symmetric_kl(logits, logits)
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)


def test_symmetric_kl_is_symmetric():
    torch.manual_seed(1)
    a = torch.randn(6, 5)
    b = torch.randn(6, 5)
    assert torch.allclose(symmetric_kl(a, b), symmetric_kl(b, a), atol=1e-6)


def test_rdrop_alpha_zero_matches_avg_ce():
    torch.manual_seed(2)
    logits_1 = torch.randn(8, 4, requires_grad=False)
    logits_2 = torch.randn(8, 4, requires_grad=False)
    targets = torch.randint(1, 4, (8,))
    ce, kl, total = compute_rdrop_loss(
        logits_1, logits_2, targets, _ce_none, alpha=0.0, ignore_index=0,
    )
    expected = 0.5 * (_ce_none(logits_1, targets) + _ce_none(logits_2, targets))
    assert torch.allclose(ce, expected, atol=1e-6)
    assert torch.allclose(total, ce, atol=1e-6)
    # KL is nonzero in general
    assert (kl > 0).any()


def test_rdrop_ignore_index_masks_kl():
    torch.manual_seed(3)
    logits_1 = torch.randn(4, 5)
    logits_2 = torch.randn(4, 5)
    targets = torch.tensor([0, 0, 0, 0])  # all pad
    _, kl, total = compute_rdrop_loss(
        logits_1, logits_2, targets, _ce_none, alpha=5.0, ignore_index=0,
    )
    assert torch.allclose(kl, torch.zeros_like(kl))
    # total equals ce (which is itself zero because all positions masked by loss_fn)
    assert torch.allclose(total, torch.zeros_like(total), atol=1e-6)


def test_rdrop_3d_logits_flatten():
    torch.manual_seed(4)
    logits_1 = torch.randn(2, 3, 5)
    logits_2 = torch.randn(2, 3, 5)
    targets = torch.randint(1, 5, (2, 3))
    ce, kl, total = compute_rdrop_loss(
        logits_1, logits_2, targets, _ce_none, alpha=1.0, ignore_index=0,
    )
    assert ce.shape == (6,)
    assert kl.shape == (6,)
    assert total.shape == (6,)


def test_rdrop_integrates_with_focal_loss():
    torch.manual_seed(5)
    focal = FocalLoss(gamma=2.0, ignore_index=0)
    logits_1 = torch.randn(6, 4)
    logits_2 = torch.randn(6, 4)
    targets = torch.randint(1, 4, (6,))
    _, _, total = compute_rdrop_loss(
        logits_1, logits_2, targets, focal, alpha=5.0, ignore_index=0,
    )
    assert total.shape == (6,)
    assert torch.isfinite(total).all()


def test_rdrop_integrates_with_label_smoothing():
    ls = LabelSmoothingCE(epsilon=0.01, ignore_index=0)
    logits_1 = torch.randn(5, 3)
    logits_2 = torch.randn(5, 3)
    targets = torch.tensor([1, 2, 0, 1, 2])
    _, _, total = compute_rdrop_loss(
        logits_1, logits_2, targets, ls, alpha=3.0, ignore_index=0,
    )
    assert torch.isfinite(total).all()
