"""Tests for EMA weights (Category C Task 4)."""
from __future__ import annotations

import torch
import torch.nn as nn

from train.ema import EMAWeights


def _make_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def test_shadow_matches_initial_params():
    model = _make_model()
    ema = EMAWeights(model, decay=0.9)
    for name, p in model.named_parameters():
        assert torch.equal(ema.shadow[name], p.detach())


def test_update_math():
    model = _make_model()
    ema = EMAWeights(model, decay=0.5)
    # Mutate params
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update()
    for name, p in model.named_parameters():
        # shadow was initial, then became 0.5*init + 0.5*(init+1) = init + 0.5
        expected = (p.detach() - 1.0) * 0.5 + p.detach() * 0.5
        assert torch.allclose(ema.shadow[name], expected, atol=1e-6)


def test_apply_restore_roundtrip():
    model = _make_model()
    ema = EMAWeights(model, decay=0.5)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(2.0)
    snapshot = {n: p.detach().clone() for n, p in model.named_parameters()}
    ema.apply()
    # Params now equal shadow (which is the pre-update init), not snapshot
    for n, p in model.named_parameters():
        assert not torch.equal(p, snapshot[n])
    ema.restore()
    for n, p in model.named_parameters():
        assert torch.equal(p, snapshot[n])


def test_decay_one_never_updates():
    model = _make_model()
    ema = EMAWeights(model, decay=1.0)
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(5.0)
    ema.update()
    for n in ema.shadow:
        assert torch.equal(ema.shadow[n], initial[n])
