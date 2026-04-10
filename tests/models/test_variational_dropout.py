"""Tests for variational dropout & weight dropout (Category C Task 2)."""
from __future__ import annotations

import torch
import torch.nn as nn

from kokturk.models.variational_dropout import VariationalDropout, WeightDropout


def test_variational_dropout_identity_when_p_zero():
    vdrop = VariationalDropout(p=0.0)
    vdrop.train()
    x = torch.randn(2, 5, 8)
    assert torch.equal(vdrop(x), x)


def test_variational_dropout_identity_in_eval():
    vdrop = VariationalDropout(p=0.5)
    vdrop.eval()
    x = torch.randn(2, 5, 8)
    assert torch.equal(vdrop(x), x)


def test_variational_dropout_mask_locked_across_time():
    torch.manual_seed(0)
    vdrop = VariationalDropout(p=0.5)
    vdrop.train()
    x = torch.ones(4, 7, 16)
    y = vdrop(x)
    # Mask must be identical across the time dim: y[:, 0, :] == y[:, t, :]
    first = y[:, 0, :]
    for t in range(1, 7):
        assert torch.equal(y[:, t, :], first)


def test_variational_dropout_scaling_preserves_mean():
    torch.manual_seed(1)
    vdrop = VariationalDropout(p=0.3)
    vdrop.train()
    x = torch.ones(64, 10, 32)
    y = vdrop(x)
    # Expected value preserved (1/(1-p) scaling).
    assert abs(y.mean().item() - 1.0) < 0.05


def test_weight_dropout_drops_during_train_restores_in_eval():
    torch.manual_seed(2)
    gru = nn.GRU(input_size=4, hidden_size=6, num_layers=1, batch_first=True)
    wd = WeightDropout(gru, ["weight_hh_l0"], dropout=0.5)
    raw = wd.module.weight_hh_l0_raw.data.clone()

    wd.train()
    x = torch.randn(2, 3, 4)
    _ = wd(x)
    # During training the effective weight may have zeros from dropout.
    w_train = wd.module.weight_hh_l0
    # Raw untouched
    assert torch.equal(wd.module.weight_hh_l0_raw.data, raw)
    # Some entries zeroed with high probability (fixed seed)
    assert (w_train == 0).any()

    wd.eval()
    _ = wd(x)
    w_eval = wd.module.weight_hh_l0
    assert torch.equal(w_eval, raw)


def test_weight_dropout_zero_is_noop():
    gru = nn.GRU(input_size=3, hidden_size=5, batch_first=True)
    raw_before = gru.weight_hh_l0.data.clone()
    wd = WeightDropout(gru, ["weight_hh_l0"], dropout=0.0)
    wd.train()
    _ = wd(torch.randn(1, 2, 3))
    assert torch.equal(wd.module.weight_hh_l0_raw.data, raw_before)
    # effective weight equals raw
    assert torch.equal(wd.module.weight_hh_l0, raw_before)
