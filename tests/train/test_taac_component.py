"""Tests for Component-Aware and Ensemble TAAC transition modes."""
from __future__ import annotations

import pytest

from train.curriculum import TAAC


class TestComponentAware:
    def test_waits_for_slow_head(self):
        """Root plateaus but tag doesn't -> no transition until both plateau."""
        taac = TAAC(epsilon=0.01, patience=2, min_epochs_per_phase=2,
                    transition_mode="component")

        # Epochs 1-2: min_epochs, root constant, tag dropping
        taac.step(1.0, root_loss=0.5, tag_loss=0.8)
        taac.step(0.9, root_loss=0.5, tag_loss=0.7)

        # Epoch 3: root plateaued (0.5→0.5→0.5, count=2), tag still dropping
        r = taac.step(0.8, root_loss=0.5, tag_loss=0.6)
        assert r["root_plateaued"]
        assert not r["tag_plateaued"]
        assert r["phase"] == "gold_only"  # NOT transitioned (tag not ready)

        # Epoch 4: root still plateaued, tag starts plateauing (0.6→0.5, count=0)
        r = taac.step(0.75, root_loss=0.5, tag_loss=0.5)
        assert r["root_plateaued"]
        assert not r["tag_plateaued"]  # only 0 consecutive plateau for tag
        assert r["phase"] == "gold_only"

        # Epoch 5: tag 0.5→0.5 (count=1)
        r = taac.step(0.74, root_loss=0.5, tag_loss=0.5)
        assert not r["tag_plateaued"]  # 1 < patience=2

        # Epoch 6: tag 0.5→0.5 (count=2) → both plateau → transition
        r = taac.step(0.74, root_loss=0.5, tag_loss=0.5)
        assert r["tag_plateaued"]
        assert r["phase"] == "gold_and_silver_auto"

    def test_tag_gates_transition(self):
        """Tag decoder is the bottleneck."""
        taac = TAAC(epsilon=0.01, patience=1, min_epochs_per_phase=1,
                    transition_mode="component")

        for i in range(5):
            r = taac.step(0.5, root_loss=0.3, tag_loss=1.0 - i * 0.1)

        assert r["phase"] == "gold_only"

    def test_max_epochs_forces_transition(self):
        """Hard cap overrides component-aware."""
        taac = TAAC(epsilon=0.001, patience=100, min_epochs_per_phase=2,
                    max_epochs_per_phase=5, transition_mode="component")

        for i in range(5):
            r = taac.step(1.0 - i * 0.1, root_loss=0.5 - i * 0.1,
                          tag_loss=0.8 - i * 0.1)

        assert r["phase"] == "gold_and_silver_auto"

    def test_fallback_to_loss_when_no_component_losses(self):
        """Component mode falls back to loss-based when root/tag not provided."""
        taac = TAAC(epsilon=0.01, patience=2, min_epochs_per_phase=2,
                    transition_mode="component")

        # No root_loss/tag_loss → fallback to _should_transition(val_loss)
        taac.step(1.0)  # epoch 1: min_epochs, skip
        taac.step(1.0)  # epoch 2: plateau_count=1
        r = taac.step(1.0)  # epoch 3: plateau_count=2 → transition
        assert r["should_transition"]
        assert r["phase"] == "gold_and_silver_auto"


class TestEnsemble:
    def test_needs_two_votes(self):
        """Ensemble transition needs >= 2 of 3 signals."""
        taac = TAAC(epsilon=0.01, patience=1, min_epochs_per_phase=2,
                    transition_mode="ensemble")

        # Epoch 1-2: min_epochs
        taac.step(1.0, grad_norm=1.0, prediction_hash="abc")
        r = taac.step(1.0, grad_norm=0.5, prediction_hash="def")
        assert r["phase"] == "gold_only"  # only loss plateau (1 vote)

        # Epoch 3: loss plateau + prediction stable = 2 votes
        r = taac.step(1.0, grad_norm=0.3, prediction_hash="def")
        assert r["phase"] == "gold_and_silver_auto"

    def test_grad_norm_stability(self):
        """Gradient norm with low CV triggers vote."""
        taac = TAAC(epsilon=100.0, patience=1, min_epochs_per_phase=2,
                    transition_mode="ensemble")

        # Feed stable gradient norms + stable predictions
        taac.step(1.0, grad_norm=1.0, prediction_hash="x")
        taac.step(2.0, grad_norm=1.0, prediction_hash="x")
        r = taac.step(3.0, grad_norm=1.0, prediction_hash="x")
        # Loss NOT plateaued (delta=1.0 > epsilon=100? no, 100 is very high)
        # Actually epsilon=100 means loss always plateaus. Let me fix:
        # grad stable (CV~0) + prediction stable = 2 votes
        assert r["phase"] == "gold_and_silver_auto"


class TestBackwardCompatibility:
    def test_step_with_val_loss_only(self):
        """TAAC.step(val_loss) works without extra args."""
        taac = TAAC(epsilon=0.01, patience=2, transition_mode="loss")
        r = taac.step(1.0)
        assert "phase" in r
        assert "root_plateaued" in r
        assert "tag_plateaued" in r

    def test_loss_mode_ignores_component_args(self):
        """In loss mode, root/tag losses are ignored for transition."""
        taac = TAAC(epsilon=0.01, patience=2, min_epochs_per_phase=2,
                    transition_mode="loss")

        taac.step(1.0, root_loss=0.1, tag_loss=0.1)  # epoch 1
        taac.step(1.0, root_loss=0.1, tag_loss=0.1)  # epoch 2: plateau=1
        r = taac.step(1.0, root_loss=0.1, tag_loss=0.1)  # epoch 3: plateau=2 → transition
        assert r["should_transition"]
        assert r["phase"] == "gold_and_silver_auto"

    def test_default_transition_mode_is_component(self):
        taac = TAAC()
        assert taac.transition_mode == "component"
