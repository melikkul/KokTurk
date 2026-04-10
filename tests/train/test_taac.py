"""Tests for TAAC (Tier-Aware Adaptive Curriculum).

Covers:
1. Initial phase is "gold_only"
2. Transitions on plateau after patience epochs
3. No transition before min_epochs_per_phase
4. Force transition at max_epochs_per_phase regardless of improving loss
5. Stays at "all_final" (final phase) indefinitely
6. step() returns dict with all required keys
7. LR multipliers are monotonically decreasing across phases
8. allowed_tiers matches expected tier set per phase
"""
from __future__ import annotations

import pytest

from train.curriculum import TAAC, GOLD, SILVER_AUTO, SILVER_AGREED


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _advance_to_final(taac: TAAC) -> None:
    """Advance TAAC through all phases until it reaches 'all_final'."""
    while taac.current_phase != "all_final":
        taac.step(1.0)  # force transition via max_epochs (max_epochs_per_phase=1)


# ---------------------------------------------------------------------------
# 1. Initial phase
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_initial_phase(self):
        taac = TAAC()
        assert taac.current_phase == "gold_only"

    def test_initial_allowed_tiers(self):
        taac = TAAC()
        info = taac.step(1.0)
        # After first step we may or may not have transitioned, but gold_only
        # should be active at least initially; check the returned dict
        assert "allowed_tiers" in info
        assert isinstance(info["allowed_tiers"], set)

    def test_initial_phase_before_any_step(self):
        """current_phase must be 'gold_only' before any step() calls."""
        taac = TAAC()
        assert taac.current_phase == "gold_only"
        assert taac._phase_idx == 0


# ---------------------------------------------------------------------------
# 2. Transition on plateau
# ---------------------------------------------------------------------------

class TestTransitionOnPlateau:
    def test_transitions_after_patience_plateau_epochs(self):
        """Feed patience plateau steps → TAAC transitions to next phase."""
        taac = TAAC(epsilon=0.1, patience=2, min_epochs_per_phase=1, max_epochs_per_phase=20)
        # First call: _prev_loss is None, no plateau detected yet
        info1 = taac.step(1.0)
        assert not info1["should_transition"]
        assert taac.current_phase == "gold_only"
        # Second call: delta = |1.0 - 1.0| = 0.0 < epsilon=0.1 → plateau_count=1
        info2 = taac.step(1.0)
        assert not info2["should_transition"]   # plateau_count=1, patience=2
        # Third call: delta = 0.0 → plateau_count=2 >= patience=2 → transition
        info3 = taac.step(1.0)
        assert info3["should_transition"]
        assert taac.current_phase == "gold_and_silver_auto"

    def test_no_transition_if_loss_keeps_improving(self):
        """Continuously improving loss never triggers a plateau transition."""
        taac = TAAC(epsilon=0.01, patience=2, min_epochs_per_phase=1, max_epochs_per_phase=100)
        loss = 2.0
        for _ in range(20):
            loss -= 0.1
            info = taac.step(loss)
            assert not info["should_transition"]
        assert taac.current_phase == "gold_only"

    def test_plateau_count_resets_on_improvement(self):
        """An improving step resets the plateau counter."""
        taac = TAAC(epsilon=0.1, patience=2, min_epochs_per_phase=1, max_epochs_per_phase=20)
        taac.step(1.0)   # _prev_loss=1.0
        taac.step(1.0)   # plateau_count=1
        taac.step(0.5)   # improvement=0.5 >= 0.1 → plateau_count resets to 0
        info = taac.step(0.5)   # delta=0 → plateau_count=1, not 2 yet
        assert not info["should_transition"]


# ---------------------------------------------------------------------------
# 3. No transition before min_epochs
# ---------------------------------------------------------------------------

class TestMinEpochs:
    def test_no_transition_before_min_epochs(self):
        """Even with plateau, TAAC must not transition before min_epochs."""
        taac = TAAC(epsilon=0.5, patience=1, min_epochs_per_phase=3, max_epochs_per_phase=20)
        # Feed plateau losses
        info1 = taac.step(1.0)  # epoch_in_phase=1 < 3 → no check
        assert not info1["should_transition"]
        info2 = taac.step(1.0)  # epoch_in_phase=2 < 3 → no check
        assert not info2["should_transition"]
        # epoch_in_phase=3 >= min_epochs=3: now check plateau
        # _prev_loss=1.0, val=1.0, delta=0 < 0.5 → plateau_count=1 >= patience=1 → transition
        info3 = taac.step(1.0)
        assert info3["should_transition"]
        assert taac.current_phase == "gold_and_silver_auto"

    def test_still_at_initial_phase_after_one_plateau_step_with_high_min(self):
        """With min_epochs=5, no transition on plateau after only 2 steps."""
        taac = TAAC(epsilon=0.5, patience=1, min_epochs_per_phase=5, max_epochs_per_phase=20)
        taac.step(1.0)
        info = taac.step(1.0)
        assert not info["should_transition"]
        assert taac.current_phase == "gold_only"


# ---------------------------------------------------------------------------
# 4. Force transition at max_epochs
# ---------------------------------------------------------------------------

class TestMaxEpochs:
    def test_force_transition_at_max_epochs(self):
        """Transition is forced after max_epochs_per_phase even with improving loss."""
        taac = TAAC(epsilon=0.001, patience=5, min_epochs_per_phase=1, max_epochs_per_phase=3)
        loss = 1.0
        info1 = taac.step(loss); loss -= 0.1  # epoch=1
        assert not info1["should_transition"]
        info2 = taac.step(loss); loss -= 0.1  # epoch=2
        assert not info2["should_transition"]
        info3 = taac.step(loss)               # epoch=3 >= max_epochs=3 → force transition
        assert info3["should_transition"]
        assert taac.current_phase == "gold_and_silver_auto"

    def test_hard_cap_overrides_patience(self):
        """Even if patience would not trigger, max_epochs forces it."""
        taac = TAAC(epsilon=0.0, patience=100, min_epochs_per_phase=1, max_epochs_per_phase=2)
        taac.step(1.0)   # epoch=1
        info = taac.step(1.0)   # epoch=2 >= max → force
        assert info["should_transition"]


# ---------------------------------------------------------------------------
# 5. Final phase stability
# ---------------------------------------------------------------------------

class TestFinalPhaseStability:
    def test_stays_at_all_final(self):
        """Once at 'all_final', TAAC never transitions further."""
        # Use max_epochs=1 to race through phases quickly
        taac = TAAC(epsilon=100.0, patience=0, min_epochs_per_phase=1, max_epochs_per_phase=1)
        # Advance to all_final (5 phases → 4 transitions)
        for _ in range(4):
            taac.step(1.0)
        assert taac.current_phase == "all_final"
        # Feed many more steps — must stay at all_final
        for _ in range(10):
            info = taac.step(1.0)
            assert not info["should_transition"]
            assert taac.current_phase == "all_final"

    def test_phase_idx_does_not_exceed_bound(self):
        """_phase_idx never goes out of bounds."""
        taac = TAAC(epsilon=100.0, patience=0, min_epochs_per_phase=1, max_epochs_per_phase=1)
        for _ in range(20):
            taac.step(1.0)
        assert taac._phase_idx == len(TAAC.PHASES) - 1


# ---------------------------------------------------------------------------
# 6. step() return dict keys
# ---------------------------------------------------------------------------

class TestStepReturnDict:
    def test_returns_all_required_keys(self):
        taac = TAAC()
        info = taac.step(1.0)
        assert set(info.keys()) == {"phase", "allowed_tiers", "should_transition",
                                    "lr_multiplier", "tier_weights",
                                    "root_plateaued", "tag_plateaued"}

    def test_phase_is_string(self):
        taac = TAAC()
        info = taac.step(1.0)
        assert isinstance(info["phase"], str)

    def test_allowed_tiers_is_set(self):
        taac = TAAC()
        info = taac.step(1.0)
        assert isinstance(info["allowed_tiers"], set)

    def test_should_transition_is_bool(self):
        taac = TAAC()
        info = taac.step(1.0)
        assert isinstance(info["should_transition"], bool)

    def test_tier_weights_is_dict(self):
        taac = TAAC()
        info = taac.step(1.0)
        assert isinstance(info["tier_weights"], dict)
        # Keys should be integer tier indices
        assert all(isinstance(k, int) for k in info["tier_weights"])


# ---------------------------------------------------------------------------
# 7. LR multipliers monotonically decrease across phases
# ---------------------------------------------------------------------------

class TestLRMultipliers:
    def test_lr_multipliers_non_increasing(self):
        """LR multiplier should not increase across phase transitions."""
        multipliers = [TAAC.LR_MULTIPLIERS[p] for p in TAAC.PHASES]
        for i in range(len(multipliers) - 1):
            assert multipliers[i] >= multipliers[i + 1], (
                f"LR multiplier at phase {i} ({multipliers[i]}) must be >= "
                f"phase {i+1} ({multipliers[i+1]})"
            )
        # Overall should decrease from first to last
        assert multipliers[0] > multipliers[-1]

    def test_lr_multiplier_at_gold_only_is_1(self):
        taac = TAAC()
        info = taac.step(1.0)  # still in gold_only (no transition on first step)
        if not info["should_transition"]:
            assert info["lr_multiplier"] == 1.0

    def test_lr_multiplier_all_final_is_smallest(self):
        """all_final must have the smallest lr_multiplier."""
        all_mults = list(TAAC.LR_MULTIPLIERS.values())
        assert TAAC.LR_MULTIPLIERS["all_final"] == min(all_mults)


# ---------------------------------------------------------------------------
# 8. allowed_tiers per phase
# ---------------------------------------------------------------------------

class TestAllowedTiersPerPhase:
    def test_gold_only_tiers(self):
        taac = TAAC()
        info = taac.step(999.0)  # No transition (large prev_loss not set yet)
        assert taac.current_phase in ("gold_only", "gold_and_silver_auto")
        # Check mapping directly
        from train.curriculum import get_allowed_tiers
        assert get_allowed_tiers("gold_only") == {GOLD}

    def test_gold_and_silver_auto_tiers(self):
        from train.curriculum import get_allowed_tiers
        assert get_allowed_tiers("gold_and_silver_auto") == {GOLD, SILVER_AUTO}

    def test_all_tiers(self):
        from train.curriculum import get_allowed_tiers
        assert get_allowed_tiers("all_tiers") == {GOLD, SILVER_AUTO, SILVER_AGREED}

    def test_gold_calibration_same_as_gold_only(self):
        from train.curriculum import get_allowed_tiers
        assert get_allowed_tiers("gold_calibration") == {GOLD}

    def test_all_final_same_as_all_tiers(self):
        from train.curriculum import get_allowed_tiers
        assert get_allowed_tiers("all_final") == {GOLD, SILVER_AUTO, SILVER_AGREED}

    def test_taac_phases_list_coverage(self):
        """Every phase in TAAC.PHASES must be in _PHASE_TIERS."""
        from train.curriculum import get_allowed_tiers
        for phase in TAAC.PHASES:
            tiers = get_allowed_tiers(phase)
            assert isinstance(tiers, set) and len(tiers) > 0, (
                f"Phase '{phase}' has empty or missing tier set"
            )
