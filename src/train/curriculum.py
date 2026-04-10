"""Curriculum training schedule for tiered morphological corpus.

Implements 4-phase curriculum, tier-weighted loss, inverse-sigmoid
scheduled sampling, phase-aware learning rate scheduling, and the
TAAC (Tier-Aware Adaptive Curriculum) automatic phase controller.
"""

from __future__ import annotations

import math

import torch

# Tier indices (must match TIER_MAP in datasets.py)
GOLD = 0
SILVER_AUTO = 1
SILVER_AGREED = 2

# Loss weights per tier: gold data is most valuable
TIER_WEIGHTS: dict[int, float] = {GOLD: 5.0, SILVER_AUTO: 1.0, SILVER_AGREED: 0.7}

# Phase definitions: which tiers are included
_PHASE_TIERS: dict[str, set[int]] = {
    "gold_only": {GOLD},
    "gold_and_silver_auto": {GOLD, SILVER_AUTO},
    "all_tiers": {GOLD, SILVER_AUTO, SILVER_AGREED},
    # TAAC-specific phase aliases
    "gold_calibration": {GOLD},
    "all_final": {GOLD, SILVER_AUTO, SILVER_AGREED},
}


def get_curriculum_phase(epoch: int, total_epochs: int = 30) -> str:
    """Determine curriculum phase for the given epoch (1-indexed).

    Phase 1 (epochs 1-3):   Gold only — clean initialization
    Phase 2 (epochs 4-12):  Gold + Silver-auto — scale with quality
    Phase 3 (epochs 13-20): All tiers — ambiguous token signal
    Phase 4 (epochs 21-25): Gold only — correct noise memorization
    Phase 5 (epochs 26-30): All tiers — final polish
    """
    if epoch <= 3:
        return "gold_only"
    elif epoch <= 12:
        return "gold_and_silver_auto"
    elif epoch <= 20:
        return "all_tiers"
    elif epoch <= 25:
        return "gold_only"
    else:
        return "all_tiers"


def get_allowed_tiers(phase: str) -> set[int]:
    """Return set of allowed tier indices for a curriculum phase."""
    return _PHASE_TIERS.get(phase, _PHASE_TIERS["all_tiers"])


def scheduled_sampling_ratio(epoch: int, k: float = 5.0) -> float:
    """Inverse sigmoid decay for teacher forcing ratio.

    Starts near 1.0 at epoch 0, drops to ~0.5 at epoch k,
    approaches 0.0 as epoch → ∞.

    Args:
        epoch: Current epoch (0-indexed).
        k: Decay speed parameter.

    Returns:
        Teacher forcing ratio in [0, 1].
    """
    return k / (k + math.exp(epoch / k))


def get_learning_rate(epoch: int, total_epochs: int = 30) -> float:
    """Phase-aware learning rate schedule matching curriculum phases.

    Phase 1 (epochs 1-3):   1e-4 → 5e-4  (warmup)
    Phase 2 (epochs 4-12):  5e-4 → 1e-4  (linear decay)
    Phase 3 (epochs 13-20): 1e-4 → 1e-5  (linear decay)
    Phase 4 (epochs 21-25): 5e-5 constant
    Phase 5 (epochs 26-30): 1e-5 constant
    """
    if epoch <= 3:
        # Warmup: linear from 1e-4 to 5e-4
        return 1e-4 + (5e-4 - 1e-4) * (epoch - 1) / max(3 - 1, 1)
    elif epoch <= 12:
        # Decay: linear from 5e-4 to 1e-4
        progress = (epoch - 4) / max(12 - 4, 1)
        return 5e-4 - (5e-4 - 1e-4) * progress
    elif epoch <= 20:
        # Decay: linear from 1e-4 to 1e-5
        progress = (epoch - 13) / max(20 - 13, 1)
        return 1e-4 - (1e-4 - 1e-5) * progress
    elif epoch <= 25:
        return 5e-5
    else:
        return 1e-5


def compute_tier_weights(tiers: torch.Tensor) -> torch.Tensor:
    """Convert tier indices to loss weight multipliers.

    Args:
        tiers: (B,) tensor of tier indices (0=gold, 1=silver-auto, 2=silver-agreed).

    Returns:
        (B,) tensor of per-sample weights.
    """
    weight_tensor = torch.tensor(
        [TIER_WEIGHTS.get(i, 1.0) for i in range(max(TIER_WEIGHTS.keys()) + 1)],
        device=tiers.device,
        dtype=torch.float,
    )
    return weight_tensor[tiers]


class TAAC:
    """Tier-Aware Adaptive Curriculum — automatic phase transitions.

    Supports three transition modes:

    ``"loss"``
        Original: transitions when aggregated validation loss plateaus.
    ``"component"``
        Component-aware: monitors root_loss and tag_loss independently.
        Transitions only when the **slower** head has plateaued (conjunctive).
        This prevents premature transitions when the root head converges
        fast but the tag decoder still needs more time on the current tier.
    ``"ensemble"``
        Majority vote of 3 signals: loss plateau, gradient norm stability,
        and prediction stability. Transitions when >= 2 signals agree.

    Phases (in order):
        1. ``"gold_only"``            — clean initialisation
        2. ``"gold_and_silver_auto"`` — scale with quality signal
        3. ``"all_tiers"``            — expose noisy silver-agreed data
        4. ``"gold_calibration"``     — correct noise memorisation
        5. ``"all_final"``            — final polish on all data

    Args:
        epsilon: Loss change threshold for plateau detection (default 0.01).
        patience: Consecutive plateau epochs required before transition (default 2).
        min_epochs_per_phase: Minimum epochs before any transition (default 2).
        max_epochs_per_phase: Force transition after this many phase-epochs (default 10).
        transition_mode: ``"loss"``, ``"component"``, or ``"ensemble"`` (default ``"component"``).
    """

    PHASES: list[str] = [
        "gold_only",
        "gold_and_silver_auto",
        "all_tiers",
        "gold_calibration",
        "all_final",
    ]

    LR_MULTIPLIERS: dict[str, float] = {
        "gold_only": 1.0,
        "gold_and_silver_auto": 1.0,
        "all_tiers": 0.8,
        "gold_calibration": 0.5,
        "all_final": 0.3,
    }

    def __init__(
        self,
        epsilon: float = 0.01,
        patience: int = 2,
        min_epochs_per_phase: int = 2,
        max_epochs_per_phase: int = 15,
        transition_mode: str = "component",
    ) -> None:
        self.epsilon = epsilon
        self.patience = patience
        self.min_epochs = min_epochs_per_phase
        self.max_epochs = max_epochs_per_phase
        self.transition_mode = transition_mode

        self._phase_idx: int = 0
        self._epoch_in_phase: int = 0
        self._plateau_count: int = 0
        self._prev_loss: float | None = None

        # Component-aware state
        self._root_prev_loss: float | None = None
        self._tag_prev_loss: float | None = None
        self._root_plateau_count: int = 0
        self._tag_plateau_count: int = 0

        # Ensemble state
        self._grad_norms: list[float] = []
        self._prediction_hashes: list[object] = []

    @property
    def current_phase(self) -> str:
        """Name of the current curriculum phase."""
        return self.PHASES[self._phase_idx]

    def step(
        self,
        val_loss: float,
        root_loss: float | None = None,
        tag_loss: float | None = None,
        grad_norm: float | None = None,
        prediction_hash: object | None = None,
    ) -> dict:
        """Consume one epoch's metrics and return curriculum state.

        Args:
            val_loss: Overall validation loss.
            root_loss: Root head validation loss (for ``"component"`` mode).
            tag_loss: Tag decoder validation loss (for ``"component"`` mode).
            grad_norm: L2 gradient norm on validation (for ``"ensemble"`` mode).
            prediction_hash: Hash of predictions on val set (for ``"ensemble"``).

        Returns:
            dict with ``phase``, ``allowed_tiers``, ``should_transition``,
            ``lr_multiplier``, ``tier_weights``, ``root_plateaued``, ``tag_plateaued``.
        """
        self._epoch_in_phase += 1

        # Choose transition strategy
        if self.transition_mode == "ensemble":
            should = self._should_transition_ensemble(val_loss, grad_norm, prediction_hash)
        elif self.transition_mode == "component" and root_loss is not None and tag_loss is not None:
            should = self._should_transition_component(root_loss, tag_loss)
        else:
            should = self._should_transition(val_loss)

        # Capture plateau state BEFORE reset (for reporting)
        root_plateaued = self._root_plateau_count >= self.patience
        tag_plateaued = self._tag_plateau_count >= self.patience

        transitioned = False
        if should and self._phase_idx < len(self.PHASES) - 1:
            self._phase_idx += 1
            self._epoch_in_phase = 1
            self._plateau_count = 0
            self._root_plateau_count = 0
            self._tag_plateau_count = 0
            self._prev_loss = None
            self._root_prev_loss = None
            self._tag_prev_loss = None
            self._grad_norms.clear()
            self._prediction_hashes.clear()
            transitioned = True

        self._prev_loss = val_loss
        if root_loss is not None:
            self._root_prev_loss = root_loss
        if tag_loss is not None:
            self._tag_prev_loss = tag_loss

        phase = self.current_phase
        return {
            "phase": phase,
            "allowed_tiers": get_allowed_tiers(phase),
            "should_transition": transitioned,
            "lr_multiplier": self.LR_MULTIPLIERS[phase],
            "tier_weights": dict(TIER_WEIGHTS),
            "root_plateaued": root_plateaued,
            "tag_plateaued": tag_plateaued,
        }

    # ------------------------------------------------------------------
    # Transition strategies
    # ------------------------------------------------------------------

    def _should_transition(self, val_loss: float) -> bool:
        """Original: aggregate loss plateau."""
        if self._epoch_in_phase >= self.max_epochs:
            return True
        if self._epoch_in_phase < self.min_epochs:
            return False
        if self._prev_loss is not None:
            if abs(self._prev_loss - val_loss) < self.epsilon:
                self._plateau_count += 1
            else:
                self._plateau_count = 0
        return self._plateau_count >= self.patience

    def _should_transition_component(self, root_loss: float, tag_loss: float) -> bool:
        """Component-aware: both root and tag heads must plateau (conjunctive).

        The slower head gates the transition — prevents premature advancement
        when root converges fast but the tag decoder still benefits from
        the current tier mix.
        """
        if self._epoch_in_phase >= self.max_epochs:
            return True
        if self._epoch_in_phase < self.min_epochs:
            return False

        if self._root_prev_loss is not None:
            if abs(root_loss - self._root_prev_loss) < self.epsilon:
                self._root_plateau_count += 1
            else:
                self._root_plateau_count = 0

        if self._tag_prev_loss is not None:
            if abs(tag_loss - self._tag_prev_loss) < self.epsilon:
                self._tag_plateau_count += 1
            else:
                self._tag_plateau_count = 0

        return (self._root_plateau_count >= self.patience
                and self._tag_plateau_count >= self.patience)

    def _should_transition_ensemble(
        self, val_loss: float,
        grad_norm: float | None, prediction_hash: object | None,
    ) -> bool:
        """Majority vote of 3 signals: loss plateau, grad stability, prediction stability."""
        if self._epoch_in_phase >= self.max_epochs:
            return True
        if self._epoch_in_phase < self.min_epochs:
            return False

        votes = 0

        # Signal 1: loss plateau
        if self._prev_loss is not None:
            if abs(val_loss - self._prev_loss) < self.epsilon:
                votes += 1

        # Signal 2: gradient norm stability (CV < 10% over last 3 epochs)
        if grad_norm is not None:
            self._grad_norms.append(grad_norm)
            if len(self._grad_norms) >= 3:
                recent = self._grad_norms[-3:]
                mean_gn = sum(recent) / len(recent)
                var_gn = sum((x - mean_gn) ** 2 for x in recent) / len(recent)
                cv = var_gn / (mean_gn ** 2 + 1e-10)
                if cv < 0.01:
                    votes += 1

        # Signal 3: prediction stability (identical predictions to prev epoch)
        if prediction_hash is not None:
            self._prediction_hashes.append(prediction_hash)
            if len(self._prediction_hashes) >= 2:
                if self._prediction_hashes[-1] == self._prediction_hashes[-2]:
                    votes += 1

        return votes >= 2
