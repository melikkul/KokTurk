"""Domain-aware curriculum scheduling.

Composes (does NOT inherit) the existing :class:`train.curriculum.TAAC` so
its loss-plateau / component / ensemble transition logic and phase LR
schedule remain untouched. Adds an orthogonal *domain phase* state machine
on top:

    1. ``gold_only``           — gold tier, any domain
    2. ``silver_news``         — gold + silver-auto from news only
    3. ``multi_domain_bronze`` — gold + silver + new-domain bronze
    4. ``all_tiers``           — everything visible
    5. ``gold_calibration``    — gold only, final polish

The domain phase advances when TAAC transitions or after a configurable
cap (``min_epochs_per_domain_phase``). The corpus filter emitted by this
curriculum tells the training loop which ``(tier, domain)`` pairs are
visible in the current macro-phase.
"""

from __future__ import annotations

from dataclasses import dataclass

from train.curriculum import TAAC


@dataclass
class DomainFilter:
    """Per-macro-phase filter.

    ``allowed_tiers`` is any subset of ``{"gold", "silver-auto",
    "silver-agreed", "synthetic"}``. ``allowed_domains`` may be ``None``
    (meaning *any domain*) or a concrete set.
    """

    allowed_tiers: set[str]
    allowed_domains: set[str] | None
    domain_phase: str


class DomainAwareCurriculum:
    DOMAIN_PHASES: list[str] = [
        "gold_only",
        "silver_news",
        "multi_domain_bronze",
        "all_tiers",
        "gold_calibration",
    ]

    PHASE_FILTERS: dict[str, DomainFilter] = {
        "gold_only": DomainFilter({"gold"}, None, "gold_only"),
        "silver_news": DomainFilter(
            {"gold", "silver-auto"}, {"news"}, "silver_news"),
        "multi_domain_bronze": DomainFilter(
            {"gold", "silver-auto", "silver-agreed"},
            {"news", "social_media", "ecommerce", "creative_writing"},
            "multi_domain_bronze",
        ),
        "all_tiers": DomainFilter(
            {"gold", "silver-auto", "silver-agreed", "synthetic"},
            None, "all_tiers",
        ),
        "gold_calibration": DomainFilter({"gold"}, None, "gold_calibration"),
    }

    def __init__(
        self,
        taac: TAAC | None = None,
        min_epochs_per_domain_phase: int = 2,
    ) -> None:
        # COMPOSITION — NOT inheritance. TAAC's internal state is preserved.
        self.taac = taac if taac is not None else TAAC()
        self.min_epochs_per_domain_phase = min_epochs_per_domain_phase
        self._domain_idx = 0
        self._epochs_in_domain_phase = 0

    @property
    def current_domain_phase(self) -> str:
        return self.DOMAIN_PHASES[self._domain_idx]

    @property
    def current_filter(self) -> DomainFilter:
        return self.PHASE_FILTERS[self.current_domain_phase]

    def step(self, val_loss: float, **taac_kwargs: float | None) -> dict:
        """Advance one epoch. Delegates tier logic entirely to TAAC."""
        self._epochs_in_domain_phase += 1
        taac_state = self.taac.step(val_loss=val_loss, **taac_kwargs)  # type: ignore[arg-type]

        # Advance domain phase if TAAC transitioned AND the minimum epoch
        # budget for the current domain phase is satisfied.
        if (
            taac_state.get("should_transition")
            and self._epochs_in_domain_phase >= self.min_epochs_per_domain_phase
            and self._domain_idx < len(self.DOMAIN_PHASES) - 1
        ):
            self._domain_idx += 1
            self._epochs_in_domain_phase = 0

        return {
            "taac": taac_state,
            "domain_phase": self.current_domain_phase,
            "filter": self.current_filter,
        }

    def advance_domain(self) -> None:
        """Manually advance the domain phase (used by unit tests)."""
        if self._domain_idx < len(self.DOMAIN_PHASES) - 1:
            self._domain_idx += 1
            self._epochs_in_domain_phase = 0
