"""Quality checking and tier assignment for morphological data."""
from __future__ import annotations

_GOLD_SOURCES: frozenset[str] = frozenset({"boun", "imst"})


class QualityChecker:
    """Assign quality tiers and validate morphological entries.

    Tier rules (source-based):
        gold   = from UD treebank (BOUN/IMST) — human-annotated
        silver = >=2 independent sources agree on analysis
        bronze = single source only
    """

    def assign_tier(self, sources: list[str], tags_agree: bool = False) -> str:
        """Assign quality tier based on source provenance."""
        if any(s in _GOLD_SOURCES for s in sources):
            return "gold"
        if len(sources) >= 2 and tags_agree:
            return "silver"
        return "bronze"

    def validate_entry(
        self, surface: str, canonical: str, pos: str = ""
    ) -> list[str]:
        """Return list of validation errors (empty if valid)."""
        errors: list[str] = []
        if not surface or not surface.strip():
            errors.append("Empty surface form")
        if not canonical or not canonical.strip():
            errors.append("Empty canonical tags")
        parts = canonical.split()
        if parts and not parts[0].startswith("+"):
            for tag in parts[1:]:
                if not tag.startswith("+"):
                    errors.append(f"Tag missing + prefix: {tag}")
        return errors
