"""Cross-source quality assessment for resource entries.

Resource tiers are determined by SOURCE TYPE, not confidence thresholds
(those belong in the training pipeline). Rules:

  gold   = entry originates from a UD treebank (boun or imst) — human annotated
  silver = ≥2 sources exist AND they agree on canonical_tags
  bronze = only 1 source, or multiple sources that disagree
"""
from __future__ import annotations
from collections import Counter

from resource.schema import MorphEntry

# Sources that are considered "gold" regardless of agreement
GOLD_SOURCES: frozenset[str] = frozenset({"boun", "imst"})


def assign_resource_tier(
    sources: list[str],
    canonical_tags_by_source: dict[str, str],
) -> str:
    """Assign resource tier based on source origin and cross-source agreement.

    Rules (checked in order):
      1. If ANY source in GOLD_SOURCES (boun, imst): tier = "gold"
      2. Else if len(sources) >= 2 AND all canonical_tags agree: tier = "silver"
      3. Else: tier = "bronze"

    Args:
        sources: List of source names that have this surface form.
        canonical_tags_by_source: Mapping of {source: canonical_tags_string}.

    Returns:
        "gold", "silver", or "bronze".
    """
    # Rule 1: any gold source → gold
    if any(s in GOLD_SOURCES for s in sources):
        return "gold"

    # Rule 2: multiple agreeing sources → silver
    if len(sources) >= 2:
        tags_values = list(canonical_tags_by_source.values())
        if len(set(tags_values)) == 1:  # all agree
            return "silver"

    # Rule 3: single source or disagreement → bronze
    return "bronze"


def compute_agreement(canonical_tags_by_source: dict[str, str]) -> float:
    """Compute agreement ratio: fraction of sources matching the plurality analysis.

    Used for statistics/reporting only — NOT used for tier assignment.

    Args:
        canonical_tags_by_source: Mapping of {source: canonical_tags_string}.

    Returns:
        Agreement ratio in [0.0, 1.0]. Returns 1.0 for single-source entries.
    """
    if not canonical_tags_by_source:
        return 0.0
    if len(canonical_tags_by_source) == 1:
        return 1.0

    counts = Counter(canonical_tags_by_source.values())
    max_count = counts.most_common(1)[0][1]
    return max_count / len(canonical_tags_by_source)


def tier_from_entries(entries: list[MorphEntry]) -> tuple[str, float]:
    """Derive tier and agreement from a list of MorphEntry for the same surface.

    Args:
        entries: All MorphEntry objects for a single surface form.

    Returns:
        Tuple of (tier, agreement_score).
    """
    sources = [e.source for e in entries]
    tags_by_source = {e.source: e.canonical_tags for e in entries}
    tier = assign_resource_tier(sources, tags_by_source)
    agreement = compute_agreement(tags_by_source)
    return tier, agreement
