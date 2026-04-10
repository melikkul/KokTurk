"""Tests for resource tier assignment."""
from __future__ import annotations
import pytest

from resource.quality_check import assign_resource_tier, compute_agreement, tier_from_entries
from resource.schema import MorphEntry


def _make_entry(source: str, tags: str = "ev +Noun", tier: str = "bronze") -> MorphEntry:
    return MorphEntry(
        surface="ev", lemma="ev", canonical_tags=tags,
        pos="NOUN", source=source, confidence=0.9,
        frequency=1, tier=tier,
    )


class TestAssignResourceTier:
    def test_boun_source_is_gold(self):
        tier = assign_resource_tier(["boun"], {"boun": "ev +Noun"})
        assert tier == "gold"

    def test_imst_source_is_gold(self):
        tier = assign_resource_tier(["imst"], {"imst": "ev +Noun"})
        assert tier == "gold"

    def test_boun_overrides_disagreement(self):
        """Even if sources disagree, boun/imst source → gold."""
        tier = assign_resource_tier(
            ["boun", "zeyrek"],
            {"boun": "ev +Noun", "zeyrek": "ev +Verb"},
        )
        assert tier == "gold"

    def test_unimorph_zeyrek_agree_is_silver(self):
        tags = "ev +Noun +PLU"
        tier = assign_resource_tier(
            ["unimorph", "zeyrek"],
            {"unimorph": tags, "zeyrek": tags},
        )
        assert tier == "silver"

    def test_zeyrek_only_is_bronze(self):
        tier = assign_resource_tier(["zeyrek"], {"zeyrek": "ev +Noun"})
        assert tier == "bronze"

    def test_disagreement_is_bronze(self):
        tier = assign_resource_tier(
            ["unimorph", "zeyrek"],
            {"unimorph": "ev +Noun +PLU", "zeyrek": "ev +Verb +PAST"},
        )
        assert tier == "bronze"


class TestComputeAgreement:
    def test_single_source_is_1(self):
        assert compute_agreement({"boun": "ev +Noun"}) == pytest.approx(1.0)

    def test_all_agree_is_1(self):
        assert compute_agreement({"a": "ev", "b": "ev", "c": "ev"}) == pytest.approx(1.0)

    def test_partial_agreement(self):
        # 2 out of 3 agree
        result = compute_agreement({"a": "ev", "b": "ev", "c": "git"})
        assert result == pytest.approx(2 / 3)

    def test_empty_is_0(self):
        assert compute_agreement({}) == pytest.approx(0.0)


class TestTierFromEntries:
    def test_boun_entry_gives_gold(self):
        entries = [_make_entry("boun")]
        tier, _ = tier_from_entries(entries)
        assert tier == "gold"

    def test_two_sources_agree_gives_silver(self):
        e1 = _make_entry("unimorph", tags="ev +Noun")
        e2 = _make_entry("zeyrek", tags="ev +Noun")
        tier, agreement = tier_from_entries([e1, e2])
        assert tier == "silver"
        assert agreement == pytest.approx(1.0)
