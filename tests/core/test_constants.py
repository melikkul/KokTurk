"""Tests for tag mapping constants.

These tests ensure the Functional Canonization is correct.
If mappings are wrong, everything downstream breaks.
"""

from __future__ import annotations

import re

from kokturk.core.constants import (
    TRMORPH_TO_CANONICAL,
    UNMARKED_FEATURES,
    ZEYREK_TO_CANONICAL,
)


class TestZeyrekToCanonical:
    def test_all_canonical_tags_follow_format(self) -> None:
        """Every non-empty canonical tag must start with + followed by alphanumeric."""
        pattern = re.compile(r"^\+[A-Za-z0-9][A-Za-z0-9.]*$")
        for key, value in ZEYREK_TO_CANONICAL.items():
            if value:  # skip unmarked (empty string)
                assert pattern.match(value), (
                    f"Canonical tag for '{key}' is '{value}' — "
                    f"expected format: +UPPERCASE (e.g., +PLU, +POSS.3SG)"
                )

    def test_no_none_values(self) -> None:
        """All values must be strings, never None."""
        for key, value in ZEYREK_TO_CANONICAL.items():
            assert isinstance(value, str), f"Value for '{key}' is {type(value)}, expected str"

    def test_case_suffixes_present(self) -> None:
        """All 7 Turkish cases must be mapped."""
        expected_cases = {"+NOM", "+ACC", "+DAT", "+LOC", "+ABL", "+GEN", "+INS"}
        actual = set(ZEYREK_TO_CANONICAL.values())
        assert expected_cases.issubset(actual), (
            f"Missing case mappings: {expected_cases - actual}"
        )

    def test_possession_paradigm_complete(self) -> None:
        """All 6 possession suffixes must be mapped."""
        expected = {"+POSS.1SG", "+POSS.2SG", "+POSS.3SG",
                    "+POSS.1PL", "+POSS.2PL", "+POSS.3PL"}
        actual = set(ZEYREK_TO_CANONICAL.values())
        assert expected.issubset(actual), f"Missing: {expected - actual}"

    def test_tense_tags_present(self) -> None:
        """Core tense/aspect tags must be mapped."""
        expected = {"+PAST", "+EVID", "+AOR", "+PROG", "+FUT"}
        actual = set(ZEYREK_TO_CANONICAL.values())
        assert expected.issubset(actual), f"Missing: {expected - actual}"

    def test_known_mappings(self) -> None:
        """Spot-check critical mappings."""
        assert ZEYREK_TO_CANONICAL["Loc"] == "+LOC"
        assert ZEYREK_TO_CANONICAL["A3pl"] == "+PLU"
        assert ZEYREK_TO_CANONICAL["P3sg"] == "+POSS.3SG"
        assert ZEYREK_TO_CANONICAL["Abl"] == "+ABL"
        assert ZEYREK_TO_CANONICAL["Past"] == "+PAST"
        assert ZEYREK_TO_CANONICAL["Narr"] == "+EVID"

    def test_unmarked_features_are_empty(self) -> None:
        """Unmarked features (A3sg, Pnon, etc.) map to empty string."""
        assert ZEYREK_TO_CANONICAL["A3sg"] == ""
        assert ZEYREK_TO_CANONICAL["Pnon"] == ""

    def test_has_sufficient_mappings(self) -> None:
        """Should have at least 50 mappings to cover core Turkish morphology."""
        assert len(ZEYREK_TO_CANONICAL) >= 50


class TestTRMorphToCanonical:
    def test_all_canonical_tags_follow_format(self) -> None:
        """Every non-empty canonical tag must start with + followed by alphanumeric."""
        pattern = re.compile(r"^\+[A-Za-z0-9][A-Za-z0-9.]*$")
        for key, value in TRMORPH_TO_CANONICAL.items():
            if value:
                assert pattern.match(value), (
                    f"TRMorph canonical tag for '{key}' is '{value}' — bad format"
                )

    def test_canonical_alignment(self) -> None:
        """TRMorph canonical values must be a subset of Zeyrek canonical values.

        Both backends must produce the same canonical tag space.
        """
        zeyrek_tags = {v for v in ZEYREK_TO_CANONICAL.values() if v}
        trmorph_tags = {v for v in TRMORPH_TO_CANONICAL.values() if v}
        unaligned = trmorph_tags - zeyrek_tags
        assert not unaligned, (
            f"TRMorph has canonical tags not in Zeyrek mapping: {unaligned}"
        )


class TestUnmarkedFeatures:
    def test_unmarked_set_populated(self) -> None:
        assert len(UNMARKED_FEATURES) >= 3
        assert "A3sg" in UNMARKED_FEATURES
        assert "Pnon" in UNMARKED_FEATURES
