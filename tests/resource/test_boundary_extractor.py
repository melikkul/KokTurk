"""Tests for morpheme boundary extraction."""
from __future__ import annotations

import pytest

from resource.boundary_extractor import extract_boundaries, _find_root_end


class TestExtractBoundaries:
    def test_simple_plural(self):
        assert extract_boundaries("evler", "ev +PLU") == "ev|ler"

    def test_accusative(self):
        result = extract_boundaries("evi", "ev +ACC")
        assert result.startswith("ev|")

    def test_dative(self):
        result = extract_boundaries("eve", "ev +DAT")
        assert result == "ev|e"

    def test_locative(self):
        result = extract_boundaries("evde", "ev +LOC")
        assert result == "ev|de"

    def test_ablative(self):
        result = extract_boundaries("evden", "ev +ABL")
        assert result == "ev|den"

    def test_genitive(self):
        result = extract_boundaries("evin", "ev +GEN")
        assert result == "ev|in"

    def test_multi_suffix(self):
        result = extract_boundaries("evlerinden", "ev +PLU +POSS.3SG +ABL")
        assert "|" in result
        assert result.startswith("ev|")
        parts = result.split("|")
        assert len(parts) >= 3

    def test_no_suffix_bare_root(self):
        assert extract_boundaries("ev", "ev") == "ev"

    def test_nominative_zero_morph(self):
        # +NOM has no surface form — should return just root
        assert extract_boundaries("ev", "ev +NOM") == "ev"

    def test_empty_canonical(self):
        assert extract_boundaries("ev", "") == "ev"


class TestConsonantMutation:
    def test_kitap_to_kitab(self):
        result = extract_boundaries("kitabı", "kitap +ACC")
        assert "kitab" in result

    def test_find_root_end_mutation(self):
        end = _find_root_end("kitabı", "kitap")
        assert end == 5  # len("kitab")


class TestVowelDeletion:
    def test_find_root_end_default(self):
        # Simple case — no mutation needed
        end = _find_root_end("evler", "ev")
        assert end == 2


class TestEdgeCases:
    def test_empty_surface(self):
        result = extract_boundaries("", "ev +PLU")
        # Should handle gracefully
        assert isinstance(result, str)

    def test_root_longer_than_surface(self):
        # Root claim is longer than surface — fallback
        result = extract_boundaries("ev", "evler +PLU")
        assert isinstance(result, str)
