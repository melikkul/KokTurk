"""Tests for fused LVC decomposition lexicon and corpus mining."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kokturk.core.compound_lexicon import (
    FUSED_LVC_TABLE,
    decompose_fused_lvc,
    is_fused_lvc,
)
from kokturk.core.lvc_mining import (
    _has_morphophonological_alternation,
    _restore_nominal_candidates,
    mine_fused_lvcs_from_corpus,
)


class TestDecomposeFusedLvc:
    def test_known_gemination_form(self):
        result = decompose_fused_lvc("reddetti")
        assert result == ("ret", "et", "ti")

    def test_known_progressive_form(self):
        result = decompose_fused_lvc("hissediyorum")
        assert result == ("his", "et", "iyorum")

    def test_known_ol_form(self):
        result = decompose_fused_lvc("kayboldu")
        assert result == ("kayıp", "ol", "du")

    def test_known_elision_form(self):
        result = decompose_fused_lvc("emretti")
        assert result == ("emir", "et", "ti")

    def test_non_fused_returns_none(self):
        assert decompose_fused_lvc("geliyor") is None
        assert decompose_fused_lvc("evlerinden") is None
        assert decompose_fused_lvc("bitirdi") is None

    def test_empty_input(self):
        assert decompose_fused_lvc("") is None

    def test_longest_prefix_match_wins(self):
        # naklet (6) must shadow shorter prefixes that share a stem.
        result = decompose_fused_lvc("nakletti")
        assert result is not None
        assert result[0] == "nakil"
        assert result[1] == "et"

    def test_case_insensitive(self):
        result = decompose_fused_lvc("Reddetti")
        assert result is not None
        assert result[0] == "ret"

    def test_is_fused_lvc(self):
        assert is_fused_lvc("reddet")
        assert is_fused_lvc("kaybol")
        assert not is_fused_lvc("gel")
        assert not is_fused_lvc("git")


class TestMorphophonologicalFilter:
    def test_gemination_accepted(self):
        # ret + et → reddet  (t doubled)
        assert _has_morphophonological_alternation("ret", "reddet", "et")

    def test_elision_accepted(self):
        # emir + et → emret  (medial i dropped)
        assert _has_morphophonological_alternation("emir", "emret", "et")

    def test_voicing_accepted(self):
        # nominal "akit" + et → akdet (t→d) is the kind of pattern.
        assert _has_morphophonological_alternation("akit", "akdet", "et")

    def test_plain_concatenation_rejected(self):
        # Plain "gel + et" would be "gelet" — no morphophonology, NOT an LVC.
        assert not _has_morphophonological_alternation("gel", "gelet", "et")

    def test_unrelated_rejected(self):
        assert not _has_morphophonological_alternation("ev", "gel", "et")


class TestRestoreCandidates:
    def test_degemination_candidate(self):
        cands = _restore_nominal_candidates("redd")
        assert "red" in cands

    def test_devoicing_candidate(self):
        cands = _restore_nominal_candidates("akd")
        assert "akt" in cands


class TestMineCorpus:
    def test_mining_smoke_skips_normal_verbs(self):
        # A toy zeyrek-like analyzer that returns:
        #   "geldi" → Verb root "gel"
        # should NOT be discovered as a fused LVC because "gel + et" is
        # plain concatenation with no morphophonology.
        def analyze(word):
            if word == "geldi":
                return [[SimpleNamespace(pos="Verb", lemma="gel")]]
            return [[]]

        zeyrek = SimpleNamespace(analyze=analyze)
        discovered = mine_fused_lvcs_from_corpus(["geldi"], zeyrek)
        assert discovered == {}

    def test_mining_handles_zeyrek_exceptions(self):
        def analyze(word):
            raise RuntimeError("zeyrek exploded")

        zeyrek = SimpleNamespace(analyze=analyze)
        # Must not propagate.
        assert mine_fused_lvcs_from_corpus(["foo"], zeyrek) == {}


def test_table_entries_are_well_formed():
    for stem, (nominal, lv) in FUSED_LVC_TABLE.items():
        assert isinstance(stem, str) and stem
        assert isinstance(nominal, str) and nominal
        assert lv in ("et", "ol")


def test_analyzer_decompose_lvc_flag(monkeypatch):
    # Skip the heavy ZeyrekBackend init by monkeypatching the registry to a
    # fake backend that returns a known parse for "reddetti".
    from kokturk.core import analyzer as analyzer_module
    from kokturk.core.datatypes import MorphologicalAnalysis

    class FakeBackend:
        def analyze(self, word):
            return [
                MorphologicalAnalysis(
                    surface=word,
                    root="reddet",
                    tags=("+PAST", "+3SG"),
                    morphemes=(),
                    source="fake",
                    score=1.0,
                )
            ]

        def close(self):
            pass

    monkeypatch.setitem(
        analyzer_module._BACKEND_REGISTRY, "fake", FakeBackend
    )
    a = analyzer_module.MorphoAnalyzer(backends=["fake"])

    # Default: opaque verb root.
    default_result = a.analyze("reddetti")
    assert default_result.analyses[0].root == "reddet"

    # With decompose_lvc=True: nominal root + LVC tag.
    lvc_result = a.analyze("reddetti", decompose_lvc=True)
    assert lvc_result.analyses[0].root == "ret"
    assert lvc_result.analyses[0].tags[0] == "+LVC.ET"
    assert "+PAST" in lvc_result.analyses[0].tags
