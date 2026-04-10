"""Tests for compound + -(s)I ambiguity evaluation (Cat B Task 4)."""

from __future__ import annotations

from benchmark.compound_eval import (
    COMPOUND_VERB_ROOTS,
    evaluate_compound_handling,
    evaluate_sI_ambiguity,
)
from benchmark.stratified_eval import _depth_bucket


def test_depth_bucket_splits_deep_tail():
    assert _depth_bucket(4) == "4"
    assert _depth_bucket(5) == "5"
    assert _depth_bucket(6) == "6"
    assert _depth_bucket(7) == "7"
    assert _depth_bucket(8) == "8+"
    assert _depth_bucket(12) == "8+"


def test_compound_verb_roots_all_in_table():
    from kokturk.core.compound_lexicon import FUSED_LVC_TABLE

    for root in COMPOUND_VERB_ROOTS:
        assert root in FUSED_LVC_TABLE, f"{root} missing from FUSED_LVC_TABLE"


def test_evaluate_compound_handling_counts_correctly():
    surfaces = ["reddetti", "geldi", "hissetti", "kayboldu", "evlerinden"]
    preds = ["ret +PAST", "gel +PAST", "his +PAST", "kayıp +PAST", "ev +PLU +ABL"]
    golds = ["ret +PAST", "gel +PAST", "his +PAST", "kayıp +PAST", "ev +PLU +ABL"]

    report = evaluate_compound_handling(surfaces, preds, golds)
    # Three compound tokens — all correct.
    assert report.n_total == 3
    assert report.n_correct == 3
    assert report.accuracy == 1.0
    assert "reddet" in report.per_stem


def test_evaluate_compound_handling_misses_count():
    surfaces = ["reddetti"]
    preds = ["reddet +PAST"]  # opaque root — wrong
    golds = ["ret +PAST"]
    report = evaluate_compound_handling(surfaces, preds, golds)
    assert report.n_total == 1
    assert report.n_correct == 0


def test_si_ambiguity_primary_stratum():
    tokens = ["evi", "kapı"]
    preds = ["ev +POSS.3SG", "kapı +NOM"]
    golds = ["ev +POSS.3SG", "kapı +NOM"]
    candidates = [
        [{"+POSS.3SG"}, {"+ACC"}],  # ambiguous
        [{"+NOM"}],                  # unambiguous
    ]
    report = evaluate_sI_ambiguity(
        tokens, preds, golds, candidate_tag_sets=candidates,
        preceding_tags=[set(), set()],
    )
    assert report.n_primary == 1
    assert report.em_primary == 1.0
    assert report.n_compound_context == 0


def test_si_ambiguity_compound_context_via_preceding_gen():
    tokens = ["kapısı"]
    preds = ["kapı +POSS.3SG"]
    golds = ["kapı +POSS.3SG"]
    report = evaluate_sI_ambiguity(
        tokens, preds, golds,
        candidate_tag_sets=[[{"+POSS.3SG"}]],
        preceding_tags=[{"+GEN"}],
    )
    assert report.n_compound_context == 1
    assert report.em_compound_context == 1.0
