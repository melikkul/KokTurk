"""Tests for benchmark.weighted_em."""

from __future__ import annotations

import pytest

from benchmark.weighted_em import (
    compute_emma_f1,
    corpus_weighted_em,
    score_pair,
    weighted_exact_match,
)


def test_emma_identical():
    assert compute_emma_f1(["+PLU", "+LOC"], ["+PLU", "+LOC"]) == 1.0


def test_emma_disjoint():
    assert compute_emma_f1(["+PLU"], ["+ACC"]) == 0.0


def test_emma_empty_both():
    assert compute_emma_f1([], []) == 1.0


def test_emma_family_partial():
    # +POSS.3SG vs +POSS.1SG share the POSS family -> 0.5 similarity.
    score = compute_emma_f1(["+POSS.3SG"], ["+POSS.1SG"])
    assert 0.0 < score < 1.0


def test_weighted_em_perfect():
    s = weighted_exact_match("ev", "ev", "+NOUN", "+NOUN", [], [], ["+PLU"], ["+PLU"])
    assert s == pytest.approx(1.0)


def test_weighted_em_root_wrong():
    # Only POS + deriv + infl match, lemma wrong -> 0.5 missing.
    s = weighted_exact_match("ev", "el", "+NOUN", "+NOUN", [], [], ["+PLU"], ["+PLU"])
    assert s == pytest.approx(0.50, abs=0.01)


def test_weighted_em_tags_wrong():
    s = weighted_exact_match("ev", "ev", "+NOUN", "+NOUN", [], [], ["+PLU"], ["+ACC"])
    # lemma 0.5 + pos 0.2 + deriv 0.15 (empty both -> 1.0) + infl 0.0 = 0.85.
    assert s == pytest.approx(0.85, abs=0.01)


def test_score_pair_perfect():
    assert score_pair("ev +NOUN +PLU", "ev +NOUN +PLU") == pytest.approx(1.0)


def test_corpus_weighted_em():
    gold = ["ev +NOUN +PLU", "el +NOUN +ACC"]
    pred = ["ev +NOUN +PLU", "el +NOUN +ACC"]
    assert corpus_weighted_em(gold, pred) == pytest.approx(1.0)


def test_corpus_weighted_em_mixed():
    gold = ["ev +NOUN +PLU", "el +NOUN +ACC"]
    pred = ["ev +NOUN +PLU", "xx +NOUN +ACC"]
    score = corpus_weighted_em(gold, pred)
    assert 0.5 < score < 1.0
