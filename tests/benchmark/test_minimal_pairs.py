"""Tests for benchmark.minimal_pairs."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.minimal_pairs import (
    Pair,
    evaluate_minimal_pairs,
    load_pairs,
)


def test_load_pairs_from_yaml():
    pairs = load_pairs()
    assert len(pairs) >= 30
    assert all(isinstance(p, Pair) for p in pairs)


def test_load_pairs_has_phenomenon_coverage():
    pairs = load_pairs()
    phenomena = {p.phenomenon for p in pairs}
    # Required linguistic categories from the plan.
    for cat in ("case_acc_dat", "vowel_harmony_back", "negation_verb", "voice_pass"):
        assert cat in phenomena


def test_evaluate_oracle_model(tmp_path: Path):
    pairs = [
        Pair("case", "evde", ("+NOUN", "+LOC"), "evden", ("+NOUN", "+ABL")),
        Pair("plur", "ev", ("+NOUN",), "evler", ("+NOUN", "+PLU")),
    ]
    table = {
        "evde": ("+NOUN", "+LOC"),
        "evden": ("+NOUN", "+ABL"),
        "ev": ("+NOUN",),
        "evler": ("+NOUN", "+PLU"),
    }

    out = tmp_path / "mp.md"
    report = evaluate_minimal_pairs(lambda w: table[w], pairs=pairs, output_path=out)
    assert out.exists()
    assert report.total_pass == 2
    assert report.total_fail == 0


def test_evaluate_broken_model_fails():
    pairs = [Pair("case", "evde", ("+NOUN", "+LOC"), "evden", ("+NOUN", "+ABL"))]
    report = evaluate_minimal_pairs(lambda w: ("+NOUN",), pairs=pairs)
    assert report.total_fail == 1
