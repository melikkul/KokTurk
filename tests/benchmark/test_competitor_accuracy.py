"""Tests for benchmark.competitor_accuracy."""

from __future__ import annotations

import dataclasses
from unittest.mock import patch

import pytest

from benchmark.competitor_accuracy import (
    CompetitorResult,
    _compare_tags,
    convert_ud_feats_to_canonical,
    evaluate_spacy,
    evaluate_stanza,
    evaluate_udpipe,
    generate_positioning_summary,
    run_competitor_benchmark,
)
from benchmark.standard_benchmarks import ud_to_canonical


def test_convert_delegates_to_ud_to_canonical():
    """convert_ud_feats_to_canonical must produce the same output as ud_to_canonical."""
    feats = "Case=Abl|Number=Plur"
    upos = "NOUN"
    assert convert_ud_feats_to_canonical(feats, upos) == ud_to_canonical(feats, upos=upos)


def test_convert_empty_feats():
    assert convert_ud_feats_to_canonical("_") == []
    assert convert_ud_feats_to_canonical("_", upos="VERB") == ["+VERB"]


def test_compare_tags_exact_match():
    em, uf, pos = _compare_tags(["+Noun", "+PLU", "+ABL"], ["+Noun", "+PLU", "+ABL"])
    assert em is True
    assert uf is True
    assert pos is True


def test_compare_tags_different_feats():
    em, uf, pos = _compare_tags(["+Noun", "+PLU"], ["+Noun", "+ABL"])
    assert em is False
    assert uf is False
    assert pos is True  # both have +Noun


def test_compare_tags_different_pos():
    em, uf, pos = _compare_tags(["+Noun", "+PLU"], ["+Verb", "+PLU"])
    assert em is False
    assert uf is True
    assert pos is False


@patch("importlib.util.find_spec", return_value=None)
def test_evaluate_stanza_returns_none_when_missing(mock_spec):
    assert evaluate_stanza([]) is None


@patch("importlib.util.find_spec", return_value=None)
def test_evaluate_spacy_returns_none_when_missing(mock_spec):
    assert evaluate_spacy([]) is None


@patch("importlib.util.find_spec", return_value=None)
def test_evaluate_udpipe_returns_none_when_missing(mock_spec):
    assert evaluate_udpipe([]) is None


def test_competitor_result_dataclass():
    r = CompetitorResult(
        name="test",
        em=0.5,
        lemma_accuracy=0.6,
        ufeats_accuracy=0.7,
        pos_accuracy=0.8,
        tps=1000.0,
        model_size_mb=100.0,
        n=50,
    )
    assert r.name == "test"
    assert r.em == 0.5
    assert r.n == 50
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.em = 0.9  # type: ignore[misc]


def test_generate_positioning_summary_nonempty():
    r = CompetitorResult(
        name="mock_system",
        em=0.4,
        lemma_accuracy=0.5,
        ufeats_accuracy=0.6,
        pos_accuracy=0.7,
        tps=500.0,
        model_size_mb=100.0,
        n=10,
    )
    summary = generate_positioning_summary([r], our_em=84.45)
    assert len(summary) > 0
    assert "mock_system" in summary
    assert "kokturk" in summary


def test_generate_positioning_summary_empty_results():
    summary = generate_positioning_summary([])
    assert "Published baselines" in summary


def test_run_benchmark_writes_table(tmp_path):
    """Mock all evaluators to return known results and verify markdown output."""
    mock_result = CompetitorResult(
        name="mock",
        em=0.42,
        lemma_accuracy=0.55,
        ufeats_accuracy=0.60,
        pos_accuracy=0.70,
        tps=123.0,
        model_size_mb=50.0,
        n=5,
    )
    test_data = [
        {"surface": "evlerinden", "expected_root": "ev", "expected_tags": ["+Noun", "+PLU"]},
    ]
    out = tmp_path / "report.md"
    with (
        patch("benchmark.competitor_accuracy.evaluate_stanza", return_value=mock_result),
        patch("benchmark.competitor_accuracy.evaluate_spacy", return_value=None),
        patch("benchmark.competitor_accuracy.evaluate_udpipe", return_value=None),
    ):
        results = run_competitor_benchmark(test_data, out, our_em=84.45)
    assert len(results) == 1
    text = out.read_text(encoding="utf-8")
    assert "Competitor Accuracy Benchmark" in text
    assert "mock" in text
    assert "84.45%" in text
    assert "Morse" in text  # published baseline
