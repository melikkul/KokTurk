"""Tests for benchmark.error_analysis."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.error_analysis import (
    DIACRITIC_COST_MATRIX,
    RootErrorType,
    TagErrorType,
    classify_errors,
    compute_severity,
    generate_confusion_matrix,
    generate_error_report,
    levenshtein,
    oracle_projection,
)


def test_levenshtein_basic():
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("abc", "abd") == 1
    assert levenshtein("", "abc") == 3
    assert levenshtein("abc", "") == 3


def test_levenshtein_diacritic_aware():
    # Standard: ı -> i is 1 full substitution.
    assert levenshtein("kırıldı", "kirildi") == 3
    # Diacritic-aware: each diacritic pair costs 0.5.
    d = levenshtein("kırıldı", "kirildi", cost_matrix=DIACRITIC_COST_MATRIX)
    assert d == pytest.approx(1.5)


def test_classify_root_substitution():
    errs = classify_errors(["ev +PLU"], ["el +PLU"])
    assert len(errs) == 1
    assert errs[0].root_error is RootErrorType.SUBSTITUTION


def test_classify_root_truncation():
    errs = classify_errors(["kalem +PLU"], ["kale +PLU"])
    assert errs[0].root_error is RootErrorType.TRUNCATION


def test_classify_root_extension():
    errs = classify_errors(["kale +PLU"], ["kalem +PLU"])
    assert errs[0].root_error is RootErrorType.EXTENSION


def test_classify_root_hallucination():
    errs = classify_errors(["ev +PLU"], ["xyzqw +PLU"])
    assert errs[0].root_error is RootErrorType.HALLUCINATION


def test_classify_tag_missing_and_extra():
    errs = classify_errors(["ev +NOUN +PLU +LOC"], ["ev +NOUN +PLU"])
    assert errs[0].root_error is None
    types = {et for et, _ in errs[0].tag_errors}
    assert TagErrorType.MISSING in types


def test_classify_exact_match_skipped():
    errs = classify_errors(["ev +PLU"], ["ev +PLU"])
    assert errs == []


def test_severity_monotonic_root_worse_than_tag():
    # Root hallucination must outweigh a minor tag-order swap.
    root_err = classify_errors(["ev +NOUN +LOC"], ["xyz +NOUN +LOC"])[0]
    tag_err = classify_errors(["ev +NOUN +LOC +PLU"], ["ev +NOUN +PLU +LOC"])[0]
    assert root_err.severity > tag_err.severity


def test_compute_severity_hallucination_weights_1():
    s = compute_severity(RootErrorType.HALLUCINATION, [])
    assert s == 1.0


def test_oracle_projection_arithmetic():
    # 10 tokens total, 2 errors: both root substitutions.
    errs = classify_errors(
        ["evler +PLU", "eller +ACC"] + ["ok"] * 8,
        ["ivler +PLU", "aller +ACC"] + ["ok"] * 8,
    )
    proj = oracle_projection(errs, total_tokens=10)
    assert proj["baseline_em"] == pytest.approx(0.8)
    assert proj["root_substitution"] == pytest.approx(1.0)


def test_confusion_matrix_case_shape():
    errs = classify_errors(
        ["ev +NOUN +ACC"], ["ev +NOUN +LOC"],
    )
    cm = generate_confusion_matrix(errs)
    assert "CASE" in cm
    assert "+ACC" in cm["CASE"]
    assert "+LOC" in cm["CASE"]["+ACC"]


def test_generate_error_report_writes_files(tmp_path: Path):
    gold = tmp_path / "gold.txt"
    pred = tmp_path / "pred.txt"
    gold.write_text("ev +PLU\nel +ACC\nokul +LOC\n", encoding="utf-8")
    pred.write_text("ev +PLU\nab +ACC\nokul +LOC\n", encoding="utf-8")
    out = tmp_path / "report.md"
    report = generate_error_report(gold, pred, out)
    assert out.exists()
    assert out.with_suffix(".json").exists()
    assert report.total == 3
    assert len(report.errors) == 1
