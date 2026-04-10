"""Tests for benchmark.checklist_morpho."""

from __future__ import annotations

from benchmark.checklist_morpho import (
    generate_dir_tests,
    generate_inv_tests,
    generate_mft_tests,
    run_checklist,
)


def test_mft_covers_six_cases_and_six_persons():
    mft = generate_mft_tests()
    cases = [tc for tc in mft if tc.phenomenon == "case"]
    poss = [tc for tc in mft if tc.phenomenon == "possessive"]
    assert len(cases) == 6
    assert len(poss) == 6


def test_mft_includes_negation_and_harmony():
    mft = generate_mft_tests()
    phenomena = {tc.phenomenon for tc in mft}
    assert "negation" in phenomena
    assert "vowel_harmony" in phenomena


def test_inv_nonempty():
    assert len(generate_inv_tests()) >= 2


def test_dir_nonempty():
    assert len(generate_dir_tests()) >= 2


def test_run_checklist_collects_pass_fail(tmp_path):
    # Oracle that only knows a subset of cases.
    table = {tc.input: tc.expected_tags for tc in generate_mft_tests()[:3]}

    def predict(text):
        return table.get(text, ())

    out = tmp_path / "cl.md"
    report = run_checklist(predict, output_path=out)
    assert out.exists()
    # The three known cases should all pass.
    assert report.per_phenomenon["case"]["pass"] >= 3
