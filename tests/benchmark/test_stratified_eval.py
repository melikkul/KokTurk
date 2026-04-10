"""Tests for stratified_eval and balanced_test_set."""

from __future__ import annotations

import json
from pathlib import Path

from benchmark.balanced_test_set import build_balanced_subset, load_balanced_indices
from benchmark.stratified_eval import (
    build_report,
    exact_match_score,
    format_report_markdown,
    morpheme_depth,
    per_tag_prf,
    stratify_by_depth,
    tag_f1_score,
    top20_confusion,
    write_report,
)


def test_morpheme_depth() -> None:
    assert morpheme_depth("ev +Noun") == 0
    assert morpheme_depth("ev +Noun +PLU") == 1
    assert morpheme_depth("ev +Noun +PLU +POSS.3SG +ABL") == 3


def test_exact_and_f1() -> None:
    preds = ["ev +Noun +PLU", "koş +Verb +PAST"]
    golds = ["ev +Noun +PLU", "koş +Verb +PAST"]
    assert exact_match_score(preds, golds) == 1.0
    assert tag_f1_score(preds, golds) == 1.0

    preds2 = ["ev +Noun +ABL", "koş +Verb +PAST"]
    assert exact_match_score(preds2, golds) == 0.5
    # Tag F1 still positive (partial overlap)
    assert 0 < tag_f1_score(preds2, golds) < 1.0


def test_per_tag_prf() -> None:
    preds = ["ev +Noun +PLU"]
    golds = ["ev +Noun +PLU"]
    prf = per_tag_prf(preds, golds)
    assert prf["+PLU"]["precision"] == 1.0
    assert prf["+PLU"]["f1"] == 1.0


def test_stratify_by_depth() -> None:
    preds = ["a +N", "a +N +PLU", "a +N +PLU +POSS.3SG +ABL +LOC +WITH"]
    golds = ["a +N", "a +N +PLU", "a +N +PLU +POSS.3SG +ABL +LOC +WITH"]
    strata = stratify_by_depth(preds, golds)
    names = {s.name for s in strata}
    assert "depth=0" in names
    assert "depth=1" in names
    # Cat B Task 4 split the deep tail; depth=5 is now its own bucket
    # (only depths ≥8 collapse into "8+").
    assert "depth=5" in names


def test_top20_confusion_aligned() -> None:
    preds = ["a +N +GEN"]
    golds = ["a +N +POSS.3SG"]
    mat = top20_confusion(preds, golds)
    # +POSS.3SG is top gold; predicted +GEN at same position
    assert "+POSS.3SG" in mat
    assert mat["+POSS.3SG"].get("+GEN") == 1


def test_build_report_with_balanced(tmp_path: Path) -> None:
    freq_json = tmp_path / "freq.json"
    freq_json.write_text(
        json.dumps(
            {
                "total_tag_occurrences": 4,
                "unique_tags": 2,
                "tags": [
                    {"tag": "+Noun", "count": 2, "percentage": 50.0,
                     "cumulative_percentage": 50.0, "frequency_class": "HIGH_FREQ"},
                    {"tag": "+PLU", "count": 2, "percentage": 50.0,
                     "cumulative_percentage": 100.0, "frequency_class": "HIGH_FREQ"},
                ],
            }
        )
    )
    preds = ["a +Noun +PLU", "b +Noun +PLU", "c +Noun"]
    golds = ["a +Noun +PLU", "b +Noun", "c +Noun"]
    pcs = [2, 1, 3]
    report = build_report(
        preds, golds, parse_counts=pcs, balanced_indices=[0, 2],
        tag_frequency_json=freq_json,
    )
    names = {s.name for s in report.full}
    assert "ALL" in names
    assert any("class=HIGH_FREQ" in n for n in names)
    assert len(report.balanced) > 0
    md = format_report_markdown(report)
    assert "# Stratified Evaluation Report" in md
    assert "Full Test Set" in md
    assert "Balanced Subset" in md

    out = tmp_path / "report.md"
    write_report(report, out)
    assert out.exists() and out.read_text().startswith("# Stratified")


def test_balanced_subset_builder(tmp_path: Path) -> None:
    test_path = tmp_path / "test.jsonl"
    lines = [
        json.dumps({"sentence_id": "s1", "token_idx": 0, "surface": "yüz", "label": "yüz +Noun"}),
        json.dumps({"sentence_id": "s1", "token_idx": 1, "surface": "ev", "label": "ev +Noun"}),
    ]
    test_path.write_text("\n".join(lines))

    def analyzer(surface: str) -> list[str]:
        # `yüz` is polysemous → 3 parses; `ev` is unambiguous → 1 parse.
        if surface == "yüz":
            return ["yüz +Noun", "yüz +Verb", "yüz +Num"]
        return ["ev +Noun"]

    out_path = tmp_path / "balanced.jsonl"
    n = build_balanced_subset(test_path, out_path, analyzer)
    assert n == 1  # only yüz kept
    indices = load_balanced_indices(out_path)
    assert indices == [("s1", 0)]
    payload = json.loads(out_path.read_text().strip())
    assert payload["gold_label"] == "yüz +Noun"
    assert len(payload["candidates"]) == 3
