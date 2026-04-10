"""Tests for silver_correction and noise_audit sampling."""

from __future__ import annotations

import json
from pathlib import Path

from data.noise_audit import generate_human_audit_sample
from data.silver_correction import (
    apply_heuristic_corrections,
    apply_rules_to_record,
    build_adj_gazetteer,
    gold_sanity_check,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))


def test_rule1_poss_on_verb_forces_noun() -> None:
    rec = {"surface": "gelmesi", "label": "gel +Verb +POSS.3SG", "tier": "silver-auto"}
    new, rule = apply_rules_to_record(rec, None, adj_gazetteer=set())
    assert rule == "rule1_poss_on_verb"
    assert "+Noun" in new and "+Verb" not in new


def test_rule2_adj_gazetteer_next_noun() -> None:
    rec = {"surface": "kırmızı", "label": "kırmızı +Noun", "tier": "silver-auto"}
    nxt = {"surface": "kapı", "label": "kapı +Noun", "tier": "silver-auto"}
    new, rule = apply_rules_to_record(rec, nxt, adj_gazetteer={"kırmızı"})
    assert rule == "rule2_adj_gazetteer"
    assert "+Adj" in new


def test_rule3_flags_only_no_mutation() -> None:
    rec = {"surface": "arabanın", "label": "araba +Noun +GEN", "tier": "silver-auto"}
    window = [
        {"label": "kırmızı +Adj"},
        {"label": "büyük +Adj"},
        {"label": "ev +Noun"},  # No POSS within 3 — flag
    ]
    new, rule = apply_rules_to_record(rec, window[0], adj_gazetteer=set(), window=window)
    assert rule == "rule3_gen_without_poss_flag"
    assert new == rec["label"]  # no mutation


def test_apply_corrections_skips_gold(tmp_path: Path) -> None:
    corpus = tmp_path / "c.jsonl"
    _write_jsonl(
        corpus,
        [
            {"sentence_id": "s1", "surface": "gelmesi", "label": "gel +Verb +POSS.3SG", "tier": "gold"},
            {"sentence_id": "s1", "surface": "gelmesi", "label": "gel +Verb +POSS.3SG", "tier": "silver-auto"},
        ],
    )
    out = tmp_path / "out.jsonl"
    report = apply_heuristic_corrections(corpus, out, adj_gazetteer=set())
    assert report.total_silver == 1
    assert report.total_corrected == 1

    rows = [json.loads(l) for l in out.read_text().splitlines()]
    # Gold untouched
    assert rows[0]["label"] == "gel +Verb +POSS.3SG"
    # Silver corrected
    assert "+Noun" in rows[1]["label"]
    assert rows[1]["_corrected_by"] == "rule1_poss_on_verb"


def test_gold_sanity_check_high_agreement(tmp_path: Path) -> None:
    corpus = tmp_path / "c.jsonl"
    _write_jsonl(
        corpus,
        [
            {"sentence_id": "s1", "surface": "ev", "label": "ev +Noun", "tier": "gold"},
            {"sentence_id": "s1", "surface": "koş", "label": "koş +Verb", "tier": "gold"},
        ],
    )
    agreement = gold_sanity_check(corpus, adj_gazetteer=set())
    assert agreement == 1.0  # no rule fires → perfect agreement


def test_build_adj_gazetteer_from_fallback(tmp_path: Path) -> None:
    gold = tmp_path / "gold.jsonl"
    _write_jsonl(
        gold,
        [
            {"surface": "kırmızı", "label": "kırmızı +Adj"},
            {"surface": "ev", "label": "ev +Noun"},
        ],
    )
    adjs = build_adj_gazetteer(boun_dir=None, gold_fallback=gold)
    assert "kırmızı" in adjs
    assert "ev" not in adjs


def test_generate_human_audit_sample(tmp_path: Path) -> None:
    corpus = tmp_path / "c.jsonl"
    rows = []
    # 10 silver-auto, 3 silver-agreed, 2 gold (gold must be excluded)
    for i in range(10):
        rows.append({"sentence_id": "s1", "token_idx": i, "surface": f"w{i}",
                     "label": f"r{i} +Noun", "tier": "silver-auto"})
    for i in range(3):
        rows.append({"sentence_id": "s2", "token_idx": i, "surface": f"w{i}",
                     "label": f"r{i} +Noun", "tier": "silver-agreed"})
    for i in range(2):
        rows.append({"sentence_id": "s3", "token_idx": i, "surface": f"w{i}",
                     "label": f"r{i} +Noun", "tier": "gold"})
    _write_jsonl(corpus, rows)

    out = tmp_path / "audit.tsv"
    n = generate_human_audit_sample(corpus, out, n=10)
    assert n <= 10
    lines = out.read_text().splitlines()
    assert lines[0].startswith("token_idx\tsentence_id")
    # Make sure no gold-tier row leaked
    assert "s3" not in "\n".join(lines[1:])
