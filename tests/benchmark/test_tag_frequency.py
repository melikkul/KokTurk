"""Tests for src.benchmark.tag_frequency."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.tag_frequency import (
    build_frequency_table,
    count_tags,
    extract_tags,
    format_markdown_table,
    resolve_corpus_path,
    write_json,
)


def test_extract_tags_basic() -> None:
    assert extract_tags("ev +Noun +PLU +POSS.3SG") == ["+Noun", "+PLU", "+POSS.3SG"]
    assert extract_tags("koş +Verb") == ["+Verb"]
    assert extract_tags("bare") == []


def test_count_tags_and_classes(tmp_path: Path) -> None:
    corpus = tmp_path / "gold.jsonl"
    lines = []
    # +A occurs 80%, +B 15%, +C 5% -> HIGH, HIGH, HIGH (all > 5%?)
    # We want to hit every class: use skewed distribution.
    for _ in range(900):
        lines.append(json.dumps({"label": "x +A"}))
    for _ in range(80):
        lines.append(json.dumps({"label": "x +B"}))
    for _ in range(15):
        lines.append(json.dumps({"label": "x +C"}))
    for _ in range(5):
        lines.append(json.dumps({"label": "x +D"}))
    corpus.write_text("\n".join(lines))

    counts = count_tags(corpus)
    stats = build_frequency_table(counts)

    by_tag = {s.tag: s for s in stats}
    assert by_tag["+A"].frequency_class == "HIGH_FREQ"
    assert by_tag["+B"].frequency_class == "HIGH_FREQ"
    assert by_tag["+C"].frequency_class == "MID_FREQ"
    assert by_tag["+D"].frequency_class == "LOW_FREQ"
    # Cumulative monotonic, reaches ~100
    cums = [s.cumulative_percentage for s in stats]
    assert cums == sorted(cums)
    assert cums[-1] == pytest.approx(100.0, abs=1e-6)


def test_resolve_corpus_path_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_corpus_path(tmp_path)


def test_resolve_corpus_path_picks_jsonl_first(tmp_path: Path) -> None:
    (tmp_path / "tr_gold_morph_v1.jsonl").write_text("")
    (tmp_path / "train.tsv").write_text("")
    assert resolve_corpus_path(tmp_path).name == "tr_gold_morph_v1.jsonl"


def test_write_json_roundtrip(tmp_path: Path) -> None:
    corpus = tmp_path / "c.jsonl"
    corpus.write_text(json.dumps({"label": "ev +Noun +PLU"}) + "\n")
    stats = build_frequency_table(count_tags(corpus))
    out = tmp_path / "sub" / "freq.json"
    write_json(stats, out)
    payload = json.loads(out.read_text())
    assert payload["unique_tags"] == 2
    assert payload["total_tag_occurrences"] == 2


def test_format_markdown_has_header() -> None:
    md = format_markdown_table([])
    assert "| Rank | Tag |" in md


def test_tsv_corpus_last_column(tmp_path: Path) -> None:
    corpus = tmp_path / "train.tsv"
    corpus.write_text("evlerinden\tev +Noun +PLU +POSS.3SG +ABL\n")
    counts = count_tags(corpus)
    assert counts["+Noun"] == 1
    assert counts["+PLU"] == 1
    assert counts["+ABL"] == 1
