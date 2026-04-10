"""Tests for paradigm_augmentation.attach_suffix and augment_corpus."""

from __future__ import annotations

import json
from pathlib import Path

from data.paradigm_augmentation import (
    NOMINAL_PARADIGM,
    attach_suffix,
    augment_corpus,
    augmentation_budget,
    generate_paradigm,
    mine_voicing_alternations,
)


def test_attach_suffix_vowel_harmony_basic() -> None:
    assert attach_suffix("ev", "+lAr") == "evler"
    assert attach_suffix("göz", "+lAr") == "gözler"
    assert attach_suffix("kitap", "+DA") == "kitapta"
    assert attach_suffix("yol", "+(y)A") == "yola"


def test_attach_suffix_buffer_consonants() -> None:
    # Vowel-final root → buffer fires.
    assert attach_suffix("araba", "+(y)A") == "arabaya"
    # Consonant-final root → buffer suppressed.
    assert attach_suffix("ev", "+(y)A") == "eve"


def test_attach_suffix_voicing_alternation() -> None:
    altmap = {"kitap": "kitab"}
    # With alternation + buffer (y) + I archiphoneme: kitap → kitab + ı
    assert attach_suffix("kitap", "+(y)I", altmap) == "kitabı"


def test_attach_suffix_ben_dan() -> None:
    # ben + DAn → benden
    assert attach_suffix("ben", "+DAn") == "benden"


def test_attach_suffix_su_da() -> None:
    # su is vowel-final; D archiphoneme after vowel → d
    assert attach_suffix("su", "+DA") == "suda"


def test_mine_voicing_alternations(tmp_path: Path) -> None:
    corpus = tmp_path / "c.jsonl"
    corpus.write_text("\n".join([
        json.dumps({"label": "kitap +Noun"}),
        json.dumps({"label": "ev +Noun"}),
        json.dumps({"label": "ağaç +Noun"}),
    ]))
    altmap = mine_voicing_alternations(corpus)
    assert altmap["kitap"] == "kitab"
    assert altmap["ağaç"] == "ağac"
    assert "ev" not in altmap


def test_augmentation_budget_rarity_rule(tmp_path: Path) -> None:
    freq = tmp_path / "f.json"
    freq.write_text(json.dumps({
        "total_tag_occurrences": 1000,
        "unique_tags": 3,
        "tags": [
            {"tag": "+GEN", "count": 500, "percentage": 50.0,
             "cumulative_percentage": 50.0, "frequency_class": "HIGH_FREQ"},
            {"tag": "+WHILE", "count": 30, "percentage": 3.0,
             "cumulative_percentage": 53.0, "frequency_class": "MID_FREQ"},
            {"tag": "+JUSTLIKE", "count": 3, "percentage": 0.3,
             "cumulative_percentage": 53.3, "frequency_class": "LOW_FREQ"},
        ],
    }))
    budget = augmentation_budget(freq)
    assert budget["+GEN"] == 0       # >100 → 0
    assert budget["+WHILE"] == 20    # 10-100 → 20
    assert budget["+JUSTLIKE"] == 50  # <10 → 50


def test_generate_paradigm_nominal() -> None:
    forms = generate_paradigm("ev", "Noun")
    assert any(surface == "evler" for surface, _ in forms)
    # Morphotactic validity: no POSS suffix attached to a Verb paradigm
    labels = [lbl for _, lbl in forms]
    assert all("+Verb" not in lbl for lbl in labels)


def test_augment_corpus_writes_synthetic_tier(tmp_path: Path) -> None:
    # Budget so +PLU (rare count) gets allowance.
    freq = tmp_path / "f.json"
    freq.write_text(json.dumps({
        "total_tag_occurrences": 10, "unique_tags": 1,
        "tags": [
            {"tag": "+PLU", "count": 5, "percentage": 50.0,
             "cumulative_percentage": 50.0, "frequency_class": "LOW_FREQ"},
        ],
    }))
    out = tmp_path / "synth.jsonl"
    roots = [("ev", "Noun"), ("göz", "Noun")]
    report = augment_corpus(freq, out, roots)
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) > 0
    assert all(r["tier"] == "synthetic" for r in rows)
    assert report.kept == len(rows)


def test_augment_corpus_validator_discard(tmp_path: Path) -> None:
    freq = tmp_path / "f.json"
    freq.write_text(json.dumps({
        "total_tag_occurrences": 10, "unique_tags": 1,
        "tags": [
            {"tag": "+PLU", "count": 5, "percentage": 50.0,
             "cumulative_percentage": 50.0, "frequency_class": "LOW_FREQ"},
        ],
    }))
    out = tmp_path / "synth.jsonl"
    report = augment_corpus(
        freq, out, roots=[("ev", "Noun")],
        validator=lambda root, surface: False,  # reject everything
    )
    assert report.discarded_invalid > 0
    assert report.kept == 0
    assert out.read_text() == ""
