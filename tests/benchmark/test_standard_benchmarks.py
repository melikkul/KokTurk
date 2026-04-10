"""Tests for benchmark.standard_benchmarks."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.standard_benchmarks import (
    StandardExample,
    evaluate_disambiguation_mode,
    evaluate_generation_mode,
    load_trmorph2018,
    load_ud_test_split,
    run_standard_benchmarks,
    ud_to_canonical,
)


def test_ud_to_canonical_case_and_number():
    tags = ud_to_canonical("Case=Abl|Number=Plur", upos="NOUN")
    assert "+NOUN" in tags
    assert "+ABL" in tags
    assert "+PLU" in tags


def test_ud_to_canonical_possessive_joins():
    tags = ud_to_canonical("Case=Dat|Number[psor]=Sing|Person[psor]=3", upos="NOUN")
    assert "+POSS.3SG" in tags
    assert "+DAT" in tags


def test_ud_to_canonical_unknown_feat_skipped(caplog):
    with caplog.at_level("WARNING"):
        tags = ud_to_canonical("Foo=Bar", upos="NOUN")
    assert "+NOUN" in tags
    assert "Foo=Bar" not in tags


def test_load_trmorph2018(tmp_path: Path):
    f = tmp_path / "trmor.tsv"
    f.write_text("evler\tev+Noun+A3pl\nkitap\tkitap+Noun\n", encoding="utf-8")
    examples = load_trmorph2018(f)
    assert len(examples) == 2
    assert examples[0].surface == "evler"
    assert examples[0].lemma == "ev"


def test_load_trmorph2018_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_trmorph2018(tmp_path / "nope.tsv")


def test_load_ud_test_split(tmp_path: Path):
    conllu = (
        "# sent_id = 1\n"
        "1\tevler\tev\tNOUN\t_\tCase=Nom|Number=Plur\t0\troot\t_\t_\n"
        "2\tkitabı\tkitap\tNOUN\t_\tCase=Acc|Number=Sing\t1\tobj\t_\t_\n"
    )
    f = tmp_path / "test.conllu"
    f.write_text(conllu, encoding="utf-8")
    ex = load_ud_test_split(f)
    assert len(ex) == 2
    assert ex[0].lemma == "ev"
    assert "+PLU" in ex[0].gold_tags


def test_evaluate_generation_mode_perfect():
    examples = [
        StandardExample("evler", ("+NOUN", "+PLU"), "ev", False),
        StandardExample("kitabı", ("+NOUN", "+ACC"), "kitap", False),
    ]

    def predict(s):
        return {"evler": ("ev", ("+NOUN", "+PLU")), "kitabı": ("kitap", ("+NOUN", "+ACC"))}[s]

    m = evaluate_generation_mode(predict, examples)
    assert m.full_parse_em == 1.0
    assert m.lemma_accuracy == 1.0


def test_evaluate_disambiguation_mode_mock_zeyrek():
    examples = [StandardExample("evler", ("+NOUN", "+PLU"), "ev", False)]
    candidates = {"evler": [("ev", ("+NOUN", "+PLU")), ("evle", ("+VERB", "+PAST"))]}

    def candidate_fn(s):
        return candidates[s]

    def score(surface, cand):
        return 1.0 if cand[0] == "ev" else -1.0

    m = evaluate_disambiguation_mode(score, candidate_fn, examples)
    assert m.full_parse_em == 1.0


def test_run_standard_benchmarks_writes_report(tmp_path: Path):
    examples = [StandardExample("evler", ("+NOUN", "+PLU"), "ev", False)]

    def predict(s):
        return ("ev", ("+NOUN", "+PLU"))

    out = tmp_path / "std.md"
    report = run_standard_benchmarks(predict, {"tiny": examples}, out)
    assert out.exists()
    assert report.generation["tiny"].full_parse_em == 1.0
