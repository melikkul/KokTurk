"""Tests for benchmark.robustness."""

from __future__ import annotations

from benchmark.robustness import (
    CasingAttack,
    CodeSwitchAttack,
    DeasciificationAttack,
    ElongationAttack,
    HashtagAttack,
    run_robustness_suite,
)


def test_deasciification_deterministic_with_seed():
    a = DeasciificationAttack(prob=1.0, seed=42)
    assert a.perturb("güzel şarkı") == "guzel sarki"


def test_deasciification_respects_prob_zero():
    a = DeasciificationAttack(prob=0.0, seed=0)
    assert a.perturb("güzel") == "güzel"


def test_casing_turkish_i_rules():
    # Python str.lower() would map "İ"->"i̇" with combining dot.
    # Turkish-correct: "İ" -> "i", "I" -> "ı".
    assert CasingAttack().perturb("İSTANBUL", "all_lower") == "istanbul"
    assert CasingAttack().perturb("Irak", "all_lower") == "ırak"
    assert CasingAttack().perturb("istanbul", "all_caps") == "İSTANBUL"


def test_casing_mode_unknown_raises():
    import pytest
    with pytest.raises(ValueError):
        CasingAttack().perturb("ev", "bogus")


def test_elongation_lengthens_vowels_and_appends_punct():
    out = ElongationAttack().perturb("güzel", vowel_repeat=3)
    assert out.endswith("!!!")
    assert "üüü" in out


def test_code_switch_replaces_known_roots():
    out = CodeSwitchAttack().perturb("beğendim")
    assert "like'la" in out


def test_hashtag_fuses_words():
    assert HashtagAttack().perturb(["evde", "kal", "türkiye"]) == "#EvdeKalTürkiye"


def test_run_suite_records_all_attacks(tmp_path):
    def score(texts, labels):
        return sum(1 for t, l in zip(texts, labels) if t == l) / len(texts)

    texts = ["ev", "güzel", "beğendim"]
    labels = ["ev", "güzel", "beğendim"]
    out = tmp_path / "robust.md"
    report = run_robustness_suite(score, texts, labels, output_path=out)
    names = {r.name for r in report.results}
    assert {"deasciification", "casing_upper", "casing_lower", "elongation", "code_switch"} <= names
    assert out.exists()
