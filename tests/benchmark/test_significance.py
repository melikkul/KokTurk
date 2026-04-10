"""Tests for statistical significance testing."""

from __future__ import annotations

from benchmark.significance import holm_bonferroni_correction, paired_bootstrap_test


class TestPairedBootstrap:
    def test_identical_systems(self) -> None:
        """Two identical systems should have p ≈ 0.5 (not significant)."""
        preds = [0, 1, 1, 0, 1, 0, 0, 1] * 50
        labels = [0, 1, 1, 0, 1, 0, 1, 1] * 50
        result = paired_bootstrap_test(preds, preds, labels)
        assert result["p_value"] >= 0.3  # Not significant
        assert abs(result["mean_diff"]) < 0.01

    def test_better_system_significant(self) -> None:
        """A clearly better system should have low p-value."""
        labels = [0, 1, 0, 1, 0, 1] * 100
        preds_good = labels  # perfect
        preds_bad = [1, 0, 1, 0, 1, 0] * 100  # all wrong
        result = paired_bootstrap_test(preds_good, preds_bad, labels)
        assert result["p_value"] < 0.05
        assert result["mean_diff"] > 0.5

    def test_cohens_d_positive(self) -> None:
        labels = [0, 1] * 200
        preds_a = labels
        preds_b = [1 - x for x in labels]
        result = paired_bootstrap_test(preds_a, preds_b, labels)
        assert result["cohens_d"] > 0


class TestHolmBonferroni:
    def test_single_pvalue(self) -> None:
        assert holm_bonferroni_correction([0.03]) == [0.03]

    def test_correction_increases_pvalues(self) -> None:
        raw = [0.01, 0.04, 0.03]
        corrected = holm_bonferroni_correction(raw)
        # Corrected should be >= raw
        for r, c in zip(raw, corrected, strict=True):
            assert c >= r

    def test_capped_at_one(self) -> None:
        raw = [0.5, 0.6, 0.7]
        corrected = holm_bonferroni_correction(raw)
        assert all(c <= 1.0 for c in corrected)

    def test_empty(self) -> None:
        assert holm_bonferroni_correction([]) == []


def test_multi_system_significance_report(tmp_path):
    from benchmark.significance import multi_system_significance_report

    labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    sys_a = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    sys_b = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    sys_same = list(labels)

    out = tmp_path / "sig.md"
    result = multi_system_significance_report(
        [("diff", labels, sys_a, sys_b), ("same", labels, sys_a, sys_same)],
        output_path=out,
        n_bootstrap=200,
    )
    assert out.exists()
    assert "significant" in result
    assert result["significant"]["diff"] is True
