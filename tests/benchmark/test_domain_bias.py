"""Tests for benchmark.domain_bias."""

from __future__ import annotations

import pytest

from benchmark.domain_bias import (
    DomainBiasReport,
    classify_domain,
    generate_bias_summary,
    measure_domain_bias,
)


# ------------------------------------------------------------------
# classify_domain tests
# ------------------------------------------------------------------


class TestClassifyDomain:
    def test_boun_source_is_formal(self):
        assert classify_domain("test", source="boun_treebank") == "formal"

    def test_imst_source_is_formal(self):
        assert classify_domain("test", source="imst_treebank") == "formal"

    def test_bounti_source_is_social_media(self):
        assert classify_domain("test", source="bounti") == "social_media"

    def test_tweet_source_is_social_media(self):
        assert classify_domain("test", source="tweet_corpus") == "social_media"

    def test_trendyol_source_is_informal(self):
        assert classify_domain("test", source="trendyol_reviews") == "informal"

    def test_ttc_source_is_news(self):
        assert classify_domain("test", source="ttc3600") == "news"

    def test_hashtag_text_is_social_media(self):
        assert classify_domain("#istanbul harika") == "social_media"

    def test_mention_text_is_social_media(self):
        assert classify_domain("@user merhaba") == "social_media"

    def test_default_is_formal(self):
        assert classify_domain("ev") == "formal"


# ------------------------------------------------------------------
# measure_domain_bias tests
# ------------------------------------------------------------------


def _make_items(domain_source: str, n: int, pred_match: bool) -> list[dict[str, str]]:
    """Helper to create test data items for a given domain."""
    items = []
    for i in range(n):
        label = f"root{i} +Noun +NOM"
        pred = label if pred_match else f"wrong{i} +Verb"
        items.append({
            "surface": f"word{i}",
            "label": label,
            "prediction": pred,
            "source": domain_source,
        })
    return items


class TestMeasureDomainBias:
    def test_perfect_predictions_dpd_zero(self):
        data = _make_items("boun", 15, pred_match=True) + \
               _make_items("bounti", 15, pred_match=True)
        report = measure_domain_bias(data)
        assert report.dpd == pytest.approx(0.0)

    def test_dpd_computation(self):
        # formal: 100% EM, social_media: 0% EM
        data = _make_items("boun", 20, pred_match=True) + \
               _make_items("bounti", 20, pred_match=False)
        report = measure_domain_bias(data)
        assert report.dpd == pytest.approx(1.0)

    def test_tpr_disparity_sums_near_zero(self):
        data = _make_items("boun", 20, pred_match=True) + \
               _make_items("bounti", 20, pred_match=False)
        report = measure_domain_bias(data)
        total = sum(report.tpr_disparity.values())
        assert total == pytest.approx(0.0, abs=1e-9)

    def test_small_domain_excluded_from_dpd(self):
        # Large domain: 100% EM, small domain (n<10): 0% EM
        data = _make_items("boun", 20, pred_match=True) + \
               _make_items("bounti", 5, pred_match=False)
        report = measure_domain_bias(data)
        # Only one eligible domain → DPD should be 0
        assert report.dpd == pytest.approx(0.0)

    def test_single_domain_dpd_zero(self):
        data = _make_items("boun", 20, pred_match=True)
        report = measure_domain_bias(data)
        assert report.dpd == pytest.approx(0.0)

    def test_predictions_override(self):
        data = _make_items("boun", 15, pred_match=False)
        # Override with correct predictions
        preds = [item["label"] for item in data]
        report = measure_domain_bias(data, predictions=preds)
        assert report.domain_results["formal"]["em"] == pytest.approx(1.0)

    def test_domain_results_contain_n(self):
        data = _make_items("boun", 12, pred_match=True)
        report = measure_domain_bias(data)
        assert report.domain_results["formal"]["n"] == 12.0


# ------------------------------------------------------------------
# generate_bias_summary tests
# ------------------------------------------------------------------


class TestGenerateBiasSummary:
    def test_returns_string(self):
        data = _make_items("boun", 20, pred_match=True)
        report = measure_domain_bias(data)
        summary = generate_bias_summary(report)
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_mentions_dpd(self):
        data = _make_items("boun", 20, pred_match=True) + \
               _make_items("bounti", 20, pred_match=False)
        report = measure_domain_bias(data)
        summary = generate_bias_summary(report)
        assert "DPD" in summary or "dpd" in summary

    def test_insufficient_data(self):
        report = DomainBiasReport(
            domain_results={"formal": {"em": 1.0, "root_acc": 1.0, "tag_f1": 1.0, "n": 3.0}},
            dpd=0.0,
            tpr_disparity={},
        )
        summary = generate_bias_summary(report)
        assert "Insufficient" in summary


# ------------------------------------------------------------------
# Frozen dataclass tests
# ------------------------------------------------------------------


class TestFrozenDataclass:
    def test_domain_bias_report_is_frozen(self):
        report = DomainBiasReport(
            domain_results={}, dpd=0.0, tpr_disparity={}
        )
        with pytest.raises(AttributeError):
            report.dpd = 0.5  # type: ignore[misc]
