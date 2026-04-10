"""Tests for DomainAwareCurriculum + domain importers (smoke)."""

from __future__ import annotations

from pathlib import Path

from resource.importers.bounti import import_bounti
from resource.importers.trendyol import clean_review, import_trendyol
from train.domain_curriculum import DomainAwareCurriculum


def test_trendyol_clean_review_strips_noise() -> None:
    dirty = "Harika ürün! ⭐⭐⭐⭐⭐ 5 yıldız https://example.com &amp; XS bedenim"
    out = clean_review(dirty)
    assert "⭐" not in out
    assert "http" not in out
    assert "&amp" not in out
    assert "XS" not in out


def test_trendyol_skips_short_reviews(tmp_path: Path) -> None:
    csv = tmp_path / "t.csv"
    csv.write_text(
        "id,review\n"
        "1,👍\n"
        "2,Çok güzel bir ürün aldım teşekkürler\n"
        "3,.\n"
    )
    n = import_trendyol(csv, db=None, min_words=3)
    # Only row 2 should contribute
    assert n >= 5


def test_bounti_missing_path_returns_zero(tmp_path: Path) -> None:
    out = import_bounti(tmp_path / "nope", db=None, try_clone=False)
    assert out == 0


def test_bounti_reads_local_txt(tmp_path: Path) -> None:
    d = tmp_path / "bounti_local"
    d.mkdir()
    (d / "tweets.txt").write_text(
        "Günaydın İstanbul\nHarika bir gün bugün\n",
    )
    n = import_bounti(d, db=None, try_clone=False)
    assert n >= 2


def test_domain_curriculum_initial_state() -> None:
    curr = DomainAwareCurriculum()
    f = curr.current_filter
    assert f.domain_phase == "gold_only"
    assert f.allowed_tiers == {"gold"}


def test_domain_curriculum_manual_advance() -> None:
    curr = DomainAwareCurriculum()
    for expected in [
        "silver_news", "multi_domain_bronze", "all_tiers", "gold_calibration",
    ]:
        curr.advance_domain()
        assert curr.current_domain_phase == expected
    # Saturates at last phase
    curr.advance_domain()
    assert curr.current_domain_phase == "gold_calibration"


def test_domain_curriculum_delegates_to_taac() -> None:
    curr = DomainAwareCurriculum()
    # First few steps should not crash and should return a filter.
    out = curr.step(val_loss=1.0)
    assert "taac" in out
    assert "domain_phase" in out
    assert out["domain_phase"] == "gold_only"
