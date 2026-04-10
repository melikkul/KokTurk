"""Tests for the acquisition function and active learning components."""

from __future__ import annotations

from data.acquisition import MorphAcquisition, select_batch


class TestMAD:
    def test_single_parse_returns_zero(self) -> None:
        acq = MorphAcquisition()
        assert acq.compute_mad(1) == 0.0

    def test_zero_parses_returns_zero(self) -> None:
        acq = MorphAcquisition()
        assert acq.compute_mad(0) == 0.0

    def test_multiple_parses_positive(self) -> None:
        acq = MorphAcquisition(max_parse_count=10)
        score = acq.compute_mad(5)
        assert 0.0 < score < 1.0
        # log(5)/log(10) ≈ 0.699
        assert abs(score - 0.699) < 0.01

    def test_max_parses_returns_one(self) -> None:
        acq = MorphAcquisition(max_parse_count=10)
        assert abs(acq.compute_mad(10) - 1.0) < 0.001

    def test_more_parses_higher_score(self) -> None:
        acq = MorphAcquisition(max_parse_count=10)
        assert acq.compute_mad(2) < acq.compute_mad(5)
        assert acq.compute_mad(5) < acq.compute_mad(10)


class TestBALD:
    def test_returns_zero_without_model(self) -> None:
        """BALD returns 0.0 when no neural model is available."""
        acq = MorphAcquisition()
        assert acq.compute_bald({}) == 0.0


class TestCONF:
    def test_returns_zero_when_disabled(self) -> None:
        acq = MorphAcquisition(lambda_conf=0.0)
        assert acq.compute_conf([]) == 0.0

    def test_returns_zero_when_single_backend(self) -> None:
        acq = MorphAcquisition(lambda_conf=0.25)
        cands = [
            {"root": "ev", "tags": ["+Noun"], "source": "zeyrek"},
        ]
        assert acq.compute_conf(cands) == 0.0

    def test_agreement_returns_zero(self) -> None:
        acq = MorphAcquisition(lambda_conf=0.25)
        cands = [
            {"root": "ev", "tags": ["+Noun"], "source": "zeyrek"},
            {"root": "ev", "tags": ["+Noun"], "source": "trmorph"},
        ]
        assert acq.compute_conf(cands) == 0.0

    def test_disagreement_returns_one(self) -> None:
        acq = MorphAcquisition(lambda_conf=0.25)
        cands = [
            {"root": "ev", "tags": ["+Noun"], "source": "zeyrek"},
            {"root": "ev", "tags": ["+Verb"], "source": "trmorph"},
        ]
        assert acq.compute_conf(cands) == 1.0


class TestSentenceScoring:
    def test_empty_sentence_zero(self) -> None:
        acq = MorphAcquisition()
        assert acq.score_sentence([]) == 0.0

    def test_unambiguous_sentence_low(self) -> None:
        acq = MorphAcquisition()
        cands = [{"parse_count": 1, "analyses": []} for _ in range(5)]
        score = acq.score_sentence(cands)
        assert score == 0.0

    def test_ambiguous_sentence_higher(self) -> None:
        acq = MorphAcquisition(max_parse_count=10)
        unambiguous = [{"parse_count": 1, "analyses": []} for _ in range(5)]
        ambiguous = [{"parse_count": 5, "analyses": []} for _ in range(5)]
        s_unamb = acq.score_sentence(unambiguous)
        s_amb = acq.score_sentence(ambiguous)
        assert s_amb > s_unamb

    def test_mad_only_mode(self) -> None:
        """When BALD=0 and CONF=0, score is pure MAD."""
        acq = MorphAcquisition(
            lambda_bald=0.55, lambda_mad=0.45, lambda_conf=0.0
        )
        cands = [{"parse_count": 5, "analyses": []}]
        score = acq.score_sentence(cands)
        expected_mad = acq.compute_mad(5)
        assert abs(score - 0.45 * expected_mad) < 0.001


class TestBatchSelection:
    def test_selects_correct_count(self) -> None:
        corpus = [
            {"sentence_id": f"s{i}", "tokens": ["a", "b"]}
            for i in range(100)
        ]
        candidates_by_sent = {
            f"s{i}": [
                {"token_idx": 0, "parse_count": i % 5 + 1, "analyses": []},
                {"token_idx": 1, "parse_count": 1, "analyses": []},
            ]
            for i in range(100)
        }
        batch = select_batch(corpus, candidates_by_sent, batch_size=10)
        assert len(batch) == 10

    def test_excludes_annotated(self) -> None:
        corpus = [
            {"sentence_id": f"s{i}", "tokens": ["a"]}
            for i in range(10)
        ]
        candidates_by_sent = {
            f"s{i}": [{"token_idx": 0, "parse_count": 3, "analyses": []}]
            for i in range(10)
        }
        exclude = {f"s{i}" for i in range(5)}
        batch = select_batch(
            corpus, candidates_by_sent,
            exclude_ids=exclude, batch_size=10,
        )
        batch_ids = {sid for sid, _ in batch}
        assert not batch_ids & exclude

    def test_highest_scored_first(self) -> None:
        corpus = [
            {"sentence_id": "low", "tokens": ["a"]},
            {"sentence_id": "high", "tokens": ["b"]},
        ]
        candidates_by_sent = {
            "low": [{"token_idx": 0, "parse_count": 1, "analyses": []}],
            "high": [{"token_idx": 0, "parse_count": 8, "analyses": []}],
        }
        batch = select_batch(corpus, candidates_by_sent, batch_size=2)
        assert batch[0][0] == "high"
        assert batch[0][1] > batch[1][1]


class TestWeightedLabelModel:
    """Test that the upgraded label model produces valid outputs."""

    def test_confidence_in_range(self) -> None:
        import json
        from pathlib import Path

        labels_path = Path("data/weak_labels/probabilistic_labels.jsonl")
        if not labels_path.exists():
            return  # skip if data not generated

        with open(labels_path) as f:
            first_100 = [json.loads(next(f)) for _ in range(100)]

        for label in first_100:
            assert 0.0 <= label["confidence"] <= 1.0
            assert isinstance(label["entropy"], float)
            assert isinstance(label["needs_human_review"], bool)
