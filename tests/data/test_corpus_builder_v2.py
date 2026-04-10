"""Tests for POS-specific thresholds in corpus_builder."""

from __future__ import annotations

import json
from pathlib import Path

from data.corpus_builder import POS_THRESHOLDS, CorpusBuilder, _extract_pos


class TestExtractPos:
    def test_noun(self) -> None:
        assert _extract_pos("ev +Noun +PLU") == "Noun"

    def test_verb(self) -> None:
        assert _extract_pos("git +Verb +PAST") == "Verb"

    def test_no_tag_returns_default(self) -> None:
        assert _extract_pos(".") == "default"

    def test_punctuation(self) -> None:
        assert _extract_pos(". +Punc") == "Punc"


class TestPosThresholds:
    def test_is_dict(self) -> None:
        assert isinstance(POS_THRESHOLDS, dict)

    def test_values_are_float_pairs(self) -> None:
        for pos, (auto, agreed) in POS_THRESHOLDS.items():
            assert isinstance(auto, float), f"{pos} auto threshold not float"
            assert isinstance(agreed, float), f"{pos} agreed threshold not float"
            assert 0.0 <= auto <= 1.0
            assert 0.0 <= agreed <= 1.0

    def test_has_default(self) -> None:
        assert "default" in POS_THRESHOLDS

    def test_verb_higher_than_noun(self) -> None:
        assert POS_THRESHOLDS["Verb"][0] > POS_THRESHOLDS["Noun"][0]


def _make_files(
    tmp_path: Path,
    gold: list[dict],
    weak: list[dict],
    prelabeled: list[dict],
) -> tuple[Path, Path, Path, Path]:
    """Write JSONL fixtures and return paths."""
    gold_path = tmp_path / "gold.jsonl"
    weak_path = tmp_path / "weak.jsonl"
    pre_path = tmp_path / "pre.jsonl"
    out_path = tmp_path / "output.jsonl"
    gold_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in gold) + "\n",
        encoding="utf-8",
    )
    weak_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in weak) + "\n",
        encoding="utf-8",
    )
    pre_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in prelabeled) + "\n",
        encoding="utf-8",
    )
    return gold_path, weak_path, pre_path, out_path


class TestCorpusBuilderPosThresholds:
    """Test that use_pos_thresholds flag controls threshold selection."""

    def _fixture(self) -> tuple[list, list, list]:
        gold = [
            {
                "sentence_id": "s1",
                "text": "a",
                "tokens": [
                    {"surface": "a", "token_idx": 0, "gold_label": "a +Noun"},
                ],
            }
        ]
        # Token with Verb label and confidence between global (0.95) and Verb (0.97)
        weak = [
            {
                "sentence_id": "s2",
                "token_idx": 0,
                "surface": "gel",
                "predicted_label": "gel +Verb +IMP",
                "confidence": 0.96,
            },
        ]
        prelabeled = [
            {"sentence_id": "s2", "token_idx": 0, "surface": "gel", "parse_count": 1},
        ]
        return gold, weak, prelabeled

    def test_pos_thresholds_reject_low_verb(self, tmp_path: Path) -> None:
        """With POS thresholds, 0.96 < 0.97 (Verb auto) => rejected."""
        gold, weak, prelabeled = self._fixture()
        gp, wp, pp, op = _make_files(tmp_path, gold, weak, prelabeled)
        cb = CorpusBuilder(
            gold_path=str(gp),
            weak_labels_path=str(wp),
            prelabeled_path=str(pp),
            output_path=str(op),
            use_pos_thresholds=True,
        )
        stats = cb.build()
        assert stats["gold"] == 1
        assert stats["silver_auto"] == 0

    def test_flat_thresholds_accept_verb(self, tmp_path: Path) -> None:
        """With flat thresholds (0.95), 0.96 >= 0.95 => accepted as silver-auto."""
        gold, weak, prelabeled = self._fixture()
        gp, wp, pp, op = _make_files(tmp_path, gold, weak, prelabeled)
        cb = CorpusBuilder(
            gold_path=str(gp),
            weak_labels_path=str(wp),
            prelabeled_path=str(pp),
            output_path=str(op),
            use_pos_thresholds=False,
        )
        stats = cb.build()
        assert stats["gold"] == 1
        assert stats["silver_auto"] == 1

    def test_gold_tokens_never_filtered(self, tmp_path: Path) -> None:
        """Gold tokens must always be included regardless of thresholds."""
        gold = [
            {
                "sentence_id": "s1",
                "text": "x",
                "tokens": [
                    {"surface": "x", "token_idx": 0, "gold_label": "x +Verb"},
                ],
            }
        ]
        # Same token appears in weak labels too — should be skipped (gold takes priority)
        weak = [
            {
                "sentence_id": "s1",
                "token_idx": 0,
                "surface": "x",
                "predicted_label": "x +Verb",
                "confidence": 0.50,
            },
        ]
        prelabeled = [
            {"sentence_id": "s1", "token_idx": 0, "surface": "x", "parse_count": 1},
        ]
        gp, wp, pp, op = _make_files(tmp_path, gold, weak, prelabeled)
        cb = CorpusBuilder(
            gold_path=str(gp),
            weak_labels_path=str(wp),
            prelabeled_path=str(pp),
            output_path=str(op),
            use_pos_thresholds=True,
        )
        stats = cb.build()
        assert stats["gold"] == 1
        assert stats["silver_auto"] == 0
        assert stats["total"] == 1
