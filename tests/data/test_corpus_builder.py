"""Tests for corpus builder."""

from __future__ import annotations

import json
from pathlib import Path

from data.corpus_builder import CorpusBuilder


class TestCorpusBuilder:
    def test_builds_three_tiers(self, tmp_path: Path) -> None:
        gold = [
            {"sentence_id": "s1", "text": "a b",
             "tokens": [
                 {"surface": "a", "token_idx": 0, "gold_label": "a +Noun",
                  "candidate_parses": []},
                 {"surface": "b", "token_idx": 1, "gold_label": "b +Verb",
                  "candidate_parses": []},
             ]}
        ]
        weak = [
            {"sentence_id": "s2", "token_idx": 0, "surface": "c",
             "predicted_label": "c +Adj", "confidence": 1.0},
            {"sentence_id": "s2", "token_idx": 1, "surface": "d",
             "predicted_label": "d +Noun +DAT", "confidence": 0.99},
            {"sentence_id": "s2", "token_idx": 2, "surface": "e",
             "predicted_label": "e +Verb", "confidence": 0.5},
        ]
        prelabeled = [
            {"sentence_id": "s2", "token_idx": 0, "parse_count": 1},
            {"sentence_id": "s2", "token_idx": 1, "parse_count": 3},
            {"sentence_id": "s2", "token_idx": 2, "parse_count": 2},
        ]

        gold_path = tmp_path / "gold.jsonl"
        weak_path = tmp_path / "weak.jsonl"
        pre_path = tmp_path / "pre.jsonl"
        out_path = tmp_path / "corpus.jsonl"

        for path, data in [(gold_path, gold), (weak_path, weak), (pre_path, prelabeled)]:
            with open(path, "w") as f:
                for r in data:
                    f.write(json.dumps(r) + "\n")

        cb = CorpusBuilder(
            gold_path=str(gold_path),
            weak_labels_path=str(weak_path),
            prelabeled_path=str(pre_path),
            output_path=str(out_path),
        )
        stats = cb.build()

        assert stats["gold"] == 2
        assert stats["silver_auto"] == 1  # c: parse_count=1, conf=1.0
        assert stats["silver_agreed"] == 1  # d: parse_count=3, conf=0.99
        assert stats["total"] == 4  # e excluded (conf=0.5)

    def test_gold_excluded_from_silver(self, tmp_path: Path) -> None:
        gold = [
            {"sentence_id": "s1", "text": "a",
             "tokens": [
                 {"surface": "a", "token_idx": 0, "gold_label": "a +Noun",
                  "candidate_parses": []},
             ]}
        ]
        weak = [
            {"sentence_id": "s1", "token_idx": 0, "surface": "a",
             "predicted_label": "a +Noun", "confidence": 1.0},
        ]
        prelabeled = [
            {"sentence_id": "s1", "token_idx": 0, "parse_count": 1},
        ]

        gold_path = tmp_path / "gold.jsonl"
        weak_path = tmp_path / "weak.jsonl"
        pre_path = tmp_path / "pre.jsonl"
        out_path = tmp_path / "corpus.jsonl"

        for path, data in [(gold_path, gold), (weak_path, weak), (pre_path, prelabeled)]:
            with open(path, "w") as f:
                for r in data:
                    f.write(json.dumps(r) + "\n")

        cb = CorpusBuilder(
            str(gold_path), str(weak_path), str(pre_path), str(out_path),
        )
        stats = cb.build()
        assert stats["gold"] == 1
        assert stats["silver_auto"] == 0  # excluded
        assert stats["total"] == 1
