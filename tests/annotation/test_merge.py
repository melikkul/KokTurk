"""Tests for annotation merge tool."""

from __future__ import annotations

import json
from pathlib import Path

from annotation.merge_annotations import merge_annotations


class TestMerge:
    def test_seed_only(self, tmp_path: Path) -> None:
        seed = [
            {"sentence_id": "s1", "text": "a b",
             "tokens": [
                 {"surface": "a", "gold_label": "a +Noun"},
                 {"surface": "b", "gold_label": "b +Verb"},
             ]},
        ]
        seed_path = tmp_path / "seed.jsonl"
        out_path = tmp_path / "combined.jsonl"
        with open(seed_path, "w") as f:
            for s in seed:
                f.write(json.dumps(s) + "\n")

        stats = merge_annotations(seed_path, [], out_path)
        assert stats["sentences"] == 1
        assert stats["labeled_tokens"] == 2
        assert out_path.exists()

    def test_seed_plus_batch(self, tmp_path: Path) -> None:
        seed = [
            {"sentence_id": "s1", "text": "a",
             "tokens": [{"surface": "a", "gold_label": "a +Noun"}]},
        ]
        batch = [
            {"sentence_id": "s2", "text": "b",
             "tokens": [{"surface": "b", "gold_label": "b +Verb"}]},
        ]

        seed_path = tmp_path / "seed.jsonl"
        batch_path = tmp_path / "batch_001.jsonl"
        out_path = tmp_path / "combined.jsonl"

        for path, data in [(seed_path, seed), (batch_path, batch)]:
            with open(path, "w") as f:
                for s in data:
                    f.write(json.dumps(s) + "\n")

        stats = merge_annotations(seed_path, [batch_path], out_path)
        assert stats["sentences"] == 2
        assert stats["labeled_tokens"] == 2

    def test_deduplicates_by_sentence_id(self, tmp_path: Path) -> None:
        data = [
            {"sentence_id": "s1", "text": "a",
             "tokens": [{"surface": "a", "gold_label": "a +Noun"}]},
        ]
        seed_path = tmp_path / "seed.jsonl"
        batch_path = tmp_path / "batch.jsonl"
        out_path = tmp_path / "combined.jsonl"

        # Same sentence in both
        for path in [seed_path, batch_path]:
            with open(path, "w") as f:
                for s in data:
                    f.write(json.dumps(s) + "\n")

        stats = merge_annotations(seed_path, [batch_path], out_path)
        assert stats["sentences"] == 1  # No duplicate
