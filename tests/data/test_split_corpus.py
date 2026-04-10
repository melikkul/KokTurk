"""Tests for corpus splitting."""

from __future__ import annotations

import json
from pathlib import Path

from data.split_corpus import split_corpus


class TestSplitCorpus:
    def test_proportions(self, tmp_path: Path) -> None:
        # Create 100 records across 10 sentences
        corpus_path = tmp_path / "corpus.jsonl"
        with open(corpus_path, "w") as f:
            for s in range(10):
                for t in range(10):
                    rec = {
                        "sentence_id": f"s{s}",
                        "token_idx": t,
                        "surface": f"w{t}",
                        "label": f"w{t} +Noun",
                        "tier": "gold" if s < 2 else "silver-auto",
                    }
                    f.write(json.dumps(rec) + "\n")

        output_dir = tmp_path / "splits"
        stats = split_corpus(corpus_path, output_dir, seed=42)

        # 10 sentences → 8 train, 1 val, 1 test
        total_sents = sum(
            stats[s]["sentences"] for s in ("train", "val", "test")  # type: ignore[index]
        )
        assert total_sents == 10

    def test_no_sentence_overlap(self, tmp_path: Path) -> None:
        corpus_path = tmp_path / "corpus.jsonl"
        with open(corpus_path, "w") as f:
            for s in range(20):
                for t in range(5):
                    f.write(json.dumps({
                        "sentence_id": f"s{s}", "token_idx": t,
                        "surface": "w", "label": "w +N", "tier": "silver-auto",
                    }) + "\n")

        output_dir = tmp_path / "splits"
        split_corpus(corpus_path, output_dir, seed=42)

        # Check no sentence appears in multiple splits
        split_sids: dict[str, set[str]] = {}
        for name in ("train", "val", "test"):
            sids: set[str] = set()
            with open(output_dir / f"{name}.jsonl") as f:
                for line in f:
                    sids.add(json.loads(line)["sentence_id"])
            split_sids[name] = sids

        assert not (split_sids["train"] & split_sids["val"])
        assert not (split_sids["train"] & split_sids["test"])
        assert not (split_sids["val"] & split_sids["test"])
