"""Tests for auto-accept annotation tool."""

from __future__ import annotations

import json
from pathlib import Path

from annotation.auto_accept import auto_accept_seed


def _make_seed(tokens: list[dict[str, object]]) -> list[dict[str, object]]:
    return [{"sentence_id": "s1", "text": "test", "tokens": tokens}]


class TestAutoAccept:
    def test_unambiguous_high_conf_accepted(self, tmp_path: Path) -> None:
        tokens = [
            {"surface": "ev", "token_idx": 0, "gold_label": None,
             "candidate_parses": [
                 {"root": "ev", "tags": ["+Noun"], "confidence": 0.99}
             ]},
        ]
        inp = tmp_path / "input.jsonl"
        out = tmp_path / "output.jsonl"
        with open(inp, "w") as f:
            for s in _make_seed(tokens):
                f.write(json.dumps(s) + "\n")

        stats = auto_accept_seed(inp, out)
        assert stats["auto_accepted"] == 1
        assert stats["remaining"] == 0

        with open(out) as f:
            result = json.loads(f.readline())
        assert result["tokens"][0]["gold_label"] == "ev +Noun"

    def test_ambiguous_low_conf_not_accepted(self, tmp_path: Path) -> None:
        tokens = [
            {"surface": "yüz", "token_idx": 0, "gold_label": None,
             "candidate_parses": [
                 {"root": "yüz", "tags": ["+Noun"], "confidence": 0.5},
                 {"root": "yüz", "tags": ["+Verb"], "confidence": 0.5},
             ]},
        ]
        inp = tmp_path / "input.jsonl"
        out = tmp_path / "output.jsonl"
        with open(inp, "w") as f:
            for s in _make_seed(tokens):
                f.write(json.dumps(s) + "\n")

        stats = auto_accept_seed(inp, out)
        assert stats["auto_accepted"] == 0
        assert stats["remaining"] == 1

    def test_already_labeled_preserved(self, tmp_path: Path) -> None:
        tokens = [
            {"surface": "ev", "token_idx": 0, "gold_label": "ev +Noun",
             "candidate_parses": [
                 {"root": "ev", "tags": ["+Noun"], "confidence": 0.99}
             ]},
        ]
        inp = tmp_path / "input.jsonl"
        out = tmp_path / "output.jsonl"
        with open(inp, "w") as f:
            for s in _make_seed(tokens):
                f.write(json.dumps(s) + "\n")

        stats = auto_accept_seed(inp, out)
        assert stats["already_labeled"] == 1
        assert stats["auto_accepted"] == 0
