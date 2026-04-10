"""Build a morphologically balanced subset of the test split.

Strategy (simplified per the approved plan):
- For each test token, run the morphological analyzer to enumerate all
  candidate parses. The gold canonical label is the positive; every other
  candidate is a negative.
- Tokens with exactly one parse (unambiguous) are **skipped** — they add no
  disambiguation signal.
- We do NOT invent negatives from scratch. The subset is therefore strictly
  smaller than the full test split; callers MUST continue to report full-set
  metrics alongside balanced-set metrics so the v2 baseline (82% EM on the
  full set) remains comparable.

Output is JSONL with fields::

    {
        "sentence_id": ...,
        "token_idx": ...,
        "surface": ...,
        "gold_label": "...",
        "candidates": ["cand_1", "cand_2", ...],  # gold + negatives
        "num_parses": N,
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Protocol


class ParseProvider(Protocol):
    """Minimal interface for a Zeyrek-like analyzer (injectable for tests)."""

    def __call__(self, surface: str) -> list[str]:  # list of canonical labels
        ...


def _iter_test_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_balanced_subset(
    test_path: Path | str,
    output_path: Path | str,
    analyzer: ParseProvider,
) -> int:
    """Write the balanced subset. Returns number of records written."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as out:
        for rec in _iter_test_jsonl(Path(test_path)):
            surface = rec.get("surface", "")
            gold = rec.get("label", "")
            if not surface or not gold:
                continue
            candidates = analyzer(surface)
            # Normalize: ensure gold is present in the candidate set.
            if gold not in candidates:
                candidates = [gold, *candidates]
            if len(candidates) < 2:
                continue  # unambiguous
            out.write(
                json.dumps(
                    {
                        "sentence_id": rec.get("sentence_id"),
                        "token_idx": rec.get("token_idx"),
                        "surface": surface,
                        "gold_label": gold,
                        "candidates": candidates,
                        "num_parses": len(candidates),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
    return written


def load_balanced_indices(balanced_path: Path | str) -> list[tuple[str | None, int | None]]:
    """Return ``[(sentence_id, token_idx), ...]`` — useful for indexing into
    the full test set when cross-referencing metrics."""
    out: list[tuple[str | None, int | None]] = []
    for rec in _iter_test_jsonl(Path(balanced_path)):
        out.append((rec.get("sentence_id"), rec.get("token_idx")))
    return out
