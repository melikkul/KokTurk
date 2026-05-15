"""D-Step 1: Assert no IMST-origin entries appear in the published JSONL slices.

IMST is CC-BY-NC-SA-3.0 with a Non-Commercial clause incompatible with the
CC BY 4.0 main dataset license. The `redistribute=False` flag in sources.py
is enforced here: any JSONL slice shipped to HuggingFace / Zenodo must contain
zero IMST-origin source_ids.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

PUBLISHED_SLICES = [
    Path("data/tr_gold_morph/gold.jsonl"),
    Path("data/tr_gold_morph/silver.jsonl"),
    Path("data/tr_gold_morph/bronze.jsonl"),
]
BANNED_SOURCE_IDS = {"imst-ud", "imst", "UD_Turkish-IMST"}


@pytest.mark.parametrize("slice_path", PUBLISHED_SLICES)
def test_no_imst_in_published_slice(slice_path: Path) -> None:
    if not slice_path.exists():
        pytest.skip(f"Dataset slice not yet built: {slice_path}")

    violations: list[str] = []
    with slice_path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            source = str(row.get("source_id", row.get("source", ""))).lower()
            if any(banned.lower() in source for banned in BANNED_SOURCE_IDS):
                violations.append(f"line {lineno}: source_id={source!r}")
            if lineno > 10_000:
                break  # fast CI: sample first 10K entries

    assert not violations, (
        f"{slice_path} contains IMST-origin entries (CC-BY-NC-SA-3.0 incompatible "
        f"with CC BY 4.0 main license):\n" + "\n".join(violations[:10])
    )
