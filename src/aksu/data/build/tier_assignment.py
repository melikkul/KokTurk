"""Assign tiers to autolabeled entries using the canonical tier policy.

Reads data/intermediate/autolabeled.jsonl, assigns tiers, writes
data/intermediate/tiered.jsonl.

Usage:
    python -m aksu.data.build.tier_assignment
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from aksu.data.tiers import Tier, assign_tier

logger = logging.getLogger(__name__)


def assign_tiers_to_file(
    input_path: Path,
    output_path: Path,
) -> dict[str, int]:
    """Read autolabeled.jsonl, add 'tier' field, write tiered.jsonl."""
    from collections import Counter
    counts: Counter[str] = Counter()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(encoding="utf-8") as inf, \
         output_path.open("w", encoding="utf-8") as outf:
        for line in inf:
            if not line.strip():
                continue
            row = json.loads(line)

            confidence = float(row.get("confidence", 0.0))
            unanimous = bool(row.get("unanimous", False))
            manually_verified = row.get("tier") == "gold"  # pre-existing gold annotation

            tier = assign_tier(
                confidence,
                ensemble_unanimous=unanimous,
                manually_verified=manually_verified,
            )
            row["tier"] = tier.value
            outf.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[tier.value] += 1

    return dict(counts)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/intermediate/autolabeled.jsonl")
    ap.add_argument("--output", default="data/intermediate/tiered.jsonl")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        ap.error(f"Input file not found: {input_path}")

    counts = assign_tiers_to_file(input_path, Path(args.output))
    logger.info("Tier distribution: %s", counts)
    total = sum(counts.values())
    for tier, n in sorted(counts.items()):
        logger.info("  %s: %d (%.1f%%)", tier, n, 100 * n / total if total else 0)


if __name__ == "__main__":
    main()
