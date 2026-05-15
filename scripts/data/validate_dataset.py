"""D-Step 7: Validate TR-Gold-Morph dataset and emit manifest.json + markdown report.

Usage:
    python scripts/data/validate_dataset.py \
        --gold data/tr_gold_morph/gold.jsonl \
        --silver data/tr_gold_morph/silver.jsonl \
        --bronze data/tr_gold_morph/bronze.jsonl \
        --output data/tr_gold_morph/manifest.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for l in path.open(encoding="utf-8") if l.strip())


def _compute_stats(path: Path) -> dict:
    if not path.exists():
        return {}
    pos_counts: Counter = Counter()
    total = 0
    with_boundary = 0
    ambiguous = 0
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        pos_counts[row.get("pos", "UNKNOWN")] += 1
        total += 1
        if row.get("boundaries"):
            with_boundary += 1
        if len(row.get("candidates", [])) > 1:
            ambiguous += 1
    return {
        "total": total,
        "boundary_coverage": with_boundary / total if total else 0.0,
        "ambiguity_rate": ambiguous / total if total else 0.0,
        "pos_distribution": dict(pos_counts.most_common(10)),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold",   default="data/tr_gold_morph/gold.jsonl")
    ap.add_argument("--silver", default="data/tr_gold_morph/silver.jsonl")
    ap.add_argument("--bronze", default="data/tr_gold_morph/bronze.jsonl")
    ap.add_argument("--output", default="data/tr_gold_morph/manifest.json")
    args = ap.parse_args()

    gold_path   = Path(args.gold)
    silver_path = Path(args.silver)
    bronze_path = Path(args.bronze)

    gold_n   = _count_lines(gold_path)
    silver_n = _count_lines(silver_path)
    bronze_n = _count_lines(bronze_path)
    total_n  = gold_n + silver_n + bronze_n

    logger.info("Entries: gold=%d silver=%d bronze=%d total=%d", gold_n, silver_n, bronze_n, total_n)

    if total_n < 2_500_000:
        logger.warning(
            "Dataset has %d entries < 2,500,000 target. "
            "Extend harvest with additional corpora per D-Step 1 size policy.",
            total_n,
        )

    gold_stats = _compute_stats(gold_path)
    silver_stats = _compute_stats(silver_path)

    manifest = {
        "version": "v2",
        "entries": {"gold": gold_n, "silver": silver_n, "bronze": bronze_n, "total": total_n},
        "gold_stats": gold_stats,
        "silver_stats": silver_stats,
        "boundary_coverage": gold_stats.get("boundary_coverage", 0.0),
        "target_entries": 2_500_000,
        "target_met": total_n >= 2_500_000,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Manifest written to %s", output_path)

    # Markdown report
    report_path = output_path.with_suffix(".md")
    lines = [
        "# TR-Gold-Morph v2 Validation Report\n",
        f"| Tier | Entries |",
        f"|------|---------|",
        f"| Gold   | {gold_n:,} |",
        f"| Silver | {silver_n:,} |",
        f"| Bronze | {bronze_n:,} |",
        f"| **Total** | **{total_n:,}** |",
        "",
        f"Target: 2,500,000 — {'MET ✅' if total_n >= 2_500_000 else 'NOT MET ❌'}",
        "",
        f"Boundary coverage (gold): {gold_stats.get('boundary_coverage', 0)*100:.1f}%",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Report written to %s", report_path)


if __name__ == "__main__":
    main()
