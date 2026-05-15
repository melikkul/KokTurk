"""Add morpheme boundary annotations to tiered dataset entries.

Usage:
    python -m aksu.data.build.boundaries --input data/intermediate/tiered.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _extract_boundary(surface: str, canonical: str) -> str | None:
    """Extract morpheme boundary string using BoundaryExtractor."""
    try:
        from aksu.ariturk.boundaries import BoundaryExtractor
        be = BoundaryExtractor()
        return be.extract(surface, canonical)
    except Exception:
        return None


def annotate_boundaries(
    input_path: Path,
    output_path: Path,
) -> dict[str, int]:
    """Add 'boundaries' field to each entry; returns coverage stats."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with_boundary = 0
    with input_path.open(encoding="utf-8") as inf, \
         output_path.open("w", encoding="utf-8") as outf:
        for line in inf:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            canonical = row.get("canonical")
            surface = row.get("token") or row.get("surface")
            if canonical and surface:
                boundary = _extract_boundary(surface, canonical)
                if boundary:
                    row["boundaries"] = boundary
                    with_boundary += 1
            outf.write(json.dumps(row, ensure_ascii=False) + "\n")

    coverage = with_boundary / total if total else 0.0
    return {"total": total, "with_boundary": with_boundary, "coverage": coverage}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default="data/intermediate/tiered.jsonl")
    ap.add_argument("--output", default="data/intermediate/with_boundaries.jsonl")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        ap.error(f"Input not found: {input_path}")

    stats = annotate_boundaries(input_path, Path(args.output))
    coverage_pct = stats["coverage"] * 100
    logger.info(
        "Boundary coverage: %.1f%% (%d / %d)",
        coverage_pct, stats["with_boundary"], stats["total"],
    )
    if coverage_pct < 95.6:
        logger.warning(
            "Coverage %.1f%% is below the README claim of 95.6%%. "
            "Update README to %.1f%%.",
            coverage_pct, coverage_pct,
        )


if __name__ == "__main__":
    main()
