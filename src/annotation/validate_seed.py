"""Validate gold-annotated seed data quality.

Checks completeness, format correctness, and agreement with Zeyrek predictions.

Usage:
    python src/annotation/validate_seed.py --input data/gold/seed/seed_200_partial.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def validate_seed(input_path: Path) -> dict[str, object]:
    """Validate seed annotation quality.

    Returns:
        Dict with validation results and statistics.
    """
    data: list[dict[str, object]] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    total = 0
    labeled = 0
    unlabeled = 0
    matches_top1 = 0
    disagreements: list[dict[str, str]] = []
    format_errors: list[str] = []

    for sent in data:
        for token in sent["tokens"]:  # type: ignore[union-attr]
            total += 1
            gold = token.get("gold_label")  # type: ignore[union-attr]

            if gold is None:
                unlabeled += 1
                continue

            labeled += 1

            # Check format: should be "root +TAG1 +TAG2 ..."
            if not isinstance(gold, str) or not gold.strip():
                format_errors.append(
                    f"{sent['sentence_id']}:{token['token_idx']}: "  # type: ignore[index]
                    f"empty gold_label"
                )
                continue

            parts = gold.split()
            root = parts[0]

            if not root:
                format_errors.append(
                    f"{sent['sentence_id']}:{token['token_idx']}: "  # type: ignore[index]
                    f"empty root in '{gold}'"
                )

            # Check if gold matches top-1 candidate
            parses = token.get("candidate_parses", [])  # type: ignore[union-attr]
            if parses:
                top1 = parses[0]
                top1_tags = " ".join(top1.get("tags", []))  # type: ignore[union-attr]
                top1_label = f"{top1['root']} {top1_tags}".strip()  # type: ignore[index]
                if gold.strip() == top1_label.strip():
                    matches_top1 += 1
                else:
                    disagreements.append({
                        "surface": str(token["surface"]),  # type: ignore[index]
                        "gold": gold,
                        "top1": top1_label,
                    })

    results: dict[str, object] = {
        "total": total,
        "labeled": labeled,
        "unlabeled": unlabeled,
        "completeness": labeled / max(total, 1),
        "matches_top1": matches_top1,
        "top1_agreement": matches_top1 / max(labeled, 1),
        "disagreements": len(disagreements),
        "format_errors": len(format_errors),
    }
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate seed annotation quality"
    )
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found")
        sys.exit(1)

    results = validate_seed(args.input)

    print(f"\n{'='*60}")
    print("SEED VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Total tokens:        {results['total']}")
    print(f"Labeled:             {results['labeled']} "
          f"({results['completeness']:.1%})")
    print(f"Unlabeled:           {results['unlabeled']}")
    print(f"Matches Zeyrek top1: {results['matches_top1']} "
          f"({results['top1_agreement']:.1%})")
    print(f"Disagreements:       {results['disagreements']}")
    print(f"Format errors:       {results['format_errors']}")

    ok = results["format_errors"] == 0
    print(f"\nStatus: {'PASS' if ok else 'ISSUES FOUND'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
