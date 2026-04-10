"""Auto-accept high-confidence morphological labels to reduce annotation burden.

For unambiguous tokens or very-high-confidence predictions, automatically
sets the gold_label without human review. This typically reduces manual
annotation by 50-65%.

Usage:
    python src/annotation/auto_accept.py \
        --input data/gold/seed/seed_200.jsonl \
        --output data/gold/seed/seed_200_partial.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path


def auto_accept_seed(
    input_path: Path,
    output_path: Path,
    unambig_threshold: float = 0.95,
    ambig_threshold: float = 0.98,
) -> dict[str, int]:
    """Pre-fill gold labels for high-confidence tokens.

    Rules:
    - parse_count == 1 AND confidence >= unambig_threshold → auto-accept
    - parse_count > 1 AND top confidence >= ambig_threshold → auto-accept
    - Otherwise → leave gold_label as null

    Args:
        input_path: Seed JSONL file.
        output_path: Output JSONL file with partial gold labels.
        unambig_threshold: Confidence threshold for unambiguous tokens.
        ambig_threshold: Confidence threshold for ambiguous tokens.

    Returns:
        Dict with auto_accepted, remaining, total counts.
    """
    data: list[dict[str, object]] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    total = 0
    auto_accepted = 0
    already_labeled = 0

    for sent in data:
        for token in sent["tokens"]:  # type: ignore[union-attr]
            total += 1

            if token["gold_label"] is not None:  # type: ignore[index]
                already_labeled += 1
                continue

            parses = token.get("candidate_parses", [])  # type: ignore[union-attr]
            if not parses:
                continue

            n_parses = len(parses)
            top_conf = parses[0].get("confidence", 0.0)  # type: ignore[union-attr]

            accept = (
                (n_parses == 1 and top_conf >= unambig_threshold)
                or (n_parses > 1 and top_conf >= ambig_threshold)
            )

            if accept:
                top = parses[0]
                tags = " ".join(top.get("tags", []))  # type: ignore[union-attr]
                label = f"{top['root']} {tags}".strip()  # type: ignore[index]
                token["gold_label"] = label  # type: ignore[index]
                auto_accepted += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    remaining = total - auto_accepted - already_labeled
    stats = {
        "total": total,
        "auto_accepted": auto_accepted,
        "already_labeled": already_labeled,
        "remaining": remaining,
    }
    return stats


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Auto-accept high-confidence morphological labels"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unambig-threshold", type=float, default=0.95)
    parser.add_argument("--ambig-threshold", type=float, default=0.98)
    args = parser.parse_args()

    stats = auto_accept_seed(
        args.input, args.output,
        args.unambig_threshold, args.ambig_threshold,
    )

    hours_saved = stats["auto_accepted"] / 250
    hours_remaining = stats["remaining"] / 250

    print(f"\n{'='*60}")
    print("AUTO-ACCEPT RESULTS")
    print(f"{'='*60}")
    print(f"Total tokens:       {stats['total']}")
    print(f"Auto-accepted:      {stats['auto_accepted']} "
          f"({stats['auto_accepted']/max(stats['total'],1):.1%})")
    print(f"Already labeled:    {stats['already_labeled']}")
    print(f"Remaining (human):  {stats['remaining']} "
          f"({stats['remaining']/max(stats['total'],1):.1%})")
    print(f"Time saved:         ~{hours_saved:.1f} hours")
    print(f"Time remaining:     ~{hours_remaining:.1f} hours")
    print(f"{'='*60}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
