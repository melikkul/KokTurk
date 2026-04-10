"""Merge gold annotations from seed + AL batches into a single file.

Validates completeness, then produces a unified gold corpus file.

Usage:
    python src/annotation/merge_annotations.py \
        --seed data/gold/seed/seed_200_annotated.jsonl \
        --output data/gold/combined_gold.jsonl
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

GOLD_DIR = Path("data/gold")


def load_jsonl(path: Path) -> list[dict[str, object]]:
    """Load JSONL file."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def merge_annotations(
    seed_path: Path,
    batch_paths: list[Path],
    output_path: Path,
) -> dict[str, object]:
    """Merge seed + batch annotations into a single gold file.

    Args:
        seed_path: Path to annotated seed JSONL.
        batch_paths: List of annotated batch JSONL files.
        output_path: Path for combined gold output.

    Returns:
        Summary statistics.
    """
    all_sentences: list[dict[str, object]] = []
    seen_ids: set[str] = set()

    sources: Counter[str] = Counter()

    # Load seed
    if seed_path.exists():
        for sent in load_jsonl(seed_path):
            sid = str(sent["sentence_id"])
            if sid not in seen_ids:
                all_sentences.append(sent)
                seen_ids.add(sid)
                sources["seed"] += 1

    # Load batches in order
    for batch_path in sorted(batch_paths):
        if not batch_path.exists():
            continue
        for sent in load_jsonl(batch_path):
            sid = str(sent["sentence_id"])
            if sid not in seen_ids:
                all_sentences.append(sent)
                seen_ids.add(sid)
                sources[batch_path.stem] += 1

    # Count tokens
    total_tokens = 0
    labeled_tokens = 0
    auto_accepted = 0
    manual = 0

    for sent in all_sentences:
        for token in sent.get("tokens", []):  # type: ignore[union-attr]
            total_tokens += 1
            gold = token.get("gold_label")  # type: ignore[union-attr]
            if gold is not None:
                labeled_tokens += 1
                # Heuristic: auto-accepted if gold matches top candidate
                parses = token.get("candidate_parses", [])  # type: ignore[union-attr]
                if parses:
                    top = parses[0]
                    top_tags = " ".join(top.get("tags", []))  # type: ignore[union-attr]
                    top_label = f"{top['root']} {top_tags}".strip()  # type: ignore[index]
                    if gold == top_label:
                        auto_accepted += 1
                    else:
                        manual += 1
                else:
                    manual += 1

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sent in all_sentences:
            f.write(json.dumps(sent, ensure_ascii=False) + "\n")

    stats: dict[str, object] = {
        "sentences": len(all_sentences),
        "total_tokens": total_tokens,
        "labeled_tokens": labeled_tokens,
        "unlabeled_tokens": total_tokens - labeled_tokens,
        "auto_accepted": auto_accepted,
        "manual": manual,
        "sources": dict(sources),
    }

    logger.info("Merged %d sentences to %s", len(all_sentences), output_path)
    return stats


def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Merge gold annotations")
    parser.add_argument(
        "--seed", type=Path,
        default=GOLD_DIR / "seed" / "seed_200_partial.jsonl",
    )
    parser.add_argument(
        "--output", type=Path,
        default=GOLD_DIR / "combined_gold.jsonl",
    )
    args = parser.parse_args()

    # Find batch files
    batch_paths = sorted(GOLD_DIR.glob("batch_*.jsonl"))
    logger.info("Found %d batch files", len(batch_paths))

    stats = merge_annotations(args.seed, batch_paths, args.output)

    print(f"\n{'='*60}")
    print("MERGE RESULTS")
    print(f"{'='*60}")
    print(f"Sentences:       {stats['sentences']}")
    print(f"Total tokens:    {stats['total_tokens']}")
    print(f"Labeled:         {stats['labeled_tokens']}")
    print(f"Unlabeled:       {stats['unlabeled_tokens']}")
    print(f"Auto-accepted:   {stats['auto_accepted']}")
    print(f"Manual:          {stats['manual']}")
    print(f"Sources:         {stats['sources']}")
    print(f"Output:          {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
