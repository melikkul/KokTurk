"""Audit the built corpus for quality and coverage.

Usage:
    PYTHONPATH=src python src/data/corpus_audit.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from kokturk.core.constants import ZEYREK_TO_CANONICAL

CORPUS_PATH = Path("data/gold/tr_gold_morph_v1.jsonl")


def audit_corpus(corpus_path: Path = CORPUS_PATH) -> dict[str, object]:
    """Run quality checks on the corpus."""
    records: list[dict[str, object]] = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    total = len(records)
    tier_counts = Counter(str(r.get("tier", "?")) for r in records)
    root_counter: Counter[str] = Counter()
    tag_seq_counter: Counter[str] = Counter()
    tag_counter: Counter[str] = Counter()

    for r in records:
        label = str(r.get("label", ""))
        parts = label.split()
        if parts:
            root_counter[parts[0]] += 1
            tags = parts[1:]
            tag_seq_counter[" ".join(tags)] += 1
            for t in tags:
                tag_counter[t] += 1

    # Check canonical tag coverage
    expected_tags = {v for v in ZEYREK_TO_CANONICAL.values() if v}
    missing_tags = expected_tags - set(tag_counter.keys())

    results: dict[str, object] = {
        "total_tokens": total,
        "tier_distribution": dict(tier_counts),
        "unique_roots": len(root_counter),
        "gold_fraction": tier_counts.get("gold", 0) / max(total, 1),
        "top20_roots": root_counter.most_common(20),
        "top20_tag_sequences": tag_seq_counter.most_common(20),
        "total_unique_tags": len(tag_counter),
        "missing_canonical_tags": sorted(missing_tags),
        "missing_count": len(missing_tags),
    }
    return results


def main() -> None:
    results = audit_corpus()

    print(f"\n{'='*65}")
    print("CORPUS AUDIT REPORT")
    print(f"{'='*65}")
    print(f"Total tokens:    {results['total_tokens']}")
    ok_total = results["total_tokens"] >= 50000  # type: ignore[operator]
    print(f"  >= 50K check:  {'PASS' if ok_total else 'FAIL'}")
    print(f"\nTier distribution: {results['tier_distribution']}")
    gf = results["gold_fraction"]
    print(f"  Gold fraction: {gf:.1%} "  # type: ignore[str-format]
          f"({'PASS' if gf >= 0.02 else 'WARN: < 2%'})")  # type: ignore[operator]
    print(f"\nUnique roots:    {results['unique_roots']}")
    print(f"Unique tags:     {results['total_unique_tags']}")
    print(f"\nMissing canonical tags ({results['missing_count']}):")
    for t in results["missing_canonical_tags"]:  # type: ignore[union-attr]
        print(f"  {t}")

    print("\nTop-20 roots:")
    for root, count in results["top20_roots"]:  # type: ignore[union-attr]
        print(f"  {root:20s} {count:6d}")

    print("\nTop-20 tag sequences:")
    for seq, count in results["top20_tag_sequences"]:  # type: ignore[union-attr]
        label = seq if seq else "(root only)"
        print(f"  {label:40s} {count:6d}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
