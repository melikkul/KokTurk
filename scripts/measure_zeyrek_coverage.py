"""Measure Zeyrek candidate coverage against gold labels.

GO/NO-GO gate for the disambiguation approach: if gold labels are not
present in Zeyrek's candidate list for most tokens, disambiguation
cannot work.

Usage:
    PYTHONPATH=src python scripts/measure_zeyrek_coverage.py

Output:
    - Coverage statistics (% tokens where gold is in candidates)
    - Candidate count distribution
    - Unambiguous token ratio (EM floor)
    - 10 side-by-side examples of Zeyrek vs gold
    - JSON report to models/benchmark/zeyrek_coverage.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kokturk.core.analyzer import ZeyrekBackend


def load_data(path: Path) -> list[dict]:
    """Load JSONL file."""
    items = []
    for line in path.open():
        items.append(json.loads(line))
    return items


def measure_coverage(splits: list[Path]) -> dict:
    """Measure Zeyrek coverage across all splits."""
    backend = ZeyrekBackend()

    total = 0
    gold_found = 0
    no_candidates = 0
    candidate_counts: list[int] = []
    side_by_side: list[dict] = []
    mismatches: list[dict] = []

    for split_path in splits:
        if not split_path.exists():
            print(f"  SKIP: {split_path} not found")
            continue

        data = load_data(split_path)
        print(f"  Processing {split_path.name}: {len(data)} tokens")

        for item in data:
            surface = item["surface"]
            gold_label = item["label"]
            total += 1

            # Get Zeyrek candidates
            analyses = backend.analyze(surface)
            candidate_strs = [a.to_str() for a in analyses]
            num_candidates = len(candidate_strs)
            candidate_counts.append(num_candidates)

            # Check if gold is in candidates
            found = gold_label in candidate_strs
            if found:
                gold_found += 1

            if num_candidates == 0:
                no_candidates += 1

            # Collect side-by-side examples (first 10 ambiguous with match)
            if len(side_by_side) < 10 and num_candidates >= 2 and found:
                side_by_side.append({
                    "surface": surface,
                    "gold": gold_label,
                    "candidates": candidate_strs,
                    "gold_idx": candidate_strs.index(gold_label),
                })

            # Collect mismatch examples (first 10 where gold NOT in candidates)
            if len(mismatches) < 10 and not found and num_candidates > 0:
                mismatches.append({
                    "surface": surface,
                    "gold": gold_label,
                    "candidates": candidate_strs[:5],  # first 5 for brevity
                })

            if total % 10000 == 0:
                print(f"    {total} tokens processed, coverage so far: "
                      f"{gold_found / total:.1%}")

    # Compute statistics
    count_dist = Counter(candidate_counts)
    unambiguous = sum(1 for c in candidate_counts if c == 1)
    ambiguous = sum(1 for c in candidate_counts if c > 1)
    coverage = gold_found / total if total > 0 else 0
    unambiguous_ratio = unambiguous / total if total > 0 else 0
    avg_candidates = sum(candidate_counts) / total if total > 0 else 0

    return {
        "total_tokens": total,
        "gold_in_candidates": gold_found,
        "coverage": coverage,
        "no_candidates_count": no_candidates,
        "oov_rate": no_candidates / total if total > 0 else 0,
        "unambiguous_count": unambiguous,
        "unambiguous_ratio": unambiguous_ratio,
        "ambiguous_count": ambiguous,
        "avg_candidates": avg_candidates,
        "candidate_count_distribution": {
            str(k): v for k, v in sorted(count_dist.items())
        },
        "em_floor": unambiguous_ratio,
        "side_by_side_examples": side_by_side,
        "mismatch_examples": mismatches,
    }


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    splits_dir = project_root / "data" / "splits"

    splits = [
        splits_dir / "train.jsonl",
        splits_dir / "val.jsonl",
        splits_dir / "test.jsonl",
    ]

    print("=== Zeyrek Coverage Diagnostic ===\n")
    results = measure_coverage(splits)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total tokens:          {results['total_tokens']:,}")
    print(f"Gold in candidates:    {results['gold_in_candidates']:,} "
          f"({results['coverage']:.1%})")
    print(f"No candidates (OOV):   {results['no_candidates_count']:,} "
          f"({results['oov_rate']:.1%})")
    print(f"Unambiguous (K=1):     {results['unambiguous_count']:,} "
          f"({results['unambiguous_ratio']:.1%})")
    print(f"Ambiguous (K>1):       {results['ambiguous_count']:,}")
    print(f"Avg candidates/token:  {results['avg_candidates']:.1f}")
    print(f"")
    print(f"EM floor (unambiguous): {results['em_floor']:.1%}")
    print(f"  (These tokens have only 1 candidate = trivially correct)")

    # Candidate count distribution
    print(f"\nCandidate count distribution:")
    for k, v in sorted(results["candidate_count_distribution"].items(),
                       key=lambda x: int(x[0])):
        pct = int(v) / results["total_tokens"] * 100
        bar = "#" * int(pct)
        print(f"  K={k:>2s}: {v:>6d} ({pct:5.1f}%) {bar}")

    # Side-by-side examples
    print(f"\n{'=' * 60}")
    print(f"SIDE-BY-SIDE: Zeyrek candidates vs gold (10 ambiguous matches)")
    print(f"{'=' * 60}")
    for i, ex in enumerate(results["side_by_side_examples"], 1):
        print(f"\n  [{i}] surface: '{ex['surface']}'")
        print(f"      gold:    '{ex['gold']}'")
        print(f"      candidates ({len(ex['candidates'])}):")
        for j, c in enumerate(ex["candidates"]):
            marker = " <<< GOLD" if j == ex["gold_idx"] else ""
            print(f"        [{j}] '{c}'{marker}")

    # Mismatch examples
    print(f"\n{'=' * 60}")
    print(f"MISMATCHES: Gold NOT in Zeyrek candidates (first 10)")
    print(f"{'=' * 60}")
    for i, ex in enumerate(results["mismatch_examples"], 1):
        print(f"\n  [{i}] surface: '{ex['surface']}'")
        print(f"      gold:    '{ex['gold']}'")
        print(f"      Zeyrek candidates:")
        for j, c in enumerate(ex["candidates"]):
            print(f"        [{j}] '{c}'")

    # GO/NO-GO decision
    print(f"\n{'=' * 60}")
    if results["coverage"] >= 0.90:
        print(f"GO: Coverage {results['coverage']:.1%} >= 90% threshold")
        print(f"Disambiguation approach is viable.")
    else:
        print(f"NO-GO: Coverage {results['coverage']:.1%} < 90% threshold")
        print(f"Disambiguation approach needs tag mapping fixes first.")

    mismatch_rate = 1.0 - results["coverage"]
    if mismatch_rate > 0.05:
        print(f"\nWARNING: Format mismatch rate {mismatch_rate:.1%} > 5%")
        print(f"Review ZEYREK_TO_CANONICAL mapping in constants.py")
    print(f"{'=' * 60}")

    # Save JSON report
    out_dir = project_root / "models" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "zeyrek_coverage.json"

    # Remove non-serializable parts for JSON
    json_results = {k: v for k, v in results.items()}
    with open(out_path, "w") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"\nJSON report saved to: {out_path}")


if __name__ == "__main__":
    main()
