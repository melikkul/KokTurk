"""Extrapolate dataset yield from autolabel pilot output.

Reads the pilot JSONL, counts entries by tier after quality filter,
extrapolates to the full target, and emits a JSON report.

Triggers the §0 halt condition if extrapolated_after_quality_filter < 1,500,000.

Usage:
    python scripts/data/extrapolate_yield.py \\
        --pilot-output data/intermediate/autolabel_pilot.jsonl \\
        --target 2500000 \\
        --output audit/benchmark_results/yield_extrapolation.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HALT_THRESHOLD = 1_500_000


def extrapolate(pilot_path: Path, target: int) -> dict:
    lines = [
        json.loads(l) for l in pilot_path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    n_pilot = len(lines)
    if n_pilot == 0:
        raise ValueError(f"Pilot file {pilot_path} is empty")

    tier_counts: dict[str, int] = {}
    for entry in lines:
        tier = entry.get("tier", "drop")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    n_publishable = sum(
        v for k, v in tier_counts.items() if k in ("gold", "silver", "bronze")
    )
    quality_rate = n_publishable / n_pilot

    total_raw_estimate = 10_000_000  # estimated raw tokens from sources.py
    extrapolated_unique_tokens = int(total_raw_estimate * 0.27)  # Turkish type/token ratio
    extrapolated_after_quality_filter = int(extrapolated_unique_tokens * quality_rate)

    report = {
        "pilot_tokens": n_pilot,
        "pilot_publishable": n_publishable,
        "quality_rate": quality_rate,
        "pilot_tier_counts": tier_counts,
        "total_raw_estimate": total_raw_estimate,
        "extrapolated_unique_tokens": extrapolated_unique_tokens,
        "extrapolated_after_quality_filter": extrapolated_after_quality_filter,
        "target": target,
        "will_reach_target": extrapolated_after_quality_filter >= target,
        "halt_threshold": HALT_THRESHOLD,
        "trigger_halt": extrapolated_after_quality_filter < HALT_THRESHOLD,
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot-output", required=True)
    ap.add_argument("--target", type=int, default=2_500_000)
    ap.add_argument("--output", default="audit/benchmark_results/yield_extrapolation.json")
    args = ap.parse_args()

    report = extrapolate(Path(args.pilot_output), args.target)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))

    if report["trigger_halt"]:
        print(
            f"\n*** HALT CONDITION TRIGGERED ***\n"
            f"Extrapolated yield {report['extrapolated_after_quality_filter']:,} < "
            f"halt threshold {HALT_THRESHOLD:,}.\n"
            f"Write audit/halt_reports/<date>-autolabel-yield.md and consult user.",
            file=sys.stderr,
        )
        sys.exit(2)
    elif not report["will_reach_target"]:
        print(
            f"\nWARNING: extrapolated {report['extrapolated_after_quality_filter']:,} < "
            f"target {args.target:,}. Consider adding more source corpora.",
            file=sys.stderr,
        )
    else:
        print(f"\nYield looks sufficient: {report['extrapolated_after_quality_filter']:,} >= {args.target:,}")


if __name__ == "__main__":
    main()
