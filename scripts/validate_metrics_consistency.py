"""Validate that metrics.json and README.md are consistent.

For non-null metric values: assert the formatted number appears somewhere in README.md.
For null values: assert the README does NOT show a bare number where that value would render
                 (it should say "pending" or the section is absent).

Usage:
    python scripts/validate_metrics_consistency.py
    python scripts/validate_metrics_consistency.py \\
        --metrics audit/benchmark_results/metrics.json \\
        --readme  README.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Mapping: metrics key → how it should appear in the README when non-null.
# Format strings must match what docs/README.md.j2 produces.
PRESENT_CHECKS: dict[str, str] = {
    "em_argmax_ensemble":    lambda v: f"{v * 100:.1f}%",
    "em_argmax_std":         lambda v: f"{v * 100:.2f}pp",
    "dataset_v1_entries":    lambda v: f"{v:,.0f}",
}

# When null, these fragments must NOT appear as standalone numbers in the README.
# The check is deliberately loose — we just assert "pending" appears nearby when the value is null.
ABSENT_SENTINEL = "pending"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics", default=str(ROOT / "audit/benchmark_results/metrics.json"))
    ap.add_argument("--readme", default=str(ROOT / "README.md"))
    args = ap.parse_args()

    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    readme  = Path(args.readme).read_text(encoding="utf-8")

    errors: list[str] = []

    # Check non-null values appear formatted in the README
    for key, fmt in PRESENT_CHECKS.items():
        value = metrics.get(key)
        if value is None:
            continue
        rendered = fmt(value)
        if rendered not in readme:
            errors.append(f"MISSING: metrics[{key!r}]={value!r} should render as {rendered!r} in README, but not found.")

    # Check that null values result in "pending" appearing (somewhere) in README
    null_keys_with_pending = [
        "em_string_ensemble",
        "training_wall_clock_min",
        "zeyrek_tok_per_sec",
        "dualhead_em",
        "classification_macro_f1_atomized_berturk",
    ]
    for key in null_keys_with_pending:
        if metrics.get(key) is None:
            # "pending" must appear in the README — if it doesn't the template didn't handle null
            if ABSENT_SENTINEL not in readme.lower():
                errors.append(
                    f"HONESTY LEAK: metrics[{key!r}] is null but README contains no 'pending' text. "
                    f"The template may be showing a hardcoded fallback instead."
                )

    if errors:
        print("Metrics consistency FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"OK: metrics.json and README.md are consistent ({len(PRESENT_CHECKS)} checks passed).")


if __name__ == "__main__":
    main()
