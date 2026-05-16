"""Ingest a SLURM job output JSON into audit/benchmark_results/metrics.json.

Validates ranges, types, and refuses to silently overwrite non-null values.
Writes atomically (temp file + rename).

Usage:
    python scripts/ingest_metrics.py \\
        --source audit/benchmark_results/zeyrek_throughput.json \\
        --keys tok_per_sec:zeyrek_tok_per_sec,host_cpu:zeyrek_hardware \\
        --target audit/benchmark_results/metrics.json

    # mapping format: source_key:dest_key (or just key if names match)
    # allow overwriting a non-null value:
    python scripts/ingest_metrics.py --source ... --keys ... --allow-overwrite
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

VALID_RANGES: dict[str, tuple[float, float] | None] = {
    "em_argmax_ensemble":   (0.0, 1.0),
    "em_argmax_std":        (0.0, 0.1),
    "em_string_ensemble":   (0.0, 1.0),
    "em_string_single_seed_min": (0.0, 1.0),
    "em_string_single_seed_max": (0.0, 1.0),
    "training_wall_clock_min": (0.0, 10000.0),
    "zeyrek_tok_per_sec":   (0.0, 1_000_000.0),
    "berturk_sent_per_sec": (0.0, 1_000_000.0),
    "reranker_tok_per_sec": (0.0, 1_000_000.0),
    "dualhead_tok_per_sec": (0.0, 1_000_000.0),
    "dualhead_em":          (0.0, 1.0),
    "dataset_v1_entries":   (1000, 10_000_000),
    "dataset_v2_entries":   (1000, 10_000_000),
    "dataset_boundary_coverage": (0.0, 1.0),
    "classification_macro_f1_atomized_berturk": (0.0, 1.0),
}


def _validate(key: str, value: object) -> None:
    if value is None:
        return
    if key not in VALID_RANGES:
        return
    rng = VALID_RANGES[key]
    if rng is None:
        return
    lo, hi = rng
    if not isinstance(value, (int, float)):
        raise ValueError(f"{key}: expected numeric, got {type(value).__name__}")
    if not (lo <= float(value) <= hi):
        raise ValueError(f"{key}={value} is outside valid range [{lo}, {hi}]")


def ingest(
    source_path: Path,
    key_map: dict[str, str],
    target_path: Path,
    allow_overwrite: bool = False,
) -> dict:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    target = json.loads(target_path.read_text(encoding="utf-8"))

    changed: dict[str, tuple] = {}
    for src_key, dst_key in key_map.items():
        if src_key not in source:
            print(f"WARNING: key {src_key!r} not in source — skipped", file=sys.stderr)
            continue
        value = source[src_key]
        _validate(dst_key, value)

        existing = target.get(dst_key)
        if existing is not None and not allow_overwrite:
            raise RuntimeError(
                f"metrics.json[{dst_key!r}] is already {existing!r}. "
                f"Use --allow-overwrite to replace (previous value will be logged)."
            )
        if existing is not None and allow_overwrite:
            print(f"OVERWRITE {dst_key}: {existing!r} → {value!r}")

        changed[dst_key] = (existing, value)
        target[dst_key] = value

    # Atomic write: temp + rename
    tmp = target_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(target, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, target_path)

    for k, (old, new) in changed.items():
        print(f"  {k}: {old!r} → {new!r}")
    return changed


def _parse_key_map(keys_str: str) -> dict[str, str]:
    result = {}
    for pair in keys_str.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" in pair:
            src, dst = pair.split(":", 1)
        else:
            src = dst = pair
        result[src.strip()] = dst.strip()
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="Source JSON file (job output)")
    ap.add_argument(
        "--keys",
        required=True,
        help="Comma-separated source_key:dest_key pairs (or just key if names match)",
    )
    ap.add_argument(
        "--target",
        default="audit/benchmark_results/metrics.json",
        help="Target metrics.json (default: audit/benchmark_results/metrics.json)",
    )
    ap.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow replacing non-null values (logs previous value)",
    )
    args = ap.parse_args()

    src = Path(args.source)
    tgt = Path(args.target)
    if not src.exists():
        print(f"ERROR: source {src} not found", file=sys.stderr)
        sys.exit(1)
    if not tgt.exists():
        print(f"ERROR: target {tgt} not found", file=sys.stderr)
        sys.exit(1)

    key_map = _parse_key_map(args.keys)
    if not key_map:
        print("ERROR: --keys is empty", file=sys.stderr)
        sys.exit(1)

    print(f"Ingesting {src} → {tgt}")
    changed = ingest(src, key_map, tgt, allow_overwrite=args.allow_overwrite)
    print(f"Done. {len(changed)} key(s) updated.")


if __name__ == "__main__":
    main()
