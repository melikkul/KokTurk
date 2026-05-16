"""Download a pilot sample from OSCAR-tr to a local JSONL file.

Run this on the TRUBA login node (has internet) BEFORE submitting the
preprocessing SLURM job. The output file goes to /arf/scratch/ so that
compute nodes (which may not have internet) can read it.

Usage:
    python scripts/data/download_oscar_pilot.py \
        --out /arf/scratch/scolakoglu/oscar-tr-pilot.jsonl \
        --max-sentences 500000

The output is one JSON object per line: {"text": "<sentence>", "source": "oscar-tr"}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default="/arf/scratch/scolakoglu/oscar-tr-pilot.jsonl",
        help="Output JSONL path",
    )
    ap.add_argument(
        "--max-sentences",
        type=int,
        default=500_000,
        help="Stop after this many sentences (default: 500K)",
    )
    ap.add_argument(
        "--hf-dataset",
        default="oscar-corpus/OSCAR-2301",
        help="HuggingFace dataset repo (default: OSCAR-2301)",
    )
    ap.add_argument(
        "--language",
        default="tr",
        help="Language subset for OSCAR (default: tr)",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading up to {args.max_sentences:,} sentences from {args.hf_dataset} [{args.language}]")
    print(f"Output: {out_path}")

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' not installed. Run: pip install datasets", file=sys.stderr)
        sys.exit(1)

    try:
        ds = load_dataset(
            args.hf_dataset,
            language=args.language,
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    except TypeError:
        ds = load_dataset(
            args.hf_dataset,
            split="train",
            streaming=True,
        )

    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            text = row.get("text") or row.get("content") or ""
            if not text or not text.strip():
                continue
            for sent in text.split("\n"):
                sent = sent.strip()
                if len(sent) < 10:
                    continue
                f.write(json.dumps({"text": sent, "source": "oscar-tr"}, ensure_ascii=False) + "\n")
                written += 1
                if written >= args.max_sentences:
                    break
            if written >= args.max_sentences:
                break
            if written % 10_000 == 0:
                print(f"  {written:,} sentences ...", flush=True)

    print(f"Done: {written:,} sentences written to {out_path}")
    print(f"Next step:")
    print(f"  sbatch scripts/truba/submit_preprocess_aksu.sh \\")
    print(f"    --local-jsonl {out_path}")


if __name__ == "__main__":
    main()
