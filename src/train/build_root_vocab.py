"""Build root vocabulary from corpus labels for DualHeadAtomizer.

Reads JSONL corpus files where each line has a ``label`` field like
``"ev +PLU +POSS.3SG +ABL"``.  The root is the first whitespace-delimited
token (everything before the first ``+`` tag).  Roots are counted, filtered
by minimum frequency, and saved as a JSON array to ``root_vocab.json`` in the
same format used by ``char_vocab.json`` and ``tag_vocab.json``.

Example usage::

    python src/train/build_root_vocab.py \\
        --corpus data/splits/train.jsonl \\
        --output models/vocabs/root_vocab.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def build_root_vocab(
    corpus_paths: list[str],
    min_count: int = 2,
    output_path: str = "models/vocabs/root_vocab.json",
) -> dict[str, int]:
    """Extract root tokens from corpus, build root_vocab.json.

    Args:
        corpus_paths: List of JSONL file paths to read from.
        min_count: Minimum frequency to include a root in the vocabulary.
        output_path: Destination path for the JSON array file.

    Returns:
        token2idx dict mapping root string to its integer index.
    """
    counter: Counter[str] = Counter()

    for corpus_path in corpus_paths:
        path = Path(corpus_path)
        if not path.exists():
            raise FileNotFoundError(f"Corpus file not found: {path}")
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                label = record.get("label", "")
                if not label or not label.strip():
                    continue
                # Root is the first whitespace-delimited token in the label.
                root = label.split()[0]
                counter[root] += 1

    total_before = len(counter)

    # Filter by minimum count.
    filtered = {root: cnt for root, cnt in counter.items() if cnt >= min_count}
    total_after = len(filtered)

    # Sort by descending frequency, then lexicographically for stability.
    sorted_roots = sorted(filtered.keys(), key=lambda r: (-filtered[r], r))

    top10 = sorted_roots[:10]
    top10_with_counts = [(r, filtered[r]) for r in top10]

    # Build final vocabulary: special tokens first, then roots.
    special_tokens: list[str] = ["<PAD>", "<UNK_ROOT>"]
    vocab_list: list[str] = special_tokens + sorted_roots

    token2idx: dict[str, int] = {tok: idx for idx, tok in enumerate(vocab_list)}

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(vocab_list, fh, ensure_ascii=False, indent=1)

    # Diagnostic output.
    print(f"Root types before filtering (min_count={min_count}): {total_before}")
    print(f"Root types after  filtering: {total_after}")
    print(f"Final vocab size (incl. special tokens): {len(vocab_list)}")
    print("Top-10 most frequent roots:")
    for rank, (root, cnt) in enumerate(top10_with_counts, start=1):
        print(f"  {rank:>2}. {root!r:30s}  freq={cnt}")

    return token2idx


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build root vocabulary from tiered JSONL corpus.",
    )
    parser.add_argument(
        "--corpus",
        action="append",
        dest="corpus",
        required=True,
        metavar="PATH",
        help="Path to a JSONL corpus file (may be repeated for multiple files).",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="Minimum frequency to include a root (default: 2).",
    )
    parser.add_argument(
        "--output",
        default="models/vocabs/root_vocab.json",
        help="Output path for the JSON array (default: models/vocabs/root_vocab.json).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for command-line invocation."""
    args = _parse_args()
    build_root_vocab(
        corpus_paths=args.corpus,
        min_count=args.min_count,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
