"""Build surface-form (word) vocabulary from corpus for ContextualDualHeadAtomizer.

Reads JSONL corpus files where each line has a ``surface`` field.
Saves a JSON object ``{word: idx, ...}`` with ``<PAD>``→0, ``<UNK>``→1,
then remaining words sorted by descending frequency.

Example usage::

    python src/train/build_word_vocab.py \\
        --corpus data/splits/train.jsonl \\
        --corpus data/splits/val.jsonl \\
        --output models/vocabs/word_vocab.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def build_word_vocab(
    corpus_paths: list[str],
    min_count: int = 2,
    output_path: str = "models/vocabs/word_vocab.json",
) -> dict[str, int]:
    """Extract surface forms from corpus, build word_vocab.json.

    Args:
        corpus_paths: List of JSONL file paths to read from.
        min_count: Minimum frequency to include a word in the vocabulary.
        output_path: Destination path for the JSON object file.

    Returns:
        token2idx dict mapping surface string to its integer index.
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
                surface = record.get("surface", "").lower().strip()
                if surface:
                    counter[surface] += 1

    total_before = len(counter)
    filtered = {w: c for w, c in counter.items() if c >= min_count}
    total_after = len(filtered)

    sorted_words = sorted(filtered.keys(), key=lambda w: (-filtered[w], w))

    vocab: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    for word in sorted_words:
        vocab[word] = len(vocab)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(vocab, fh, ensure_ascii=False, indent=1)

    print(f"Surface types before filtering (min_count={min_count}): {total_before}")
    print(f"Surface types after  filtering: {total_after}")
    print(f"Final vocab size (incl. special tokens): {len(vocab)}")
    print(f"Saved to: {out}")
    return vocab


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build surface-form vocabulary from tiered JSONL corpus.",
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
        help="Minimum frequency to include a word (default: 2).",
    )
    parser.add_argument(
        "--output",
        default="models/vocabs/word_vocab.json",
        help="Output path for the JSON object (default: models/vocabs/word_vocab.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_word_vocab(
        corpus_paths=args.corpus,
        min_count=args.min_count,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
