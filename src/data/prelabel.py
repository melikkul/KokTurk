"""Parallel pre-labeling — run morphological analyzers on the entire corpus.

For every token, queries Zeyrek (and TRMorph if available) to produce
candidate parses. Outputs candidates.jsonl with per-token analyses and
agreement statistics.

Usage:
    python src/data/prelabel.py
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

from kokturk.core.analyzer import MorphoAnalyzer

logger = logging.getLogger(__name__)

CORPUS_PATH = Path("data/processed/corpus.jsonl")
OUTPUT_PATH = Path("data/prelabeled/candidates.jsonl")


def load_corpus(path: Path = CORPUS_PATH) -> list[dict[str, object]]:
    """Load the ingested corpus from JSONL."""
    sentences: list[dict[str, object]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            sentences.append(json.loads(line))
    return sentences


def prelabel_corpus(
    corpus: list[dict[str, object]],
    output_path: Path = OUTPUT_PATH,
    backends: list[str] | None = None,
) -> dict[str, object]:
    """Run morphological analysis on every token in the corpus.

    Args:
        corpus: List of sentence dicts from ingest.py.
        output_path: Path for the output JSONL file.
        backends: Analyzer backend names. Default: ["zeyrek"].

    Returns:
        Dict of summary statistics.
    """
    if backends is None:
        backends = ["zeyrek"]

    analyzer = MorphoAnalyzer(backends=backends)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_tokens = 0
    zeyrek_has_parse = 0
    trmorph_has_parse = 0
    both_agree = 0
    both_zero = 0
    parse_count_dist: Counter[str] = Counter()
    oov_tokens: list[str] = []

    with open(output_path, "w", encoding="utf-8") as f:
        for sent_idx, sent in enumerate(corpus):
            sentence_id = str(sent["sentence_id"])
            tokens: list[str] = sent["tokens"]  # type: ignore[assignment]

            for token_idx, surface in enumerate(tokens):
                total_tokens += 1
                result = analyzer.analyze(surface)

                # Track per-backend coverage
                zeyrek_parses = [a for a in result.analyses if a.source == "zeyrek"]
                trmorph_parses = [a for a in result.analyses if a.source == "trmorph"]

                has_zeyrek = len(zeyrek_parses) > 0
                has_trmorph = len(trmorph_parses) > 0

                if has_zeyrek:
                    zeyrek_has_parse += 1
                if has_trmorph:
                    trmorph_has_parse += 1

                # Check agreement: do they share at least one parse identity?
                zeyrek_ids = {a.parse_identity() for a in zeyrek_parses}
                trmorph_ids = {a.parse_identity() for a in trmorph_parses}
                agree = bool(zeyrek_ids & trmorph_ids) if (has_zeyrek and has_trmorph) else False
                if agree:
                    both_agree += 1

                if not has_zeyrek and not has_trmorph:
                    both_zero += 1
                    oov_tokens.append(surface)

                # Parse count bucket
                pc = result.parse_count
                bucket = str(pc) if pc <= 4 else "5+"
                parse_count_dist[bucket] += 1

                # Serialize analyses
                analyses_json = [
                    {
                        "root": a.root,
                        "tags": list(a.tags),
                        "source": a.source,
                        "score": a.score,
                    }
                    for a in result.analyses
                ]

                record = {
                    "sentence_id": sentence_id,
                    "token_idx": token_idx,
                    "surface": surface,
                    "analyses": analyses_json,
                    "parse_count": result.parse_count,
                    "zeyrek_trmorph_agree": agree,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (sent_idx + 1) % 1000 == 0:
                logger.info("Processed %d/%d sentences", sent_idx + 1, len(corpus))

    analyzer.close()

    # Compute statistics
    stats: dict[str, object] = {
        "total_tokens": total_tokens,
        "zeyrek_coverage": zeyrek_has_parse / max(total_tokens, 1),
        "trmorph_coverage": trmorph_has_parse / max(total_tokens, 1),
        "agreement_rate": both_agree / max(total_tokens, 1),
        "both_zero_count": both_zero,
        "parse_count_distribution": dict(sorted(parse_count_dist.items())),
        "cache_hit_rate": analyzer.cache.hit_rate,
        "unique_oov_tokens": len(set(oov_tokens)),
    }

    logger.info("Pre-labeling complete. Stats: %s", json.dumps(stats, indent=2))
    return stats


def print_stats(stats: dict[str, object]) -> None:
    """Print pre-labeling summary statistics."""
    print(f"\n{'='*60}")
    print("PRE-LABELING SUMMARY")
    print(f"{'='*60}")
    print(f"Total tokens:          {stats['total_tokens']}")
    print(f"Zeyrek coverage:       {stats['zeyrek_coverage']:.1%}")
    print(f"TRMorph coverage:      {stats['trmorph_coverage']:.1%}")
    print(f"Agreement rate:        {stats['agreement_rate']:.1%}")
    print(f"Both zero (OOV):       {stats['both_zero_count']}")
    print(f"Cache hit rate:        {stats['cache_hit_rate']:.1%}")
    print("\nParse count distribution:")
    dist = stats["parse_count_distribution"]
    for k, v in sorted(dist.items()):  # type: ignore[union-attr]
        print(f"  {k} parses: {v}")
    print(f"{'='*60}\n")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    if not CORPUS_PATH.exists():
        logger.error("Corpus not found at %s. Run ingest.py first.", CORPUS_PATH)
        sys.exit(1)

    corpus = load_corpus()
    logger.info("Loaded %d sentences from corpus", len(corpus))

    # Only use zeyrek — trmorph requires foma which isn't available on this node
    stats = prelabel_corpus(corpus, backends=["zeyrek"])
    print_stats(stats)


if __name__ == "__main__":
    main()
