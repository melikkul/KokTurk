"""Noise-aware label model aggregation.

Aggregates votes from 6 labeling functions into probabilistic labels
using weighted majority vote with estimated LF precision weights.

Snorkel's LabelModel requires fixed-cardinality label space, but our
labels are per-token candidate indices (variable cardinality). We use
precision-weighted voting instead, which handles this correctly.

Usage:
    python src/data/label_model.py
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

import numpy as np

from data.labeling_functions import (
    ABSTAIN,
    ALL_LFS,
    build_label_matrix,
    compute_lf_stats,
)

logger = logging.getLogger(__name__)

OUTPUT_PATH = Path("data/weak_labels/probabilistic_labels.jsonl")
ENTROPY_THRESHOLD = 1.5
CONFIDENCE_THRESHOLD = 0.6

# Estimated precision weights per LF (from domain knowledge + LF design).
# Higher weight = more trusted. Zero-coverage LFs get weight 0.
LF_PRECISION_WEIGHTS: dict[str, float] = {
    "zeyrek_unambiguous": 0.95,   # Single-parse Zeyrek is very reliable
    "trmorph_unambiguous": 0.90,  # Unused (foma unavailable) but ready
    "suffix_regex": 0.92,         # High-precision patterns
    "pos_bigram": 0.70,           # Noisy — POS context is weak signal
    "gazetteer": 0.80,            # Good for proper nouns, some noise
    "neural_draft": 0.85,         # Stub — will be updated in Phase 2
}


def _build_result_record(
    record: dict[str, object],
    best_idx: int,
    confidence: float,
    entropy: float,
) -> dict[str, object]:
    """Build a single output record."""
    analyses: list[dict[str, object]] = record["analyses"]  # type: ignore[assignment]
    best = analyses[best_idx] if 0 <= best_idx < len(analyses) else None
    if best:
        tags_str = " ".join(best.get("tags", []))  # type: ignore[union-attr]
        predicted_label = f"{best['root']} {tags_str}".strip()
    else:
        predicted_label = str(record["surface"])

    return {
        "sentence_id": record["sentence_id"],
        "token_idx": record["token_idx"],
        "surface": record["surface"],
        "predicted_label": predicted_label,
        "predicted_idx": best_idx,
        "confidence": round(confidence, 4),
        "entropy": round(entropy, 4),
        "needs_human_review": bool(
            entropy > ENTROPY_THRESHOLD or confidence < CONFIDENCE_THRESHOLD
        ),
    }


def _weighted_vote_model(
    label_matrix: np.ndarray,
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Precision-weighted majority vote aggregation.

    Each LF's vote is weighted by its estimated precision. The candidate
    with the highest weighted vote sum wins. Confidence is the normalized
    weight of the winner. This outperforms uniform majority vote because
    it down-weights noisy LFs (pos_bigram) and up-weights precise ones
    (zeyrek_unambiguous, suffix_regex).
    """
    n_tokens, n_lfs = label_matrix.shape
    results: list[dict[str, object]] = []

    # Build weight vector from LF names
    weights = np.array([
        LF_PRECISION_WEIGHTS.get(lf.name, 0.5)  # type: ignore[attr-defined]
        for lf in ALL_LFS
    ])

    for i in range(n_tokens):
        record = records[i]
        analyses: list[dict[str, object]] = record["analyses"]  # type: ignore[assignment]
        n_candidates = len(analyses)

        if n_candidates == 0:
            results.append(_build_result_record(record, -1, 0.0, 0.0))
            continue

        # Accumulate weighted votes per candidate
        weighted_votes: dict[int, float] = {}
        total_weight = 0.0
        for j in range(n_lfs):
            v = int(label_matrix[i, j])
            if v != ABSTAIN and 0 <= v < n_candidates:
                weighted_votes[v] = weighted_votes.get(v, 0.0) + weights[j]
                total_weight += weights[j]

        if total_weight == 0:
            # No LF voted — low confidence on first candidate
            best_idx = 0
            confidence = 1.0 / max(n_candidates, 1)
            entropy = math.log2(max(n_candidates, 1))
        else:
            best_idx = max(weighted_votes, key=lambda k: weighted_votes[k])
            confidence = weighted_votes[best_idx] / total_weight

            # Entropy from normalized weight distribution
            probs = np.zeros(n_candidates)
            for idx, w in weighted_votes.items():
                probs[idx] = w / total_weight
            entropy = float(-sum(
                p * math.log2(p) for p in probs if p > 0
            ))

        results.append(
            _build_result_record(record, best_idx, confidence, entropy)
        )

    return results


def run_label_model(
    candidates_path: Path = Path("data/prelabeled/candidates.jsonl"),
    corpus_path: Path = Path("data/processed/corpus.jsonl"),
    output_path: Path = OUTPUT_PATH,
) -> dict[str, object]:
    """Run the full label model pipeline.

    1. Build label matrix from LFs
    2. Aggregate with precision-weighted voting
    3. Output probabilistic labels

    Returns:
        Summary statistics dict.
    """
    logger.info("Building label matrix from %d LFs...", len(ALL_LFS))
    label_matrix, records = build_label_matrix(candidates_path, corpus_path)
    logger.info("Label matrix shape: %s", label_matrix.shape)

    # LF statistics
    lf_stats = compute_lf_stats(label_matrix)
    for name, stats in lf_stats.items():
        logger.info(
            "LF %-25s coverage=%.3f overlap=%.3f conflict=%.3f",
            name, stats["coverage"], stats["overlap_rate"],
            stats["conflict_rate"],
        )

    # Precision-weighted majority vote
    logger.info("Running precision-weighted vote aggregation...")
    results = _weighted_vote_model(label_matrix, records)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Compute summary stats
    confidences = [float(r["confidence"]) for r in results]
    entropies = [float(r["entropy"]) for r in results]
    needs_review = sum(1 for r in results if r["needs_human_review"])

    summary: dict[str, object] = {
        "total_tokens": len(results),
        "mean_confidence": round(float(np.mean(confidences)), 4),
        "median_confidence": round(float(np.median(confidences)), 4),
        "mean_entropy": round(float(np.mean(entropies)), 4),
        "needs_human_review": needs_review,
        "review_fraction": round(needs_review / max(len(results), 1), 4),
        "lf_stats": lf_stats,
    }

    logger.info(
        "Wrote %d probabilistic labels to %s", len(results), output_path
    )
    review_pct = 100 * float(summary["review_fraction"])
    logger.info(
        "Tokens needing review: %d (%.1f%%)", needs_review, review_pct
    )

    return summary


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    candidates_path = Path("data/prelabeled/candidates.jsonl")
    corpus_path = Path("data/processed/corpus.jsonl")

    if not candidates_path.exists():
        logger.error("Candidates not found. Run prelabel.py first.")
        sys.exit(1)

    summary = run_label_model(candidates_path, corpus_path)

    print(f"\n{'='*60}")
    print("LABEL MODEL SUMMARY")
    print(f"{'='*60}")
    print(f"Total tokens:          {summary['total_tokens']}")
    print(f"Mean confidence:       {summary['mean_confidence']}")
    print(f"Median confidence:     {summary['median_confidence']}")
    print(f"Mean entropy:          {summary['mean_entropy']}")
    n_review = summary["needs_human_review"]
    r_frac = summary["review_fraction"]
    print(f"Needs human review:    {n_review} ({r_frac:.1%})")
    print("\nPer-LF statistics:")
    lf_stats = summary["lf_stats"]
    for name, stats in lf_stats.items():  # type: ignore[union-attr]
        print(
            f"  {name:25s}  cov={stats['coverage']:.3f}"
            f"  ovl={stats['overlap_rate']:.3f}"
            f"  cnf={stats['conflict_rate']:.3f}"
        )
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
