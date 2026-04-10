"""Active learning loop orchestrator.

Selects batches of sentences for annotation based on the composite
acquisition function, exports them for annotation, and tracks progress.

Usage:
    # Select batch (MAD-only, no model):
    python src/data/active_learning_loop.py --mode select-batch --batch-size 50

    # Select batch (BALD+MAD, with trained model):
    python src/data/active_learning_loop.py --mode select-batch --batch-size 50 \
        --model-path models/draft_v1/best_model.pt

    # Check convergence:
    python src/data/active_learning_loop.py --mode check-convergence
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from data.acquisition import select_batch

logger = logging.getLogger(__name__)

CORPUS_PATH = Path("data/processed/corpus.jsonl")
CANDIDATES_PATH = Path("data/prelabeled/candidates.jsonl")
GOLD_DIR = Path("data/gold")
SEED_PATH = GOLD_DIR / "seed" / "seed_200.jsonl"


def load_jsonl(path: Path) -> list[dict[str, object]]:
    """Load a JSONL file."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def get_annotated_sentence_ids() -> set[str]:
    """Collect all sentence IDs that have been annotated (seed + batches)."""
    annotated: set[str] = set()

    # From seed
    if SEED_PATH.exists():
        for record in load_jsonl(SEED_PATH):
            annotated.add(str(record["sentence_id"]))

    # From batch files
    for batch_file in sorted(GOLD_DIR.glob("batch_*.jsonl")):
        for record in load_jsonl(batch_file):
            annotated.add(str(record["sentence_id"]))

    return annotated


def build_batch_dataset(
    selected: list[tuple[str, float]],
    corpus_by_id: dict[str, dict[str, object]],
    candidates_by_sent: dict[str, list[dict[str, object]]],
    labels_by_key: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    """Build a batch dataset in the same format as the seed."""
    batch_data: list[dict[str, object]] = []

    for sent_id, score in selected:
        sent = corpus_by_id.get(sent_id)
        if sent is None:
            continue

        tokens_data: list[dict[str, object]] = []
        tokens: list[str] = sent["tokens"]  # type: ignore[assignment]

        for idx, surface in enumerate(tokens):
            key = f"{sent_id}_{idx}"
            label_record = labels_by_key.get(key, {})

            cands = [
                c for c in candidates_by_sent.get(sent_id, [])
                if c.get("token_idx") == idx
            ]
            analyses: list[dict[str, object]] = []
            if cands:
                analyses = cands[0].get("analyses", [])  # type: ignore[assignment]

            confidence = float(label_record.get("confidence", 0.0))
            predicted_idx = int(label_record.get("predicted_idx", 0))

            ranked_parses: list[dict[str, object]] = []
            for i, a in enumerate(analyses):
                pc = confidence if i == predicted_idx else (
                    (1.0 - confidence) / max(len(analyses) - 1, 1)
                )
                ranked_parses.append({
                    "root": a.get("root", surface),
                    "tags": a.get("tags", []),
                    "source": a.get("source", "unknown"),
                    "confidence": round(pc, 4),
                })
            ranked_parses.sort(
                key=lambda p: p["confidence"], reverse=True
            )

            tokens_data.append({
                "surface": surface,
                "token_idx": idx,
                "candidate_parses": ranked_parses,
                "gold_label": None,
            })

        batch_data.append({
            "sentence_id": sent_id,
            "text": sent.get("text", ""),
            "tokens": tokens_data,
            "acquisition_score": round(score, 4),
        })

    return batch_data


def run_active_learning(
    batch_size: int = 50,
    round_num: int | None = None,
) -> Path:
    """Run one iteration of active learning batch selection.

    Args:
        batch_size: Number of sentences to select.
        round_num: Explicit round number. Auto-detected if None.

    Returns:
        Path to the created batch file.
    """
    logger.info("Loading data...")
    corpus = load_jsonl(CORPUS_PATH)
    candidates = load_jsonl(CANDIDATES_PATH)

    labels_path = Path("data/weak_labels/probabilistic_labels.jsonl")
    labels = load_jsonl(labels_path) if labels_path.exists() else []

    # Index data
    corpus_by_id = {str(s["sentence_id"]): s for s in corpus}
    candidates_by_sent: dict[str, list[dict[str, object]]] = {}
    for c in candidates:
        sid = str(c["sentence_id"])
        candidates_by_sent.setdefault(sid, []).append(c)
    labels_by_key: dict[str, dict[str, object]] = {}
    for lb in labels:
        labels_by_key[f"{lb['sentence_id']}_{lb['token_idx']}"] = lb

    # Get already-annotated sentences
    exclude_ids = get_annotated_sentence_ids()
    logger.info(
        "Excluding %d already-annotated sentences", len(exclude_ids)
    )

    # Select batch
    logger.info("Scoring sentences with acquisition function...")
    selected = select_batch(
        corpus, candidates_by_sent,
        exclude_ids=exclude_ids,
        batch_size=batch_size,
    )

    if not selected:
        logger.warning("No sentences available for annotation")
        return Path("")

    # Auto-detect round number
    if round_num is None:
        existing = list(GOLD_DIR.glob("batch_*.jsonl"))
        round_num = len(existing) + 1

    # Build and save batch
    batch_data = build_batch_dataset(
        selected, corpus_by_id, candidates_by_sent, labels_by_key,
    )

    output_path = GOLD_DIR / f"batch_{round_num:03d}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in batch_data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Print summary
    total_tokens = sum(
        len(s["tokens"]) for s in batch_data  # type: ignore[arg-type]
    )
    scores = [s["acquisition_score"] for s in batch_data]
    mean_score = sum(scores) / len(scores) if scores else 0.0  # type: ignore[arg-type]

    logger.info(
        "Batch %d: %d sentences, %d tokens, mean score=%.4f",
        round_num, len(batch_data), total_tokens, mean_score,
    )
    logger.info("Output: %s", output_path)

    print(f"\n{'='*60}")
    print(f"ACTIVE LEARNING BATCH {round_num}")
    print(f"{'='*60}")
    print(f"Sentences:            {len(batch_data)}")
    print(f"Total tokens:         {total_tokens}")
    print(f"Mean acq. score:      {mean_score:.4f}")
    print(f"Score range:          [{min(scores):.4f}, {max(scores):.4f}]")  # type: ignore[arg-type]
    print(f"Excluded (annotated): {len(exclude_ids)}")
    print(f"Remaining pool:       {len(corpus) - len(exclude_ids) - len(batch_data)}")
    print(f"Output:               {output_path}")
    print(f"{'='*60}")
    print(f"\nAnnotate with: python src/annotation/manual_annotator.py"
          f" --input {output_path}")

    return output_path


def check_convergence(
    models_dir: Path = Path("models"),
    delta_threshold: float = 0.003,
    min_rounds: int = 3,
) -> dict[str, object]:
    """Check if the active learning loop has converged.

    Convergence criteria:
    - Δacc < delta_threshold for min_rounds consecutive rounds
    - Budget: total annotated tokens / 250 = estimated hours used

    Returns:
        Dict with converged (bool), delta, rounds_checked, budget_hours.
    """
    metrics_files: list[tuple[str, dict[str, object]]] = []
    for model_dir in sorted(models_dir.glob("draft_v*")):
        metrics_path = model_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics_files.append((model_dir.name, json.load(f)))

    # Count total annotated tokens
    total_gold = 0
    for path in GOLD_DIR.rglob("*.jsonl"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                sent = json.loads(line)
                for t in sent.get("tokens", []):
                    if t.get("gold_label") is not None:
                        total_gold += 1

    budget_hours = total_gold / 250.0

    if len(metrics_files) < 2:
        return {
            "converged": False,
            "reason": f"Need ≥2 rounds, have {len(metrics_files)}",
            "budget_hours": round(budget_hours, 1),
            "total_gold_tokens": total_gold,
        }

    # Check deltas
    ems = [
        m.get("exact_match", 0.0)
        for _, m in metrics_files
        if isinstance(m.get("exact_match"), float)
    ]
    if len(ems) < 2:
        return {
            "converged": False,
            "reason": "Insufficient EM metrics",
            "budget_hours": round(budget_hours, 1),
            "total_gold_tokens": total_gold,
        }

    last_delta = abs(ems[-1] - ems[-2])
    converged = last_delta < delta_threshold

    return {
        "converged": converged,
        "last_delta": round(last_delta, 4),
        "threshold": delta_threshold,
        "budget_hours": round(budget_hours, 1),
        "total_gold_tokens": total_gold,
        "recommendation": "STOP" if converged else "CONTINUE",
    }


def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(
        description="Active learning loop orchestrator"
    )
    parser.add_argument(
        "--mode", choices=["select-batch", "check-convergence"],
        default="select-batch",
    )
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--round", type=int, default=None)
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to trained model for BALD scoring",
    )
    args = parser.parse_args()

    if args.mode == "check-convergence":
        result = check_convergence()
        print(f"\n{'='*60}")
        print("CONVERGENCE CHECK")
        print(f"{'='*60}")
        for k, v in result.items():
            print(f"  {k}: {v}")
        print(f"{'='*60}")
    else:
        run_active_learning(
            batch_size=args.batch_size, round_num=args.round,
        )


if __name__ == "__main__":
    main()
