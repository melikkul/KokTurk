"""Training metrics dashboard.

Reads saved metrics from training rounds and displays a formatted summary.

Usage:
    python src/train/training_dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

MODELS_DIR = Path("models")
GOLD_DIR = Path("data/gold")


def count_gold_tokens(gold_dir: Path = GOLD_DIR) -> int:
    """Count total labeled tokens across seed + batches."""
    total = 0
    for path in gold_dir.rglob("*.jsonl"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                sent = json.loads(line)
                for token in sent.get("tokens", []):
                    if token.get("gold_label") is not None:
                        total += 1
    return total


def load_training_metrics(models_dir: Path = MODELS_DIR) -> list[dict[str, object]]:
    """Load metrics from all training rounds."""
    rounds: list[dict[str, object]] = []

    for model_dir in sorted(models_dir.glob("draft_v*")):
        meta_path = model_dir / "model_meta.json"
        ckpt_path = model_dir / "best_model.pt"

        entry: dict[str, object] = {
            "round": model_dir.name,
            "path": str(model_dir),
        }

        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            entry.update(meta)

        # Try to load metrics from checkpoint (without importing torch)
        metrics_path = model_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                entry["metrics"] = json.load(f)

        entry["has_checkpoint"] = ckpt_path.exists()
        rounds.append(entry)

    return rounds


def print_dashboard() -> None:
    """Print the formatted training dashboard."""
    gold_tokens = count_gold_tokens()
    rounds = load_training_metrics()

    print(f"\n{'═'*75}")
    print("  TRAINING DASHBOARD — Neural Morphological Atomizer")
    print(f"{'═'*75}")
    print(f"  Gold tokens: {gold_tokens}")
    print()

    if not rounds:
        print("  No training rounds found.")
        print("  Submit training: sbatch scripts/truba/submit_train.sh")
    else:
        header = (
            f"  {'Round':12s} {'Checkpoint':12s} {'Val EM':>8s} "
            f"{'Root Acc':>9s} {'Tag F1':>8s} {'Status':15s}"
        )
        print(header)
        print(f"  {'-'*70}")

        for r in rounds:
            name = str(r.get("round", "?"))
            has_ckpt = "Yes" if r.get("has_checkpoint") else "No"
            status = str(r.get("status", "unknown"))

            metrics = r.get("metrics", {})
            if isinstance(metrics, dict):
                em = metrics.get("exact_match", "—")
                root = metrics.get("root_accuracy", "—")
                f1 = metrics.get("tag_f1", "—")
                em_str = f"{em:.1%}" if isinstance(em, float) else str(em)
                root_str = f"{root:.1%}" if isinstance(root, float) else str(root)
                f1_str = f"{f1:.1%}" if isinstance(f1, float) else str(f1)
            else:
                em_str = root_str = f1_str = "—"

            print(
                f"  {name:12s} {has_ckpt:12s} {em_str:>8s} "
                f"{root_str:>9s} {f1_str:>8s} {status:15s}"
            )

    print(f"\n{'═'*75}")

    # Convergence check
    if len(rounds) >= 2:
        metrics_list = [
            r.get("metrics", {}) for r in rounds
            if isinstance(r.get("metrics"), dict)
            and isinstance(r["metrics"].get("exact_match"), float)  # type: ignore[union-attr]
        ]
        if len(metrics_list) >= 2:
            last_em = metrics_list[-1]["exact_match"]  # type: ignore[index]
            prev_em = metrics_list[-2]["exact_match"]  # type: ignore[index]
            delta = abs(last_em - prev_em)  # type: ignore[operator]
            converged = delta < 0.003  # type: ignore[operator]
            print(
                f"  Convergence: Δacc = {delta:.4f} → "  # type: ignore[str-format]
                f"{'CONVERGED' if converged else 'NOT converged'}"
            )
        else:
            print("  Convergence: insufficient data")
    else:
        print("  Convergence: need ≥2 rounds")

    print(f"{'═'*75}\n")


def main() -> None:
    print_dashboard()


if __name__ == "__main__":
    main()
