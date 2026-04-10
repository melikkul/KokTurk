"""Generate the final benchmark report from saved results.

Usage:
    PYTHONPATH=src python src/benchmark/generate_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS_PATH = Path("models/benchmark/classification_results.json")
REPORT_PATH = Path("models/benchmark/BENCHMARK_REPORT.md")


def generate_report() -> None:
    """Generate markdown benchmark report from saved results."""
    results = {}
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            results = json.load(f)

    intrinsic = results.get("intrinsic", {})
    classification = results.get("classification", {})
    efficiency = results.get("efficiency", {})
    bpe_count = results.get("bpe_failure_count", 0)

    lines: list[str] = []
    lines.append("# Benchmark Report: Neural Morphological Atomization for Turkish\n")

    # Section 1
    lines.append("## 1. Morphological Analysis (Intrinsic Evaluation)\n")
    lines.append("### Model: GRU Seq2Seq (2.25M params, 8.6MB)\n")
    lines.append("| Metric | Score |")
    lines.append("|--------|-------|")
    for key, label in [
        ("exact_match", "Exact Match"),
        ("root_accuracy", "Root Accuracy"),
        ("f1", "Tag F1"),
        ("precision", "Tag Precision"),
        ("recall", "Tag Recall"),
    ]:
        val = intrinsic.get(key, 0)
        lines.append(f"| {label} | {val:.1%} |")

    root_err = int(intrinsic.get("root_errors", 0))
    tag_err = int(intrinsic.get("tag_errors", 0))
    total = int(intrinsic.get("total_test_tokens", 0))
    lines.append(
        f"\nTest split: {total} tokens (never used in training)."
    )
    lines.append(
        f"Error breakdown: {root_err} root errors, {tag_err} tag errors.\n"
    )
    lines.append(
        "Training corpus: 80,537 tokens "
        "(2,496 gold + 61,516 silver-auto + 16,525 silver-agreed)\n"
    )

    # Section 2
    lines.append("## 2. Text Classification (TTC-3600, 5-fold CV)\n")
    lines.append("### Pre-computed TF-IDF features (UCI dataset)\n")
    lines.append("| Method | Macro-F1 |")
    lines.append("|--------|----------|")
    for method in [
        "ttc3600_original_tfidf",
        "ttc3600_zemberek_tfidf",
    ]:
        res = classification.get(method, {})
        if isinstance(res, dict) and "macro_f1_mean" in res:
            note = res.get("note", "")
            lines.append(
                f"| {note} | "
                f"{res['macro_f1_mean']:.3f} ± {res['macro_f1_std']:.3f} |"
            )

    lines.append(
        "\n*Note: UCI distributes pre-computed TF-IDF features only. "
        "Raw text comparison (TextCNN, BERTurk, FastText) pending "
        "TTC-3600 raw text download (see scripts/download_ttc3600_raw.sh).*\n"
    )

    # Section 3
    lines.append("## 3. Efficiency Comparison\n")
    lines.append("| System | Params | Size | Speed |")
    lines.append("|--------|--------|------|-------|")
    zeyrek = efficiency.get("zeyrek", {})
    gru = efficiency.get("gru_atomizer", {})
    if zeyrek:
        lines.append(
            f"| Zeyrek (rule-based) | — | — | "
            f"{zeyrek.get('tokens_per_sec', 0):.0f} tok/s |"
        )
    if gru:
        lines.append(
            f"| GRU Atomizer | "
            f"{int(gru.get('params', 0)):,} | "
            f"{gru.get('model_size_mb', 0):.1f} MB | — |"
        )
    lines.append(
        "| BERTurk | 110M | ~440 MB | *not tested (download restricted)* |"
    )
    lines.append("")

    # Section 4 — BPE failures inline
    lines.append("## 4. BPE Failure Analysis\n")
    lines.append(f"{bpe_count} cases documented where BPE/WordPiece fragments "
                 "Turkish morphemes while our atomizer preserves them.\n")
    lines.append(
        "Key failure modes:\n"
        "1. **Over-segmentation**: BPE splits roots (kitap → kit + ##ap)\n"
        "2. **Cross-morpheme fragments**: BPE merges root end with suffix start\n"
        "3. **Allomorph inconsistency**: -ler/-lar treated as different tokens\n"
        "4. **Suffix chain confusion**: no morpheme boundary detection\n"
        "5. **OOV agglutination**: BPE produces [UNK] for novel suffix combos\n"
    )
    lines.append(
        "See `src/benchmark/linguistic_analysis.py` for all examples.\n"
    )

    # Section 5
    lines.append("## 5. Limitations\n")
    lines.append(
        "- TRMorph (foma FST) unavailable — single analyzer "
        "(Zeyrek) only\n"
        "- Gold annotations: 2,496 human-verified out of 80,537 total "
        "(3.1%)\n"
        "- GRU Seq2Seq is Phase 2 draft — Phase 3 Char Transformer "
        "expected to exceed 90% EM\n"
        "- TTC-3600 raw text unavailable — TextCNN/BERTurk/FastText "
        "comparison pending\n"
        "- Model trained on CPU — GPU training may yield better "
        "convergence\n"
    )

    report = "\n".join(lines)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    generate_report()
