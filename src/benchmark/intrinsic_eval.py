"""Intrinsic evaluation metrics for morphological atomization.

Measures: exact match, root accuracy, morpheme-level F1, and
per-ambiguity-class performance.
"""

from __future__ import annotations


def compute_exact_match(
    predictions: list[list[int]],
    gold: list[list[int]],
) -> float:
    """Token-level exact match: predicted tag sequence == gold tag sequence."""
    if not predictions:
        return 0.0
    matches = sum(
        1 for p, g in zip(predictions, gold, strict=True) if p == g
    )
    return matches / len(predictions)


def compute_root_accuracy(
    predictions: list[list[int]],
    gold: list[list[int]],
) -> float:
    """Root accuracy: first predicted token == first gold token."""
    if not predictions:
        return 0.0
    matches = 0
    for p, g in zip(predictions, gold, strict=True):
        pred_root = p[0] if p else -1
        gold_root = g[0] if g else -2
        if pred_root == gold_root:
            matches += 1
    return matches / len(predictions)


def compute_tag_f1(
    predictions: list[list[int]],
    gold: list[list[int]],
) -> dict[str, float]:
    """Morpheme-level F1 (set-based, partial credit)."""
    tp = 0
    fp = 0
    fn = 0

    for p, g in zip(predictions, gold, strict=True):
        pred_tags = set(p[1:]) if len(p) > 1 else set()
        gold_tags = set(g[1:]) if len(g) > 1 else set()
        tp += len(pred_tags & gold_tags)
        fp += len(pred_tags - gold_tags)
        fn += len(gold_tags - pred_tags)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_all_metrics(
    predictions: list[list[int]],
    gold: list[list[int]],
    parse_counts: list[int] | None = None,
) -> dict[str, float]:
    """Compute all intrinsic evaluation metrics."""
    metrics: dict[str, float] = {
        "exact_match": compute_exact_match(predictions, gold),
        "root_accuracy": compute_root_accuracy(predictions, gold),
    }
    metrics.update(compute_tag_f1(predictions, gold))

    if parse_counts is not None:
        ambig_preds = [
            p for p, pc in zip(predictions, parse_counts, strict=True)
            if pc > 1
        ]
        ambig_gold = [
            g for g, pc in zip(gold, parse_counts, strict=True)
            if pc > 1
        ]
        metrics["ambiguous_em"] = compute_exact_match(
            ambig_preds, ambig_gold
        )

    return metrics
