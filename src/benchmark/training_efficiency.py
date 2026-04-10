"""Training efficiency metrics for curriculum learning evaluation.

Three metrics that prove TAAC's value even with null EM improvement:

1. **TTT** (Time to Threshold): epochs to reach X% of final EM.
2. **AULC** (Area Under Learning Curve): integral of EM over training.
3. **Relative Efficiency**: ``TTT_baseline / TTT_method``.
"""
from __future__ import annotations


def time_to_threshold(
    em_history: list[float],
    threshold_fraction: float = 0.9,
    final_em: float | None = None,
) -> int:
    """Compute epochs needed to reach *threshold_fraction* of final EM.

    Args:
        em_history: EM values per epoch (1-indexed in output).
        threshold_fraction: Fraction of final performance (e.g. 0.9).
        final_em: If None, uses ``max(em_history)``.

    Returns:
        1-indexed epoch number when threshold first reached, or
        ``len(em_history)`` if never reached.
    """
    if not em_history:
        return 0
    if final_em is None:
        final_em = max(em_history)
    threshold = threshold_fraction * final_em
    for i, em in enumerate(em_history):
        if em >= threshold:
            return i + 1
    return len(em_history)


def area_under_learning_curve(
    em_history: list[float],
    normalize: bool = True,
) -> float:
    """Compute AULC — area under the EM learning curve.

    Higher AULC means the model spent more training time at high performance.

    Args:
        em_history: EM values per epoch.
        normalize: If True, divide by number of epochs (result in [0, max_em]).
    """
    if not em_history:
        return 0.0
    # Trapezoidal integration
    total = sum(
        (em_history[i] + em_history[i + 1]) / 2
        for i in range(len(em_history) - 1)
    )
    if normalize and len(em_history) > 1:
        total /= len(em_history) - 1
    return float(total)


def relative_efficiency(ttt_baseline: int, ttt_method: int) -> float:
    """Compute how much faster *method* reaches threshold vs *baseline*.

    Returns ratio > 1.0 when method is faster.
    """
    if ttt_method == 0:
        return float("inf")
    return ttt_baseline / ttt_method


def compute_all_efficiency_metrics(
    em_history: list[float],
    baseline_em_history: list[float] | None = None,
) -> dict:
    """Compute all efficiency metrics for a single training run."""
    ttt_90 = time_to_threshold(em_history, 0.9)
    ttt_95 = time_to_threshold(em_history, 0.95)
    aulc = area_under_learning_curve(em_history, normalize=True)

    result: dict[str, object] = {
        "ttt_90": ttt_90,
        "ttt_95": ttt_95,
        "aulc": round(aulc, 4),
        "final_em": round(max(em_history), 4) if em_history else 0.0,
    }

    if baseline_em_history:
        b90 = time_to_threshold(baseline_em_history, 0.9)
        b95 = time_to_threshold(baseline_em_history, 0.95)
        result["relative_efficiency_90"] = round(relative_efficiency(b90, ttt_90), 2)
        result["relative_efficiency_95"] = round(relative_efficiency(b95, ttt_95), 2)

    return result


def format_efficiency_table(results: dict[str, dict]) -> str:
    """Format multiple runs' efficiency metrics as a markdown table."""
    header = "| Method | Final EM | TTT@90% | TTT@95% | AULC | Rel.Eff@90% |"
    sep = "|--------|----------|---------|---------|------|-------------|"
    rows = [header, sep]

    for name, m in results.items():
        rel = m.get("relative_efficiency_90", "—")
        if isinstance(rel, (int, float)):
            rel = f"{rel:.1f}x"
        row = (
            f"| {name} | {m['final_em']:.1%} | "
            f"{m['ttt_90']} ep | {m['ttt_95']} ep | "
            f"{m['aulc']:.3f} | {rel} |"
        )
        rows.append(row)

    return "\n".join(rows)
