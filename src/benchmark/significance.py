"""Statistical significance testing for benchmark comparisons.

Implements paired bootstrap resampling (Koehn 2004) with
Holm-Bonferroni correction for multiple comparisons.
"""

from __future__ import annotations

import numpy as np


def paired_bootstrap_test(
    preds_a: list[int],
    preds_b: list[int],
    labels: list[int],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap resampling significance test.

    Tests whether system A is significantly different from system B.

    Args:
        preds_a: Predictions from system A.
        preds_b: Predictions from system B.
        labels: Ground truth labels.
        n_bootstrap: Number of bootstrap iterations.
        seed: Random seed.

    Returns:
        Dict with p_value, mean_diff, ci_lower, ci_upper, cohens_d.
    """
    rng = np.random.RandomState(seed)
    n = len(labels)
    labels_arr = np.array(labels)
    preds_a_arr = np.array(preds_a)
    preds_b_arr = np.array(preds_b)

    correct_a = (preds_a_arr == labels_arr).astype(float)
    correct_b = (preds_b_arr == labels_arr).astype(float)
    observed_diff = correct_a.mean() - correct_b.mean()

    count_a_wins = 0
    diffs: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        diff = correct_a[idx].mean() - correct_b[idx].mean()
        diffs.append(diff)
        if diff <= 0:
            count_a_wins += 1

    p_value = count_a_wins / n_bootstrap
    diffs_arr = np.array(diffs)
    ci_lower = float(np.percentile(diffs_arr, 2.5))
    ci_upper = float(np.percentile(diffs_arr, 97.5))

    # Cohen's d
    pooled_std = np.sqrt(
        (correct_a.std() ** 2 + correct_b.std() ** 2) / 2
    )
    cohens_d = float(observed_diff / max(pooled_std, 1e-8))

    return {
        "p_value": p_value,
        "mean_diff": float(observed_diff),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "cohens_d": cohens_d,
    }


def holm_bonferroni_correction(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of raw p-values from pairwise tests.

    Returns:
        List of corrected p-values (same order as input).
    """
    n = len(p_values)
    if n == 0:
        return []

    # Sort by p-value, keeping track of original index
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [0.0] * n

    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted = p * (n - rank)
        corrected[orig_idx] = min(adjusted, 1.0)

    # Enforce monotonicity
    sorted_corrected = sorted(
        range(n), key=lambda i: p_values[i],
    )
    max_so_far = 0.0
    for idx in sorted_corrected:
        corrected[idx] = max(corrected[idx], max_so_far)
        max_so_far = corrected[idx]

    return corrected


def multi_system_significance_report(
    comparisons: list[tuple[str, list[int], list[int], list[int]]],
    output_path,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Run paired bootstrap on multiple ``(name, labels, sys_a, sys_b)``
    comparisons, apply Holm-Bonferroni, and emit a markdown report.

    This is a pure wrapper: it does not modify any existing function
    signatures or behavior, and delegates all math to
    :func:`paired_bootstrap_test` and :func:`holm_bonferroni_correction`.
    """
    from pathlib import Path

    raw: dict[str, dict] = {}
    for name, labels, sys_a, sys_b in comparisons:
        raw[name] = paired_bootstrap_test(
            sys_a, sys_b, labels, n_bootstrap=n_bootstrap, seed=seed
        )
    names = list(raw.keys())
    p_vals = [raw[n]["p_value"] for n in names]
    corrected = holm_bonferroni_correction(p_vals)
    significant = {n: c < alpha for n, c in zip(names, corrected)}

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Significance Report\n",
        f"Bootstrap iterations: {n_bootstrap}, alpha: {alpha}\n",
        "| Comparison | mean_diff | p | p_corrected | significant |",
        "|---|---|---|---|---|",
    ]
    for n, c in zip(names, corrected):
        r = raw[n]
        lines.append(
            f"| {n} | {r['mean_diff']:+.4f} | {r['p_value']:.4f} | {c:.4f} | {significant[n]} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "raw": raw,
        "corrected": dict(zip(names, corrected)),
        "significant": significant,
    }
