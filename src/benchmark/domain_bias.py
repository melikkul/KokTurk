"""Domain bias measurement for morphological analyzer.

Stratifies test data by formality/domain and measures accuracy disparity.
Reports Demographic Parity Difference (DPD) across domains.

Usage::

    PYTHONPATH=src python -m benchmark.domain_bias --test-path data/gold/test.tsv

"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Data types
# ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DomainBiasReport:
    """Results of domain bias measurement.

    Attributes:
        domain_results: ``{domain: {"em": float, "root_acc": float, "tag_f1": float, "n": int}}``
        dpd: Demographic Parity Difference (max EM - min EM across domains with n >= 10).
        tpr_disparity: Per-domain EM delta from the mean EM.
    """

    domain_results: dict[str, dict[str, float]]
    dpd: float
    tpr_disparity: dict[str, float]


# ------------------------------------------------------------------
# Domain classification
# ------------------------------------------------------------------

_SOCIAL_MEDIA_RE = re.compile(r"[#@][\w]+|https?://")
_EMOJI_RE = re.compile(
    "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF]"
)


def classify_domain(text: str, source: str | None = None) -> str:
    """Heuristic domain classifier for Turkish text.

    Checks *source* metadata first, then falls back to text heuristics.

    Returns:
        One of ``"formal"``, ``"informal"``, ``"news"``, ``"social_media"``.
    """
    if source is not None:
        src = source.lower()
        if "bounti" in src or "tweet" in src:
            return "social_media"
        if "trendyol" in src or "review" in src:
            return "informal"
        if "ttc" in src or "news" in src:
            return "news"
        if "boun" in src or "imst" in src:
            return "formal"

    # Text-based fallbacks
    if _SOCIAL_MEDIA_RE.search(text) or _EMOJI_RE.search(text):
        return "social_media"

    return "formal"


# ------------------------------------------------------------------
# Metric helpers (inline to avoid hard dependency on stratified_eval)
# ------------------------------------------------------------------


def _root(label: str) -> str:
    """Extract root from a canonical label like ``'ev +PLU +ABL'``."""
    return label.split()[0] if label.strip() else ""


def _tags(label: str) -> list[str]:
    """Extract ordered tag list from a canonical label."""
    parts = label.split()
    return [p for p in parts[1:] if p.startswith("+")]


def _exact_match(pred: str, gold: str) -> float:
    return 1.0 if pred.strip() == gold.strip() else 0.0


def _root_match(pred: str, gold: str) -> float:
    return 1.0 if _root(pred) == _root(gold) else 0.0


def _tag_f1(pred: str, gold: str) -> float:
    pred_tags = set(_tags(pred))
    gold_tags = set(_tags(gold))
    if not gold_tags and not pred_tags:
        return 1.0
    if not gold_tags or not pred_tags:
        return 0.0
    tp = len(pred_tags & gold_tags)
    prec = tp / len(pred_tags) if pred_tags else 0.0
    rec = tp / len(gold_tags) if gold_tags else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


# ------------------------------------------------------------------
# Core measurement
# ------------------------------------------------------------------


def measure_domain_bias(
    test_data: list[dict[str, str]],
    predictions: list[str] | None = None,
) -> DomainBiasReport:
    """Stratify test data by domain and compute per-domain metrics.

    Args:
        test_data: List of dicts with ``"surface"``, ``"label"``, and
            optionally ``"source"`` and ``"prediction"`` keys.
        predictions: If provided, used instead of ``"prediction"`` key.

    Returns:
        :class:`DomainBiasReport` with per-domain metrics, DPD, and
        TPR disparity.
    """
    # Group by domain
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for i, item in enumerate(test_data):
        gold = item.get("label", "")
        pred = predictions[i] if predictions is not None else item.get("prediction", "")
        source = item.get("source")
        text = item.get("surface", item.get("text", ""))
        domain = classify_domain(text, source)
        groups[domain].append((pred, gold))

    # Compute per-domain metrics
    domain_results: dict[str, dict[str, float]] = {}
    for domain, pairs in sorted(groups.items()):
        n = len(pairs)
        em = sum(_exact_match(p, g) for p, g in pairs) / n if n else 0.0
        root_acc = sum(_root_match(p, g) for p, g in pairs) / n if n else 0.0
        tag_f1 = sum(_tag_f1(p, g) for p, g in pairs) / n if n else 0.0
        domain_results[domain] = {
            "em": em,
            "root_acc": root_acc,
            "tag_f1": tag_f1,
            "n": float(n),
        }

    # DPD: max - min EM across domains with n >= 10
    eligible = {d: r["em"] for d, r in domain_results.items() if r["n"] >= 10}
    if len(eligible) >= 2:
        dpd = max(eligible.values()) - min(eligible.values())
    else:
        dpd = 0.0

    # TPR disparity: per-domain delta from mean EM
    if eligible:
        mean_em = sum(eligible.values()) / len(eligible)
        tpr_disparity = {d: em - mean_em for d, em in eligible.items()}
    else:
        tpr_disparity = {}

    return DomainBiasReport(
        domain_results=domain_results,
        dpd=dpd,
        tpr_disparity=tpr_disparity,
    )


# ------------------------------------------------------------------
# Report generation
# ------------------------------------------------------------------


def generate_bias_summary(report: DomainBiasReport) -> str:
    """Generate a one-paragraph summary suitable for the model card."""
    eligible = {
        d: r for d, r in report.domain_results.items() if r["n"] >= 10
    }
    if not eligible:
        return "Insufficient domain-stratified data for bias analysis."

    n_domains = len(eligible)
    best_domain = max(eligible, key=lambda d: eligible[d]["em"])
    worst_domain = min(eligible, key=lambda d: eligible[d]["em"])
    best_em = eligible[best_domain]["em"]
    worst_em = eligible[worst_domain]["em"]

    return (
        f"Domain bias analysis across {n_domains} domains shows a DPD of "
        f"{report.dpd:.4f}. The highest-performing domain is {best_domain} "
        f"(EM={best_em:.2%}) and the lowest is {worst_domain} "
        f"(EM={worst_em:.2%}), a gap of {report.dpd:.2%} percentage points."
    )


def _write_report(report: DomainBiasReport, output_path: Path) -> None:
    """Write a full markdown report."""
    lines: list[str] = []
    lines.append("# Domain Bias Report")
    lines.append("")
    lines.append("## Per-Domain Results")
    lines.append("")
    lines.append("| Domain | N | EM | Root Acc | Tag F1 |")
    lines.append("|--------|---|----|----------|--------|")
    for domain, r in sorted(report.domain_results.items()):
        lines.append(
            f"| {domain} | {int(r['n'])} | {r['em']:.4f} | "
            f"{r['root_acc']:.4f} | {r['tag_f1']:.4f} |"
        )
    lines.append("")
    lines.append(f"## DPD (Demographic Parity Difference): {report.dpd:.4f}")
    lines.append("")
    if report.tpr_disparity:
        lines.append("## TPR Disparity (EM delta from mean)")
        lines.append("")
        for domain, delta in sorted(report.tpr_disparity.items()):
            lines.append(f"- {domain}: {delta:+.4f}")
        lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(generate_bias_summary(report))
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Domain bias report written to %s", output_path)


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------


def run_domain_bias(
    test_path: str | Path = "data/gold/test.tsv",
    pred_path: str | Path | None = None,
    output_path: str | Path = "models/benchmark/domain_bias_report.md",
) -> DomainBiasReport:
    """High-level runner: load data, measure bias, write report."""
    test_path = Path(test_path)
    test_data: list[dict[str, str]] = []

    if test_path.suffix == ".jsonl":
        for line in test_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                test_data.append(json.loads(line))
    else:
        # TSV: surface<tab>label
        for line in test_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                test_data.append({"surface": parts[0], "label": parts[1]})

    predictions: list[str] | None = None
    if pred_path is not None:
        pred_path = Path(pred_path)
        predictions = [
            line.strip()
            for line in pred_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    report = measure_domain_bias(test_data, predictions)
    _write_report(report, Path(output_path))
    return report


def main() -> None:  # pragma: no cover
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Domain bias measurement for morphological analyzer"
    )
    parser.add_argument(
        "--test-path", type=Path, default=Path("data/gold/test.tsv"),
    )
    parser.add_argument("--pred-path", type=Path, default=None)
    parser.add_argument(
        "--output", type=Path,
        default=Path("models/benchmark/domain_bias_report.md"),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    report = run_domain_bias(args.test_path, args.pred_path, args.output)
    print(f"DPD: {report.dpd:.4f}")
    for domain, delta in sorted(report.tpr_disparity.items()):
        print(f"  {domain}: {delta:+.4f}")


if __name__ == "__main__":
    main()
