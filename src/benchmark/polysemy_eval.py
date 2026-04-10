"""Polysemy-aware evaluation for ambiguous Turkish roots."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PolysemyReport:
    per_root_accuracy: dict[str, float] = field(default_factory=dict)
    per_root_support: dict[str, int] = field(default_factory=dict)
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    context_buckets: dict[str, float] = field(default_factory=dict)


def load_polysemous_roots(path: Path | str) -> set[str]:
    """Parse ``configs/eval/polysemous_roots.yaml`` without needing pyyaml."""
    roots: set[str] = set()
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        if line.startswith("polysemous_roots"):
            continue
        head = line.split(":", 1)[0].strip()
        if head and not head.startswith("-") and not head.startswith("["):
            roots.add(head)
    return roots


def _root_of(label: str) -> str:
    parts = label.split()
    return parts[0] if parts else ""


def _sense_of(label: str) -> str:
    """Crude sense signature: first POS tag after the root."""
    parts = label.split()
    for tok in parts[1:]:
        if tok.startswith("+") and not tok[1:].isupper():
            return tok
        if tok.startswith("+"):
            return tok
    return "+?"


def evaluate_polysemy(
    preds: list[str],
    golds: list[str],
    context_lengths: list[int] | None = None,
    polysemous: set[str] | None = None,
) -> PolysemyReport:
    if polysemous is None:
        polysemous = set()
    per_root_correct: Counter = Counter()
    per_root_total: Counter = Counter()
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    ctx_correct: dict[str, list[int]] = defaultdict(list)

    for i, (p, g) in enumerate(zip(preds, golds)):
        r = _root_of(g)
        if polysemous and r not in polysemous:
            continue
        per_root_total[r] += 1
        gs = _sense_of(g)
        ps = _sense_of(p)
        ok = (p == g)
        if ok:
            per_root_correct[r] += 1
        confusion[gs][ps] += 1
        if context_lengths is not None:
            bucket = "short" if context_lengths[i] <= 5 else \
                     "medium" if context_lengths[i] <= 15 else "long"
            ctx_correct[bucket].append(1 if ok else 0)

    report = PolysemyReport()
    for r, total in per_root_total.items():
        report.per_root_accuracy[r] = per_root_correct[r] / total
        report.per_root_support[r] = total
    report.confusion = {k: dict(v) for k, v in confusion.items()}
    for b, arr in ctx_correct.items():
        if arr:
            report.context_buckets[b] = sum(arr) / len(arr)
    return report
