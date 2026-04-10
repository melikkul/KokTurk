"""Stratified evaluation for the kokturk atomizer.

Reports EM / Root Accuracy / Tag F1 stratified by morpheme depth, tag
frequency class, and ambiguity. Computes per-tag P/R/F1 and a confusion
matrix over the top-20 most frequent tags.

Per the approved plan, the report MUST show full-test-set metrics AND
balanced-subset metrics side by side so the 82% EM v2 baseline (reported on
the full test set) remains comparable.

Design notes
------------
- Checkpoint loading uses plain ``torch.load`` and inspects ``.keys()``
  before constructing any model. Vocabs are probed both from the checkpoint
  itself (``char_vocab`` / ``tag_vocab`` keys) and from ``models/vocabs/``.
- Raw model inference is optional. The public API accepts pre-decoded
  ``(prediction, gold)`` pairs, which keeps this module unit-testable without
  loading a real checkpoint or calling ``NeuralBackend``.
- Tag frequency classes are loaded from ``models/benchmark/tag_frequency.json``
  produced by :mod:`benchmark.tag_frequency`.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from benchmark.tag_frequency import extract_tags


@dataclass(frozen=True, slots=True)
class StratumMetrics:
    name: str
    n: int
    exact_match: float
    root_accuracy: float
    tag_f1: float


@dataclass
class StratifiedReport:
    full: list[StratumMetrics] = field(default_factory=list)
    balanced: list[StratumMetrics] = field(default_factory=list)
    per_tag: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion_top20: dict[str, dict[str, int]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core metric helpers
# ---------------------------------------------------------------------------

def _root(label: str) -> str:
    parts = label.split()
    return parts[0] if parts else ""


def morpheme_depth(label: str) -> int:
    """Number of ``+TAG`` tokens in the canonical label (excluding POS)."""
    tags = extract_tags(label)
    # The POS tag (first +X) is not a suffix; subtract it if present.
    return max(len(tags) - 1, 0)


def _depth_bucket(depth: int) -> str:
    """Bucket morpheme depth into reportable strata.

    Cat B Task 4 splits the deep tail (4 / 5 / 6 / 7 / 8+) so the
    stratified report can show how the model degrades at each suffix
    depth instead of collapsing everything ≥5.
    """
    if depth >= 8:
        return "8+"
    return str(depth)


def exact_match_score(preds: list[str], golds: list[str]) -> float:
    if not preds:
        return 0.0
    return sum(1 for p, g in zip(preds, golds) if p == g) / len(preds)


def root_accuracy_score(preds: list[str], golds: list[str]) -> float:
    if not preds:
        return 0.0
    return sum(1 for p, g in zip(preds, golds) if _root(p) == _root(g)) / len(preds)


def tag_f1_score(preds: list[str], golds: list[str]) -> float:
    tp = fp = fn = 0
    for p, g in zip(preds, golds):
        pt = set(extract_tags(p))
        gt = set(extract_tags(g))
        tp += len(pt & gt)
        fp += len(pt - gt)
        fn += len(gt - pt)
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def _metrics(name: str, preds: list[str], golds: list[str]) -> StratumMetrics:
    return StratumMetrics(
        name=name,
        n=len(preds),
        exact_match=exact_match_score(preds, golds),
        root_accuracy=root_accuracy_score(preds, golds),
        tag_f1=tag_f1_score(preds, golds),
    )


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------

def stratify_by_depth(
    preds: list[str], golds: list[str],
) -> list[StratumMetrics]:
    buckets: dict[str, tuple[list[str], list[str]]] = defaultdict(lambda: ([], []))
    for p, g in zip(preds, golds):
        b = _depth_bucket(morpheme_depth(g))
        buckets[b][0].append(p)
        buckets[b][1].append(g)
    out = []
    for b in ("0", "1", "2", "3", "4", "5", "6", "7", "8+"):
        if b in buckets:
            ps, gs = buckets[b]
            out.append(_metrics(f"depth={b}", ps, gs))
    return out


def load_frequency_classes(
    json_path: Path | str,
) -> dict[str, str]:
    """Return ``{tag: class_name}`` from a tag_frequency.json file."""
    payload = json.loads(Path(json_path).read_text())
    return {t["tag"]: t["frequency_class"] for t in payload["tags"]}


def stratify_by_frequency_class(
    preds: list[str],
    golds: list[str],
    classes: dict[str, str],
) -> list[StratumMetrics]:
    """Bucket by the **majority** frequency class of the gold label's tags.

    Tokens whose gold label contains a RARE tag are scored against RARE, etc.
    A token contributes to every class its gold tags belong to.
    """
    buckets: dict[str, tuple[list[str], list[str]]] = defaultdict(lambda: ([], []))
    for p, g in zip(preds, golds):
        gtags = extract_tags(g)
        tok_classes = {classes.get(t, "RARE") for t in gtags}
        for c in tok_classes:
            buckets[c][0].append(p)
            buckets[c][1].append(g)
    out = []
    for c in ("HIGH_FREQ", "MID_FREQ", "LOW_FREQ", "RARE"):
        if c in buckets:
            ps, gs = buckets[c]
            out.append(_metrics(f"class={c}", ps, gs))
    return out


def stratify_by_ambiguity(
    preds: list[str],
    golds: list[str],
    parse_counts: list[int],
) -> list[StratumMetrics]:
    amb_p, amb_g, un_p, un_g = [], [], [], []
    for p, g, pc in zip(preds, golds, parse_counts):
        if pc > 1:
            amb_p.append(p)
            amb_g.append(g)
        else:
            un_p.append(p)
            un_g.append(g)
    out = []
    if un_p:
        out.append(_metrics("ambiguity=unambiguous", un_p, un_g))
    if amb_p:
        out.append(_metrics("ambiguity=ambiguous", amb_p, amb_g))
    return out


# ---------------------------------------------------------------------------
# Per-tag P/R/F1 and confusion matrix
# ---------------------------------------------------------------------------

def per_tag_prf(
    preds: list[str], golds: list[str],
) -> dict[str, dict[str, float]]:
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    for p, g in zip(preds, golds):
        pset = set(extract_tags(p))
        gset = set(extract_tags(g))
        for t in pset & gset:
            tp[t] += 1
        for t in pset - gset:
            fp[t] += 1
        for t in gset - pset:
            fn[t] += 1
    out: dict[str, dict[str, float]] = {}
    all_tags = set(tp) | set(fp) | set(fn)
    for t in sorted(all_tags):
        prec = tp[t] / max(tp[t] + fp[t], 1)
        rec = tp[t] / max(tp[t] + fn[t], 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        out[t] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": tp[t] + fn[t],
        }
    return out


def top20_confusion(
    preds: list[str], golds: list[str], frequency_classes: dict[str, str] | None = None,
) -> dict[str, dict[str, int]]:
    """Build a confusion matrix over the 20 most frequent gold tags.

    For every gold tag in a label, count which *prediction tag* at the same
    position was emitted. If the positions are unaligned (different lengths),
    we fall back to set-difference: any gold tag missing from the prediction
    counts as confused with the predicted tag that replaced it in order.
    """
    counts: Counter[str] = Counter()
    for g in golds:
        counts.update(extract_tags(g))
    top20 = [t for t, _ in counts.most_common(20)]
    top20_set = set(top20)
    matrix: dict[str, dict[str, int]] = {t: defaultdict(int) for t in top20}
    for p, g in zip(preds, golds):
        pt = extract_tags(p)
        gt = extract_tags(g)
        for i, tag in enumerate(gt):
            if tag not in top20_set:
                continue
            pred_tag = pt[i] if i < len(pt) else "<MISSING>"
            matrix[tag][pred_tag] += 1
    return {k: dict(v) for k, v in matrix.items()}


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def build_report(
    preds: list[str],
    golds: list[str],
    parse_counts: list[int] | None = None,
    balanced_indices: Iterable[int] | None = None,
    tag_frequency_json: Path | str | None = None,
) -> StratifiedReport:
    classes = load_frequency_classes(tag_frequency_json) if tag_frequency_json else {}

    def _compute(preds_: list[str], golds_: list[str], pcs: list[int]) -> list[StratumMetrics]:
        strata: list[StratumMetrics] = [_metrics("ALL", preds_, golds_)]
        strata.extend(stratify_by_depth(preds_, golds_))
        if classes:
            strata.extend(stratify_by_frequency_class(preds_, golds_, classes))
        strata.extend(stratify_by_ambiguity(preds_, golds_, pcs))
        return strata

    pcs = parse_counts if parse_counts is not None else [1] * len(preds)

    full = _compute(preds, golds, pcs)

    if balanced_indices is not None:
        idx = list(balanced_indices)
        bp = [preds[i] for i in idx]
        bg = [golds[i] for i in idx]
        bpc = [pcs[i] for i in idx]
        balanced = _compute(bp, bg, bpc) if bp else []
    else:
        balanced = []

    return StratifiedReport(
        full=full,
        balanced=balanced,
        per_tag=per_tag_prf(preds, golds),
        confusion_top20=top20_confusion(preds, golds),
    )


def format_report_markdown(report: StratifiedReport) -> str:
    lines = ["# Stratified Evaluation Report", ""]

    def _table(rows: list[StratumMetrics], title: str) -> list[str]:
        out = [f"## {title}", "", "| Stratum | N | EM | Root Acc | Tag F1 |",
               "| --- | ---: | ---: | ---: | ---: |"]
        for r in rows:
            out.append(
                f"| {r.name} | {r.n} | {r.exact_match:.4f} | "
                f"{r.root_accuracy:.4f} | {r.tag_f1:.4f} |"
            )
        out.append("")
        return out

    lines += _table(report.full, "Full Test Set")
    if report.balanced:
        lines += _table(report.balanced, "Balanced Subset (ambiguous-only)")
    else:
        lines += ["## Balanced Subset", "", "_(none supplied)_", ""]

    lines += ["## Per-Tag P/R/F1", "",
              "| Tag | Precision | Recall | F1 | Support |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for tag, m in sorted(report.per_tag.items(), key=lambda kv: -kv[1]["support"]):
        lines.append(
            f"| `{tag}` | {m['precision']:.3f} | {m['recall']:.3f} | "
            f"{m['f1']:.3f} | {int(m['support'])} |"
        )
    lines.append("")

    lines += ["## Top-20 Tag Confusion Matrix (gold → predicted)", ""]
    for gold, row in report.confusion_top20.items():
        top = sorted(row.items(), key=lambda kv: -kv[1])[:5]
        entries = ", ".join(f"{p}:{c}" for p, c in top)
        lines.append(f"- `{gold}` → {entries}")
    lines.append("")
    return "\n".join(lines)


def write_report(report: StratifiedReport, path: Path | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(format_report_markdown(report))


# ---------------------------------------------------------------------------
# Optional checkpoint-based entry point
# ---------------------------------------------------------------------------

def inspect_checkpoint(ckpt_path: Path | str) -> dict:
    """Return a summary of the checkpoint structure (keys + shapes).

    This is a safety helper — the plan mandates inspecting ``.keys()`` before
    instantiating any model, because training embeds vocabs in the checkpoint
    and those layouts change between revisions.
    """
    import torch
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        return {k: (type(v).__name__) for k, v in ckpt.items()}
    return {"root": type(ckpt).__name__}
