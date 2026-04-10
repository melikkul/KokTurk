"""Automated error taxonomy and severity analysis.

Classifies every non-exact-match token into a specific error class,
computes a linguistic severity index, generates confusion matrices, and
runs oracle projections to estimate the EM gain from fixing each class.

Works purely on the project's canonical string format:
    "root +TAG1 +TAG2 ..."
Parsing is delegated to :func:`benchmark.stratified_eval._root` and
:func:`benchmark.tag_frequency.extract_tags`.

Diacritic-aware Levenshtein
---------------------------
Standard edit distance treats ``ı`` and ``i`` as a full substitution.
For Turkish that is misleading: an asciified prediction ``kirildi`` vs
the gold ``kırıldı`` would otherwise be classified as SUBSTITUTION when
it is really a diacritic normalization issue. The local ``levenshtein``
routine accepts an optional ``cost_matrix`` mapping character pairs to
substitution costs; ``DIACRITIC_COST_MATRIX`` assigns 0.5 to each
Turkish diacritic pair, which is the default used by
:func:`classify_errors` when ``diacritic_aware=True``.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

from benchmark.stratified_eval import _root
from benchmark.tag_frequency import extract_tags


# ---------------------------------------------------------------------------
# Enums and data types
# ---------------------------------------------------------------------------


class RootErrorType(Enum):
    SUBSTITUTION = auto()
    TRUNCATION = auto()
    EXTENSION = auto()
    HALLUCINATION = auto()


class TagErrorType(Enum):
    MISSING = auto()
    EXTRA = auto()
    WRONG = auto()
    ORDER = auto()


@dataclass(frozen=True, slots=True)
class TokenError:
    surface: str
    gold_root: str
    pred_root: str
    gold_tags: tuple[str, ...]
    pred_tags: tuple[str, ...]
    root_error: RootErrorType | None
    tag_errors: tuple[tuple[TagErrorType, str], ...]
    severity: float
    edit_distance: float


@dataclass
class ErrorReport:
    total: int
    errors: list[TokenError] = field(default_factory=list)
    root_counts: dict[str, int] = field(default_factory=dict)
    tag_counts: dict[str, int] = field(default_factory=dict)
    confusion: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)
    oracle: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Severity weights (module-level constants)
# ---------------------------------------------------------------------------

SEVERITY_WEIGHTS: dict[object, float] = {
    RootErrorType.HALLUCINATION: 1.0,
    RootErrorType.SUBSTITUTION: 1.0,
    RootErrorType.EXTENSION: 0.5,
    RootErrorType.TRUNCATION: 0.3,
    "POS_SUBSTITUTION": 0.9,
    "CASE_ERROR": 0.6,
    "TENSE_ERROR": 0.4,
    TagErrorType.WRONG: 0.4,
    TagErrorType.MISSING: 0.4,
    TagErrorType.EXTRA: 0.3,
    TagErrorType.ORDER: 0.1,
}


# ---------------------------------------------------------------------------
# Diacritic-aware Levenshtein
# ---------------------------------------------------------------------------

_DIACRITIC_PAIRS = [("ı", "i"), ("ö", "o"), ("ü", "u"), ("ç", "c"), ("ş", "s"), ("ğ", "g")]

DIACRITIC_COST_MATRIX: dict[tuple[str, str], float] = {}
for _a, _b in _DIACRITIC_PAIRS:
    for x, y in ((_a, _b), (_b, _a), (_a.upper(), _b.upper()), (_b.upper(), _a.upper())):
        DIACRITIC_COST_MATRIX[(x, y)] = 0.5


def levenshtein(
    a: str, b: str, cost_matrix: dict[tuple[str, str], float] | None = None
) -> float:
    """Weighted Levenshtein distance.

    Insertions and deletions cost 1.0. Substitutions cost 1.0 unless
    ``cost_matrix`` overrides a specific ``(from, to)`` pair.
    """
    if a == b:
        return 0.0
    if not a:
        return float(len(b))
    if not b:
        return float(len(a))
    prev = [float(j) for j in range(len(b) + 1)]
    for i in range(1, len(a) + 1):
        cur = [float(i)] + [0.0] * len(b)
        ca = a[i - 1]
        for j in range(1, len(b) + 1):
            cb = b[j - 1]
            if ca == cb:
                sub_cost = 0.0
            elif cost_matrix is not None and (ca, cb) in cost_matrix:
                sub_cost = cost_matrix[(ca, cb)]
            else:
                sub_cost = 1.0
            cur[j] = min(
                prev[j] + 1.0,            # deletion
                cur[j - 1] + 1.0,         # insertion
                prev[j - 1] + sub_cost,   # substitution
            )
        prev = cur
    return prev[-1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_label(label: str) -> tuple[str, tuple[str, ...]]:
    return _root(label), tuple(extract_tags(label))


def _classify_root_error(gold: str, pred: str, distance: float) -> RootErrorType | None:
    if gold == pred:
        return None
    if pred and gold and pred != gold and pred in gold and len(pred) < len(gold):
        return RootErrorType.TRUNCATION
    if pred and gold and gold != pred and gold in pred and len(gold) < len(pred):
        return RootErrorType.EXTENSION
    # Hallucination heuristic: distance exceeds half the gold length, or
    # the character sets are disjoint.
    if not gold or distance > max(1.0, len(gold) / 2.0):
        return RootErrorType.HALLUCINATION
    if not (set(gold) & set(pred)):
        return RootErrorType.HALLUCINATION
    return RootErrorType.SUBSTITUTION


_CASE_TAGS = {"+ACC", "+DAT", "+LOC", "+ABL", "+GEN", "+INS", "+EQU", "+NOM"}
_TENSE_TAGS = {"+PAST", "+PRES", "+FUT", "+PROG", "+AOR", "+EVID"}
_POS_TAGS = {"+NOUN", "+VERB", "+ADJ", "+ADV", "+PRON", "+NUM", "+CONJ", "+POSTP"}


def _tag_category(tag: str) -> str:
    if tag in _CASE_TAGS:
        return "CASE"
    if tag in _TENSE_TAGS:
        return "TENSE"
    if tag in _POS_TAGS:
        return "POS"
    return "OTHER"


def _diff_tag_sequences(
    gold: tuple[str, ...], pred: tuple[str, ...]
) -> list[tuple[TagErrorType, str]]:
    errors: list[tuple[TagErrorType, str]] = []
    gset = set(gold)
    pset = set(pred)
    for t in gold:
        if t not in pset:
            errors.append((TagErrorType.MISSING, t))
    for t in pred:
        if t not in gset:
            errors.append((TagErrorType.EXTRA, t))
    if not errors and gold != pred:
        errors.append((TagErrorType.ORDER, ",".join(gold)))
    return errors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_severity(
    root_error: RootErrorType | None,
    tag_errors: list[tuple[TagErrorType, str]],
    gold_root: str = "",
    pred_root: str = "",
    gold_tags: tuple[str, ...] = (),
    pred_tags: tuple[str, ...] = (),
) -> float:
    """Severity index: weighted sum of root + tag error contributions."""
    score = 0.0
    if root_error is not None:
        score += SEVERITY_WEIGHTS[root_error]
    for err_type, tag in tag_errors:
        cat = _tag_category(tag.split(",")[0] if err_type is TagErrorType.ORDER else tag)
        if cat == "POS" and err_type in (TagErrorType.WRONG, TagErrorType.EXTRA, TagErrorType.MISSING):
            score += SEVERITY_WEIGHTS["POS_SUBSTITUTION"]
        elif cat == "CASE":
            score += SEVERITY_WEIGHTS["CASE_ERROR"]
        elif cat == "TENSE":
            score += SEVERITY_WEIGHTS["TENSE_ERROR"]
        else:
            score += SEVERITY_WEIGHTS.get(err_type, 0.2)
    return score


def classify_errors(
    gold_labels: list[str],
    pred_labels: list[str],
    surfaces: list[str] | None = None,
    diacritic_aware: bool = True,
) -> list[TokenError]:
    """Classify every non-matching (gold, pred) pair into a TokenError."""
    if len(gold_labels) != len(pred_labels):
        raise ValueError("gold and pred length mismatch")
    cost_matrix = DIACRITIC_COST_MATRIX if diacritic_aware else None
    if surfaces is None:
        surfaces = [""] * len(gold_labels)
    out: list[TokenError] = []
    for surface, g, p in zip(surfaces, gold_labels, pred_labels):
        if g == p:
            continue
        gr, gt = _parse_label(g)
        pr, pt = _parse_label(p)
        dist = levenshtein(gr, pr, cost_matrix=cost_matrix)
        root_err = _classify_root_error(gr, pr, dist)
        tag_errs = _diff_tag_sequences(gt, pt) if root_err is None else []
        severity = compute_severity(root_err, tag_errs, gr, pr, gt, pt)
        out.append(
            TokenError(
                surface=surface,
                gold_root=gr,
                pred_root=pr,
                gold_tags=gt,
                pred_tags=pt,
                root_error=root_err,
                tag_errors=tuple(tag_errs),
                severity=severity,
                edit_distance=dist,
            )
        )
    return out


def generate_confusion_matrix(
    errors: list[TokenError], top_k: int = 20
) -> dict[str, dict[str, dict[str, int]]]:
    """Build category-level confusion counts for case, tense, and POS tags."""
    cats: dict[str, dict[str, dict[str, int]]] = {
        "CASE": defaultdict(lambda: defaultdict(int)),
        "TENSE": defaultdict(lambda: defaultdict(int)),
        "POS": defaultdict(lambda: defaultdict(int)),
    }
    for err in errors:
        gset = set(err.gold_tags)
        pset = set(err.pred_tags)
        for cat, tagset in (("CASE", _CASE_TAGS), ("TENSE", _TENSE_TAGS), ("POS", _POS_TAGS)):
            g_in = gset & tagset
            p_in = pset & tagset
            for gt in g_in:
                for pt in p_in:
                    if gt != pt:
                        cats[cat][gt][pt] += 1
    # Materialize defaultdicts into plain dicts and trim to top_k per row.
    out: dict[str, dict[str, dict[str, int]]] = {}
    for cat, rows in cats.items():
        out[cat] = {}
        for g, cols in rows.items():
            top = dict(Counter(cols).most_common(top_k))
            out[cat][g] = top
    return out


def oracle_projection(errors: list[TokenError], total_tokens: int) -> dict[str, float]:
    """Projected EM if every error of a given class were magically fixed."""
    if total_tokens <= 0:
        return {}
    base_correct = total_tokens - len(errors)
    projections: dict[str, float] = {}
    classes: dict[str, int] = Counter()
    for err in errors:
        if err.root_error is not None:
            classes[f"root_{err.root_error.name.lower()}"] += 1
        else:
            for et, _ in err.tag_errors:
                classes[f"tag_{et.name.lower()}"] += 1
    for klass, n in classes.items():
        projections[klass] = (base_correct + n) / total_tokens
    projections["baseline_em"] = base_correct / total_tokens
    return projections


def generate_error_report(
    gold_path: str | Path,
    pred_path: str | Path,
    output_path: str | Path,
    diacritic_aware: bool = True,
) -> ErrorReport:
    """End-to-end: load labels, classify, render markdown + JSON companion."""
    gold = [ln.strip() for ln in Path(gold_path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    pred = [ln.strip() for ln in Path(pred_path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    errors = classify_errors(gold, pred, diacritic_aware=diacritic_aware)
    total = len(gold)
    root_counts = Counter(e.root_error.name for e in errors if e.root_error is not None)
    tag_counts = Counter(et.name for e in errors for et, _ in e.tag_errors)
    confusion = generate_confusion_matrix(errors)
    oracle = oracle_projection(errors, total)
    report = ErrorReport(
        total=total,
        errors=errors,
        root_counts=dict(root_counts),
        tag_counts=dict(tag_counts),
        confusion=confusion,
        oracle=oracle,
    )
    _render_markdown(report, Path(output_path))
    return report


def _render_markdown(report: ErrorReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Error Analysis Report\n")
    lines.append(f"Total tokens: {report.total}\n")
    lines.append(f"Errors: {len(report.errors)}\n")
    lines.append(f"EM baseline: {report.oracle.get('baseline_em', 0.0):.4f}\n")
    lines.append("\n## Root error distribution\n")
    for k, v in sorted(report.root_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {k}: {v}")
    lines.append("\n## Tag error distribution\n")
    for k, v in sorted(report.tag_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {k}: {v}")
    lines.append("\n## Oracle projections (EM if class fixed)\n")
    for k, v in sorted(report.oracle.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {k}: {v:.4f}")
    lines.append("\n## Confusion (category level)\n")
    for cat, rows in report.confusion.items():
        lines.append(f"### {cat}")
        for g, cols in rows.items():
            lines.append(f"- {g} -> " + ", ".join(f"{p}:{n}" for p, n in cols.items()))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path = path.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "total": report.total,
                "root_counts": report.root_counts,
                "tag_counts": report.tag_counts,
                "oracle": report.oracle,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
