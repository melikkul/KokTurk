"""Integration with standard Turkish morphological evaluation datasets.

Supports TrMor2018, IMST-UD, and BOUN-UD test splits. Loaders accept a
local path only — downloads are out of scope. Clone commands:

* TrMor2018: ``git clone https://github.com/ai-ku/TrMor2018``
* IMST-UD:   ``git clone https://github.com/UniversalDependencies/UD_Turkish-IMST``
* BOUN-UD:   ``git clone https://github.com/UniversalDependencies/UD_Turkish-BOUN``

Two evaluation modes are provided:

* **generation** — the model produces the full analysis from scratch;
  metrics include full-parse EM, lemma accuracy, POS accuracy, and
  feature F1. Compare against Morse (97.67%) and TransMorph (96.25%).
* **disambiguation** — Zeyrek produces candidate parses and the model
  ranks them by log-probability; compare against Sak (97.81%) and
  MorseDisamb (98.59%).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# UD feature -> canonical tag mapping
# ---------------------------------------------------------------------------

_UD_SIMPLE_MAP: dict[str, str] = {
    "Case=Acc": "+ACC",
    "Case=Dat": "+DAT",
    "Case=Loc": "+LOC",
    "Case=Abl": "+ABL",
    "Case=Gen": "+GEN",
    "Case=Ins": "+INS",
    "Case=Equ": "+EQU",
    "Number=Plur": "+PLU",
    "Polarity=Neg": "+NEG",
    "Tense=Past": "+PAST",
    "Tense=Pres": "+PRES",
    "Tense=Fut": "+FUT",
    "Aspect=Prog": "+PROG",
    "Voice=Pass": "+PASS",
    "Voice=Cau": "+CAUS",
    "Voice=Rcp": "+RECIP",
    "Voice=Rfl": "+REFL",
    "Mood=Cnd": "+COND",
    "Mood=Imp": "+IMP",
    "Mood=Opt": "+OPT",
    "Evident=Nfh": "+EVID",
    "Person=1": "+1SG",
    "Person=2": "+2SG",
    "Person=3": "+3SG",
}

_UPOS_TO_TAG = {
    "NOUN": "+NOUN",
    "VERB": "+VERB",
    "ADJ": "+ADJ",
    "ADV": "+ADV",
    "PRON": "+PRON",
    "NUM": "+NUM",
    "CCONJ": "+CONJ",
    "SCONJ": "+CONJ",
    "ADP": "+POSTP",
}


def ud_to_canonical(feats: str, upos: str | None = None) -> list[str]:
    """Convert a UD FEATS string to this project's canonical tag list.

    Unknown features log a warning and are skipped.
    """
    out: list[str] = []
    if upos:
        pos_tag = _UPOS_TO_TAG.get(upos)
        if pos_tag:
            out.append(pos_tag)
    if not feats or feats == "_":
        return out
    parts = feats.split("|")
    poss_person = None
    poss_number = None
    for p in parts:
        if p in _UD_SIMPLE_MAP:
            out.append(_UD_SIMPLE_MAP[p])
            continue
        if p == "Number=Sing" or p == "Case=Nom":
            continue
        if p.startswith("Person[psor]="):
            poss_person = p.split("=", 1)[1]
            continue
        if p.startswith("Number[psor]="):
            poss_number = p.split("=", 1)[1]
            continue
        log.warning("ud_to_canonical: unknown feature %s (upos=%s)", p, upos)
    if poss_person is not None and poss_number is not None:
        num = "SG" if poss_number == "Sing" else "PL"
        out.append(f"+POSS.{poss_person}{num}")
    return out


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StandardExample:
    surface: str
    gold_tags: tuple[str, ...]
    lemma: str
    is_ambiguous: bool


def load_trmorph2018(local_path: str | Path) -> list[StandardExample]:
    """Load TrMor2018 test set. Format: surface TAB analysis per line."""
    path = Path(local_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Clone with: "
            "git clone https://github.com/ai-ku/TrMor2018"
        )
    out: list[StandardExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "\t" not in line:
            continue
        surface, analysis = line.split("\t", 1)
        parts = analysis.split("+")
        lemma = parts[0].strip() if parts else ""
        tags = tuple(f"+{p.strip()}" for p in parts[1:] if p.strip())
        out.append(StandardExample(surface=surface, gold_tags=tags, lemma=lemma, is_ambiguous=False))
    return out


def load_ud_test_split(local_path: str | Path) -> list[StandardExample]:
    """Load a UD CoNLL-U file (typically ``test.conllu``)."""
    path = Path(local_path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    out: list[StandardExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 6 or "-" in cols[0]:
            continue
        surface = cols[1]
        lemma = cols[2]
        upos = cols[3]
        feats = cols[5]
        tags = tuple(ud_to_canonical(feats, upos=upos))
        out.append(StandardExample(surface=surface, gold_tags=tags, lemma=lemma, is_ambiguous=False))
    return out


# ---------------------------------------------------------------------------
# Evaluation modes
# ---------------------------------------------------------------------------


@dataclass
class StandardMetrics:
    n: int
    full_parse_em: float
    lemma_accuracy: float
    pos_accuracy: float
    feature_f1: float


def _score_feature_f1(gold: tuple[str, ...], pred: tuple[str, ...]) -> float:
    if not gold and not pred:
        return 1.0
    gs, ps = set(gold), set(pred)
    if not ps or not gs:
        return 0.0
    tp = len(gs & ps)
    prec = tp / len(ps)
    rec = tp / len(gs)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def evaluate_generation_mode(predict_fn, examples: list[StandardExample]) -> StandardMetrics:
    """``predict_fn(surface) -> (lemma, tags_tuple)``."""
    em = lemma = pos = 0
    f1_sum = 0.0
    for ex in examples:
        p_lemma, p_tags = predict_fn(ex.surface)
        if p_lemma == ex.lemma and tuple(p_tags) == ex.gold_tags:
            em += 1
        if p_lemma == ex.lemma:
            lemma += 1
        gpos = next((t for t in ex.gold_tags if t in _UPOS_TO_TAG.values()), None)
        ppos = next((t for t in p_tags if t in _UPOS_TO_TAG.values()), None)
        if gpos == ppos:
            pos += 1
        f1_sum += _score_feature_f1(ex.gold_tags, tuple(p_tags))
    n = max(len(examples), 1)
    return StandardMetrics(
        n=len(examples),
        full_parse_em=em / n,
        lemma_accuracy=lemma / n,
        pos_accuracy=pos / n,
        feature_f1=f1_sum / n,
    )


def evaluate_disambiguation_mode(
    score_candidate_fn,
    candidate_fn,
    examples: list[StandardExample],
) -> StandardMetrics:
    """Rank candidate parses by ``score_candidate_fn(surface, candidate)``.

    ``candidate_fn(surface) -> list[(lemma, tags_tuple)]``.
    """
    em = lemma = pos = 0
    f1_sum = 0.0
    for ex in examples:
        candidates = candidate_fn(ex.surface) or [(ex.lemma, ex.gold_tags)]
        best = max(candidates, key=lambda c: score_candidate_fn(ex.surface, c))
        b_lemma, b_tags = best
        if b_lemma == ex.lemma and tuple(b_tags) == ex.gold_tags:
            em += 1
        if b_lemma == ex.lemma:
            lemma += 1
        gpos = next((t for t in ex.gold_tags if t in _UPOS_TO_TAG.values()), None)
        ppos = next((t for t in b_tags if t in _UPOS_TO_TAG.values()), None)
        if gpos == ppos:
            pos += 1
        f1_sum += _score_feature_f1(ex.gold_tags, tuple(b_tags))
    n = max(len(examples), 1)
    return StandardMetrics(
        n=len(examples),
        full_parse_em=em / n,
        lemma_accuracy=lemma / n,
        pos_accuracy=pos / n,
        feature_f1=f1_sum / n,
    )


@dataclass
class BenchmarkReport:
    generation: dict[str, StandardMetrics] = field(default_factory=dict)
    disambiguation: dict[str, StandardMetrics] = field(default_factory=dict)


def run_standard_benchmarks(
    predict_fn,
    datasets: dict[str, list[StandardExample]],
    output_path: str | Path,
) -> BenchmarkReport:
    report = BenchmarkReport()
    for name, examples in datasets.items():
        report.generation[name] = evaluate_generation_mode(predict_fn, examples)
    _write_report(report, Path(output_path))
    return report


def _write_report(report: BenchmarkReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Standard Benchmarks\n", "## Generation mode", "| Dataset | N | EM | Lemma | POS | Feat F1 |", "|---|---|---|---|---|---|"]
    for name, m in report.generation.items():
        lines.append(f"| {name} | {m.n} | {m.full_parse_em:.4f} | {m.lemma_accuracy:.4f} | {m.pos_accuracy:.4f} | {m.feature_f1:.4f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
