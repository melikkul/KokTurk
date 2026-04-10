"""Compound-verb and -(s)I ambiguity evaluation (Cat B Task 4).

Two strata that the existing stratified eval does not cover:

1. **Fused LVCs.** Tokens whose surface contains a known fused
   light-verb stem (``reddet``, ``hisset``, ``kaybol``, …). The metric
   reports per-stem accuracy of recovering the underlying *nominal*
   root, with vs. without the analyzer's ``decompose_lvc`` flag.

2. **-(s)I ambiguity.** Tokens whose Zeyrek candidates differ on
   ``+POSS.3SG`` vs. ``+ACC``. Optionally also reports a *secondary*
   compound-context bucket: tokens whose immediately preceding token
   carries ``+GEN`` (a syntactic indicator of a noun-noun compound).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from kokturk.core.compound_lexicon import FUSED_LVC_TABLE, decompose_fused_lvc

# Hand-picked fused-LVC verbal stems we expect models to handle.  All
# entries here are present in :data:`FUSED_LVC_TABLE`.
COMPOUND_VERB_ROOTS: tuple[str, ...] = (
    "reddet", "hisset", "zannet", "affet", "sabret", "kaybol",
    "emret", "devret", "hükmet", "mahvet", "bahset", "seyret",
    "hallet", "naklet",
)


@dataclass(frozen=True, slots=True)
class CompoundReport:
    """Per-stem and overall compound-handling metrics."""

    n_total: int
    n_correct: int
    accuracy: float
    per_stem: dict[str, tuple[int, int]]  # stem -> (correct, total)


@dataclass(frozen=True, slots=True)
class SIAmbiguityReport:
    """Metrics for the -(s)I ambiguity stratum."""

    n_primary: int
    em_primary: float
    n_compound_context: int
    em_compound_context: float


def _root_of(label: str) -> str:
    return label.split(" ", 1)[0] if label else ""


def evaluate_compound_handling(
    surfaces: Sequence[str],
    predictions: Sequence[str],
    golds: Sequence[str],
) -> CompoundReport:
    """Compute compound-LVC handling accuracy.

    A token is counted as a compound-test instance iff its surface form
    starts with a known fused-LVC stem.  A prediction is correct iff its
    root equals the gold root (which, in a properly decomposed gold
    label, is the underlying nominal — e.g. ``ret`` for ``reddetti``).

    Args:
        surfaces: token surface forms (e.g. ``["reddetti", "geldi", ...]``).
        predictions: model output labels (``"root +TAG +TAG"``).
        golds: gold labels in the same format.
    """
    n_total = 0
    n_correct = 0
    per_stem: dict[str, list[int]] = {}

    for surface, pred, gold in zip(surfaces, predictions, golds):
        decomposition = decompose_fused_lvc(surface)
        if decomposition is None:
            continue
        nominal, light_verb, _ = decomposition
        # Use the matched fused stem as the bucket key.
        lower = surface.lower()
        stem_key = next(
            (
                stem
                for stem in sorted(FUSED_LVC_TABLE, key=len, reverse=True)
                if lower.startswith(stem)
            ),
            f"{nominal}+{light_verb}",
        )
        per_stem.setdefault(stem_key, [0, 0])
        per_stem[stem_key][1] += 1
        n_total += 1
        if _root_of(pred) == _root_of(gold):
            per_stem[stem_key][0] += 1
            n_correct += 1

    return CompoundReport(
        n_total=n_total,
        n_correct=n_correct,
        accuracy=(n_correct / n_total) if n_total else 0.0,
        per_stem={k: (v[0], v[1]) for k, v in per_stem.items()},
    )


def evaluate_sI_ambiguity(
    tokens: Sequence[str],
    predictions: Sequence[str],
    golds: Sequence[str],
    candidate_tag_sets: Sequence[Sequence[set[str]]] | None = None,
    preceding_tags: Sequence[set[str]] | None = None,
) -> SIAmbiguityReport:
    """Compute -(s)I ambiguity metrics.

    Args:
        tokens: surface forms (unused except for length alignment).
        predictions: model output labels.
        golds: gold labels.
        candidate_tag_sets: per-token list of Zeyrek candidate tag sets.
            A token enters the *primary* stratum iff at least two of its
            candidate sets differ on ``+POSS.3SG`` vs ``+ACC``.
        preceding_tags: per-token tag set of the immediately preceding
            token (or empty set at sentence-initial position).  A token
            enters the *compound-context* stratum iff its preceding tag
            set contains ``+GEN``.
    """
    primary_pred: list[str] = []
    primary_gold: list[str] = []
    if candidate_tag_sets is not None:
        for pred, gold, candidates in zip(predictions, golds, candidate_tag_sets):
            has_poss = any("+POSS.3SG" in c for c in candidates)
            has_acc = any("+ACC" in c for c in candidates)
            if has_poss and has_acc:
                primary_pred.append(pred)
                primary_gold.append(gold)

    compound_pred: list[str] = []
    compound_gold: list[str] = []
    if preceding_tags is not None:
        for pred, gold, prev in zip(predictions, golds, preceding_tags):
            if "+GEN" in prev:
                compound_pred.append(pred)
                compound_gold.append(gold)

    def em(p: list[str], g: list[str]) -> float:
        if not p:
            return 0.0
        return sum(1 for a, b in zip(p, g) if a == b) / len(p)

    return SIAmbiguityReport(
        n_primary=len(primary_pred),
        em_primary=em(primary_pred, primary_gold),
        n_compound_context=len(compound_pred),
        em_compound_context=em(compound_pred, compound_gold),
    )
