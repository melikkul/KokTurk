"""Weighted exact-match metric reflecting linguistic severity.

S(w) = alpha * delta(lemma) + beta * delta(POS) + gamma * EMMA(deriv) + lam * EMMA(infl)

EMMA is the Evaluation Metric for Morphological Analysis: maximum
matching in a weighted bipartite graph between reference and predicted
morphemes. We use :func:`scipy.optimize.linear_sum_assignment` when
SciPy is available and fall back to a greedy matcher otherwise so that
the metric works in minimal test environments.
"""

from __future__ import annotations

from benchmark.stratified_eval import _root
from benchmark.tag_frequency import extract_tags

ALPHA_LEMMA = 0.50
BETA_POS = 0.20
GAMMA_DERIV = 0.15
LAMBDA_INFL = 0.15

_DERIV_TAGS = {"+CAUS", "+PASS", "+RECIP", "+REFL", "+LVC.ET", "+LVC.OL", "+DER"}
_POS_TAGS = {"+NOUN", "+VERB", "+ADJ", "+ADV", "+PRON", "+NUM", "+CONJ", "+POSTP"}


def _split_parts(label: str) -> tuple[str, str, list[str], list[str]]:
    root = _root(label)
    tags = extract_tags(label)
    pos = next((t for t in tags if t in _POS_TAGS), "")
    deriv = [t for t in tags if t in _DERIV_TAGS]
    infl = [t for t in tags if t not in _DERIV_TAGS and t not in _POS_TAGS]
    return root, pos, deriv, infl


def _morpheme_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    # Prefix / suffix family overlap, e.g. "+POSS.3SG" vs "+POSS.1SG".
    if "." in a and "." in b:
        fa = a.split(".")[0]
        fb = b.split(".")[0]
        if fa == fb:
            return 0.5
    return 0.0


def compute_emma_f1(gold: list[str], pred: list[str]) -> float:
    """EMMA-style F1 via bipartite matching on morpheme similarity."""
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        cost = np.zeros((len(gold), len(pred)), dtype=float)
        for i, g in enumerate(gold):
            for j, p in enumerate(pred):
                cost[i, j] = -_morpheme_similarity(g, p)
        r, c = linear_sum_assignment(cost)
        matched = float(-cost[r, c].sum())
    except Exception:
        used_p: set[int] = set()
        matched = 0.0
        for g in gold:
            best_j = -1
            best = 0.0
            for j, p in enumerate(pred):
                if j in used_p:
                    continue
                s = _morpheme_similarity(g, p)
                if s > best:
                    best = s
                    best_j = j
            if best_j >= 0:
                matched += best
                used_p.add(best_j)
    precision = matched / len(pred)
    recall = matched / len(gold)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def weighted_exact_match(
    gold_root: str,
    pred_root: str,
    gold_pos: str,
    pred_pos: str,
    gold_deriv: list[str],
    pred_deriv: list[str],
    gold_infl: list[str],
    pred_infl: list[str],
    alpha: float = ALPHA_LEMMA,
    beta: float = BETA_POS,
    gamma: float = GAMMA_DERIV,
    lam: float = LAMBDA_INFL,
) -> float:
    lemma_ok = 1.0 if gold_root == pred_root else 0.0
    pos_ok = 1.0 if gold_pos == pred_pos else 0.0
    deriv_f1 = compute_emma_f1(gold_deriv, pred_deriv)
    infl_f1 = compute_emma_f1(gold_infl, pred_infl)
    return alpha * lemma_ok + beta * pos_ok + gamma * deriv_f1 + lam * infl_f1


def score_pair(gold_label: str, pred_label: str) -> float:
    gr, gp, gd, gi = _split_parts(gold_label)
    pr, pp, pd, pi = _split_parts(pred_label)
    # When gold has no POS tag, treat both sides as matching on POS so
    # label sets without explicit POS tags still score correctly.
    if not gp and not pp:
        gp = pp = "+NONE"
    return weighted_exact_match(gr, pr, gp, pp, gd, pd, gi, pi)


def corpus_weighted_em(gold_labels: list[str], pred_labels: list[str]) -> float:
    if not gold_labels:
        return 0.0
    total = 0.0
    for g, p in zip(gold_labels, pred_labels):
        total += score_pair(g, p)
    return total / len(gold_labels)
