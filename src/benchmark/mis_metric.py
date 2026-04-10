"""Morphological Informativeness Score (MIS) for Turkish surface forms.

MIS quantifies how much morphological information a token contributes to a
corpus — higher scores indicate tokens that are more valuable for training
the atomizer (many parses, high allomorphic variation, complex suffix chains).

Formula::

    MIS(x) = α · H_morph + β · D_canon + γ · C_struct

where:
    H_morph  = morphological parse entropy (log₂ of number of parses)
    D_canon  = mean allomorphic variant count per canonical tag
    C_struct = mean suffix chain length × (1 + derivational tag ratio)

Default weights: α=0.4, β=0.3, γ=0.3

Usage::

    from benchmark.mis_metric import compute_mis

    parses = ["ev +PLU +ABL", "ev +PLU +LOC"]
    score = compute_mis("evlerden", parses)
    # Returns a float > 0; ambiguous tokens score higher
"""
from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonical tag → number of allomorphic surface variants
# (vowel harmony + voicing assimilation produce multiple surface forms)
ALLOMORPH_COUNTS: dict[str, int] = {
    "+LOC": 4,    # -da / -de / -ta / -te
    "+ABL": 4,    # -dan / -den / -tan / -ten
    "+DAT": 4,    # -a / -e / -ya / -ye
    "+GEN": 4,    # -ın / -in / -un / -ün
    "+ACC": 4,    # -ı / -i / -u / -ü
    "+PAST": 8,   # -di/-dı/-dü/-du/-ti/-tı/-tü/-tu
    "+PROG1": 4,  # -(i/ı/u/ü)yor
    "+FUT": 4,    # -acak/-ecek/-yacak/-yecek
    "+COND": 4,   # -sa/-se/-ysa/-yse
    "+PLU": 2,    # -lar / -ler
}

# Tags that introduce derivational morphology (higher structural complexity)
DERIVATIONAL_TAGS: frozenset[str] = frozenset({
    "+CAUS",
    "+PASS",
    "+BECOME",
    "+AGT",
    "+INF",
    "+PASTPART",
    "+FUTPART",
})

# Default MIS component weights (must sum to 1.0)
MIS_ALPHA: float = 0.4   # entropy weight
MIS_BETA: float = 0.3    # allomorphic density weight
MIS_GAMMA: float = 0.3   # structural complexity weight


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_mis(
    token: str,
    parses: list[str],
    alpha: float = MIS_ALPHA,
    beta: float = MIS_BETA,
    gamma: float = MIS_GAMMA,
) -> float:  # noqa: ARG001
    """Compute Morphological Informativeness Score for a token.

    All components are normalized to [0, 1] before the weighted sum,
    guaranteeing the final score is in [0, 1].

    Args:
        token: Surface form of the token (unused in computation; kept for
            API consistency and future surface-level features).
        parses: Canonical parse strings, e.g.
            ``["ev +PLU +ABL", "ev +PLU +LOC"]``.  Empty list → 0.0.
        alpha: Weight for morphological entropy component.
        beta: Weight for canonicalization density component.
        gamma: Weight for structural complexity component.

    Returns:
        MIS score in [0.0, 1.0].
    """
    if not parses or all(not p.strip() for p in parses):
        return 0.0

    h = _morphological_entropy(parses)
    d = _canonicalization_density(parses)
    c = _structural_complexity(parses)

    # Normalize each component to [0, 1]
    # H_morph: max realistic entropy = log2(10) ≈ 3.32 (≤10 parses)
    h_norm = min(h / 3.32, 1.0) if h > 0 else 0.0

    # D_canon: raw values range ~1-8; use 8.0 as practical max
    d_norm = min(d / 8.0, 1.0) if d > 0 else 0.0

    # C_struct: max ~16 (8 suffixes × 2.0); use 16.0 as practical max
    c_norm = min(c / 16.0, 1.0) if c > 0 else 0.0

    score = alpha * h_norm + beta * d_norm + gamma * c_norm
    return min(max(score, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Component helpers
# ---------------------------------------------------------------------------

def _morphological_entropy(parses: list[str]) -> float:
    """H_morph: log₂(number of parses) assuming a uniform parse prior.

    Args:
        parses: List of parse strings (non-empty).

    Returns:
        log₂(len(parses)); 0.0 for a single unambiguous parse.
    """
    return math.log2(len(parses)) if len(parses) > 1 else 0.0


def _canonicalization_density(parses: list[str]) -> float:
    """D_canon: mean allomorphic variant count per tag over all parses.

    For each tag across all parses, look up its entry in ALLOMORPH_COUNTS
    (default 1 for unknown tags), then return the mean over all (parse, tag)
    pairs.  Unknown tags default to 1 (no allomorphic variation known).

    Args:
        parses: List of parse strings (non-empty).

    Returns:
        Mean allomorph count ≥ 1.0; or 1.0 when no tags are present.
    """
    counts: list[float] = []
    for parse in parses:
        tags = parse.split()[1:]  # index 0 is the root
        for tag in tags:
            counts.append(float(ALLOMORPH_COUNTS.get(tag, 1)))
    return sum(counts) / len(counts) if counts else 1.0


def _structural_complexity(parses: list[str]) -> float:
    """C_struct: mean (chain_len × (1 + deriv_ratio)) over all parses.

    chain_len    = number of suffix tags (tokens after the root)
    deriv_ratio  = fraction of those tags that are derivational

    Args:
        parses: List of parse strings (non-empty).

    Returns:
        Mean structural complexity ≥ 0.0.
    """
    scores: list[float] = []
    for parse in parses:
        tags = parse.split()[1:]
        chain_len = len(tags)
        if chain_len == 0:
            scores.append(0.0)
            continue
        n_deriv = sum(1 for t in tags if t in DERIVATIONAL_TAGS)
        deriv_ratio = n_deriv / chain_len
        scores.append(chain_len * (1.0 + deriv_ratio))
    return sum(scores) / len(scores) if scores else 0.0
