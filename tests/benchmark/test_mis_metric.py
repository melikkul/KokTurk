"""Tests for MIS (Morphological Informativeness Score).

Covers:
1. Empty parses → 0.0
2. Single parse → H_morph = 0 (no entropy)
3. Entropy increases with number of parses
4. Allomorphic density uses ALLOMORPH_COUNTS (+LOC → 4)
5. Structural complexity grows with suffix chain length
6. Derivational tags increase structural complexity
7. compute_mis returns a float
8. MIS_ALPHA + MIS_BETA + MIS_GAMMA == 1.0
"""
from __future__ import annotations

import math

import pytest

from benchmark.mis_metric import (
    ALLOMORPH_COUNTS,
    DERIVATIONAL_TAGS,
    MIS_ALPHA,
    MIS_BETA,
    MIS_GAMMA,
    _canonicalization_density,
    _morphological_entropy,
    _structural_complexity,
    compute_mis,
)


# ---------------------------------------------------------------------------
# 1. Empty parses → 0.0
# ---------------------------------------------------------------------------

class TestEmptyParses:
    def test_compute_mis_empty_list(self):
        assert compute_mis("ev", []) == 0.0

    def test_entropy_empty_returns_zero(self):
        # _morphological_entropy only called with non-empty from compute_mis
        # but we test the component directly too
        # Passing empty list to entropy: edge case — should not error
        # Actually compute_mis guards; test entropy with 1 parse
        assert _morphological_entropy(["ev"]) == 0.0

    def test_density_empty_tags_returns_one(self):
        """Parse with no tags (root only) → density defaults to 1.0."""
        result = _canonicalization_density(["ev"])
        assert result == pytest.approx(1.0)

    def test_complexity_root_only_returns_zero(self):
        """Parse with no suffix tags → chain_len=0 → complexity=0.0."""
        result = _structural_complexity(["ev"])
        assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. Single parse → zero entropy
# ---------------------------------------------------------------------------

class TestSingleParse:
    def test_single_parse_entropy_zero(self):
        assert _morphological_entropy(["ev +PLU"]) == pytest.approx(0.0)

    def test_mis_single_unambiguous_parse(self):
        score = compute_mis("evler", ["ev +PLU"])
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        # With 1 parse, entropy=0, so MIS depends only on D_norm and C_norm
        # D_canon for +PLU = 2.0, normalized = 2.0/8.0 = 0.25
        # C_struct for single non-deriv tag = 1.0, normalized = 1.0/16.0 = 0.0625
        expected = MIS_BETA * (2.0 / 8.0) + MIS_GAMMA * (1.0 / 16.0)
        assert score == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 3. Entropy increases with number of parses
# ---------------------------------------------------------------------------

class TestEntropyIncreasesWithAmbiguity:
    def test_one_parse_lower_than_two(self):
        h1 = _morphological_entropy(["ev +PLU"])
        h2 = _morphological_entropy(["ev +PLU", "ev +PLU +DAT"])
        assert h1 < h2

    def test_two_parses_lower_than_four(self):
        parses_2 = ["ev +PLU", "ev +PLU +DAT"]
        parses_4 = ["ev +PLU", "ev +PLU +DAT", "ev +PLU +LOC", "ev +PLU +ABL"]
        assert _morphological_entropy(parses_2) < _morphological_entropy(parses_4)

    def test_entropy_equals_log2_of_n_parses(self):
        for n in [1, 2, 4, 8]:
            parses = [f"root_{i} +PLU" for i in range(n)]
            expected = math.log2(n) if n > 1 else 0.0
            assert _morphological_entropy(parses) == pytest.approx(expected)

    def test_mis_increases_with_more_parses(self):
        """Higher ambiguity → higher MIS."""
        score1 = compute_mis("ev", ["ev +PLU"])
        score4 = compute_mis("ev", ["ev +PLU", "ev +DAT", "ev +LOC", "ev +ABL"])
        assert score4 > score1


# ---------------------------------------------------------------------------
# 4. Allomorphic density uses ALLOMORPH_COUNTS
# ---------------------------------------------------------------------------

class TestCanonicalizationDensity:
    def test_loc_tag_uses_count_four(self):
        # +LOC has allomorph count 4
        density = _canonicalization_density(["ev +LOC"])
        assert density == pytest.approx(ALLOMORPH_COUNTS["+LOC"])  # 4.0

    def test_plu_tag_uses_count_two(self):
        density = _canonicalization_density(["ev +PLU"])
        assert density == pytest.approx(ALLOMORPH_COUNTS["+PLU"])  # 2.0

    def test_unknown_tag_defaults_to_one(self):
        density = _canonicalization_density(["ev +UNKNOWNTAG"])
        assert density == pytest.approx(1.0)

    def test_mean_over_multiple_tags(self):
        # +LOC=4, +PLU=2 → mean=3.0
        density = _canonicalization_density(["ev +LOC +PLU"])
        assert density == pytest.approx(3.0)

    def test_mean_over_multiple_parses(self):
        # parse1: +LOC (4)   parse2: +PLU (2)  → all tags: [4, 2] → mean=3.0
        density = _canonicalization_density(["ev +LOC", "ev +PLU"])
        assert density == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 5. Structural complexity grows with suffix chain length
# ---------------------------------------------------------------------------

class TestStructuralComplexity:
    def test_longer_chain_higher_complexity(self):
        c1 = _structural_complexity(["ev +PLU"])
        c2 = _structural_complexity(["ev +PLU +LOC"])
        c3 = _structural_complexity(["ev +PLU +LOC +GEN"])
        assert c1 < c2 < c3

    def test_root_only_zero_complexity(self):
        assert _structural_complexity(["ev"]) == pytest.approx(0.0)

    def test_single_non_deriv_tag(self):
        # chain_len=1, deriv_ratio=0 → 1*(1+0)=1.0
        assert _structural_complexity(["ev +PLU"]) == pytest.approx(1.0)

    def test_complexity_is_mean_over_parses(self):
        # parse1: "ev +PLU" → 1*(1+0)=1.0
        # parse2: "ev +PLU +LOC" → 2*(1+0)=2.0
        # mean = 1.5
        result = _structural_complexity(["ev +PLU", "ev +PLU +LOC"])
        assert result == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# 6. Derivational tags increase structural complexity
# ---------------------------------------------------------------------------

class TestDerivationalTags:
    def test_derivational_tag_increases_complexity_vs_nonderiv(self):
        """A parse with +CAUS should score higher than same-length non-deriv."""
        # Both have 1 tag; +CAUS is derivational, +PLU is not
        c_deriv = _structural_complexity(["ev +CAUS"])
        c_nonderiv = _structural_complexity(["ev +PLU"])
        assert c_deriv > c_nonderiv

    def test_all_derivational_tags_in_frozenset(self):
        expected = {"+CAUS", "+PASS", "+BECOME", "+AGT", "+INF", "+PASTPART", "+FUTPART"}
        assert DERIVATIONAL_TAGS == expected

    def test_mixed_deriv_ratio(self):
        # "ev +CAUS +PLU": chain=2, n_deriv=1, ratio=0.5 → 2*(1+0.5)=3.0
        result = _structural_complexity(["ev +CAUS +PLU"])
        assert result == pytest.approx(3.0)

    def test_full_deriv_chain(self):
        # "ev +CAUS +PASS": chain=2, n_deriv=2, ratio=1.0 → 2*(1+1.0)=4.0
        result = _structural_complexity(["ev +CAUS +PASS"])
        assert result == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# 7. compute_mis returns a float
# ---------------------------------------------------------------------------

class TestComputeMisReturnType:
    def test_returns_float(self):
        result = compute_mis("evlerden", ["ev +PLU +ABL"])
        assert isinstance(result, float)

    def test_result_in_unit_range(self):
        result = compute_mis("evlerden", ["ev +PLU +ABL"])
        assert 0.0 <= result <= 1.0

    def test_token_arg_ignored(self):
        """token argument does not affect the result."""
        r1 = compute_mis("evlerden", ["ev +PLU +ABL"])
        r2 = compute_mis("completely_different_token", ["ev +PLU +ABL"])
        assert r1 == pytest.approx(r2)

    def test_consistency_with_normalized_components(self):
        """compute_mis equals the weighted sum of normalized components."""
        parses = ["ev +PLU +ABL", "ev +PLU +LOC"]
        score = compute_mis("ev", parses)
        h = _morphological_entropy(parses)
        d = _canonicalization_density(parses)
        c = _structural_complexity(parses)
        expected = (
            MIS_ALPHA * min(h / 3.32, 1.0)
            + MIS_BETA * min(d / 8.0, 1.0)
            + MIS_GAMMA * min(c / 16.0, 1.0)
        )
        assert score == pytest.approx(expected)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# 8. Weights sum to 1.0
# ---------------------------------------------------------------------------

class TestWeights:
    def test_weights_sum_to_one(self):
        total = MIS_ALPHA + MIS_BETA + MIS_GAMMA
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_individual_weights_positive(self):
        assert MIS_ALPHA > 0
        assert MIS_BETA > 0
        assert MIS_GAMMA > 0

    def test_allomorph_counts_non_empty(self):
        assert len(ALLOMORPH_COUNTS) > 0
        assert all(v >= 1 for v in ALLOMORPH_COUNTS.values())
