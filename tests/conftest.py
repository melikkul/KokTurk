"""Shared test fixtures for morpho-tr tests."""

from __future__ import annotations

import pytest

from kokturk.core.analyzer import MorphoAnalyzer
from kokturk.core.datatypes import Morpheme, MorphologicalAnalysis, TokenAnalyses


@pytest.fixture
def sample_morpheme() -> Morpheme:
    """A sample Morpheme for testing."""
    return Morpheme(surface="ler", canonical="+PLU", category="inflectional")


@pytest.fixture
def sample_analysis() -> MorphologicalAnalysis:
    """A sample MorphologicalAnalysis for 'evlerinden'."""
    return MorphologicalAnalysis(
        surface="evlerinden",
        root="ev",
        tags=("+PLU", "+POSS.3SG", "+ABL"),
        morphemes=(
            Morpheme(surface="ev", canonical="ev", category="inflectional"),
            Morpheme(surface="ler", canonical="+PLU", category="inflectional"),
            Morpheme(surface="i", canonical="+POSS.3SG", category="inflectional"),
            Morpheme(surface="nden", canonical="+ABL", category="inflectional"),
        ),
        source="zeyrek",
        score=0.5,
    )


@pytest.fixture
def sample_token_analyses(sample_analysis: MorphologicalAnalysis) -> TokenAnalyses:
    """A sample TokenAnalyses with two parses."""
    alt_analysis = MorphologicalAnalysis(
        surface="evlerinden",
        root="ev",
        tags=("+PLU", "+POSS.3PL", "+ABL"),
        morphemes=(),
        source="zeyrek",
        score=0.3,
    )
    return TokenAnalyses(
        surface="evlerinden",
        analyses=(sample_analysis, alt_analysis),
    )


@pytest.fixture
def analyzer() -> MorphoAnalyzer:
    """A MorphoAnalyzer with Zeyrek backend only."""
    return MorphoAnalyzer(backends=["zeyrek"])
