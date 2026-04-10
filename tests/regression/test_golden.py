"""Golden regression tests using structured JSON test set.

Compare analyzer output against saved known-good analyses.
Each entry verifies that at least one analysis from the analyzer
matches the expected root AND contains all expected tags.

Different from test_known_analyses.py: uses structured JSON format
with category metadata, focused on canonical tag output verification.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kokturk.core.analyzer import MorphoAnalyzer

GOLDEN_PATH = Path(__file__).parent / "golden_test_set.json"


def load_golden() -> list[dict]:
    """Load golden test entries from JSON."""
    with open(GOLDEN_PATH) as f:
        return json.load(f)["entries"]


@pytest.fixture(scope="module")
def analyzer() -> MorphoAnalyzer:
    """Module-scoped analyzer (avoid repeated init)."""
    return MorphoAnalyzer(backends=["zeyrek"])


@pytest.mark.parametrize("entry", load_golden(), ids=lambda e: e["id"])
def test_golden_root_present(entry: dict, analyzer: MorphoAnalyzer) -> None:
    """At least one analysis must match the expected root."""
    result = analyzer.analyze(entry["surface"])
    roots = {a.root for a in result.analyses}
    assert entry["expected_root"] in roots, (
        f"Root regression on '{entry['surface']}' [{entry['category']}]: "
        f"expected root '{entry['expected_root']}', got roots: {roots}"
    )


@pytest.mark.parametrize("entry", load_golden(), ids=lambda e: e["id"])
def test_golden_tags_present(entry: dict, analyzer: MorphoAnalyzer) -> None:
    """At least one analysis with the expected root must contain all expected tags."""
    result = analyzer.analyze(entry["surface"])
    matching = [
        a for a in result.analyses
        if a.root == entry["expected_root"]
    ]
    if not matching:
        pytest.skip(f"Root '{entry['expected_root']}' not found — covered by test_golden_root_present")

    expected_tags = entry["expected_tags"]
    for analysis in matching:
        if list(analysis.tags) == expected_tags:
            return  # exact match found

    # Show what we got for debugging
    actual_tag_sets = [list(a.tags) for a in matching]
    assert False, (
        f"Tag regression on '{entry['surface']}' [{entry['category']}]: "
        f"expected {expected_tags}, got {actual_tag_sets}"
    )
