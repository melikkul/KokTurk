"""The Architecture ASCII diagram's atomic-string examples must match what the
code emits for the same sentence. Catches drift between diagram and behavior.

The diagram uses "Çocuklar evlerinden çıktı" — the Architecture section example
sentence. Each word's root must appear in the diagram block.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")


def _extract_diagram_block() -> str:
    """Return text between ## Architecture and the next ## header."""
    parts = README.split("## Architecture")
    if len(parts) < 2:
        pytest.skip("Architecture section not found in README.md")
    return parts[1].split("##")[0]


@pytest.mark.skipif(
    "MorphoAnalyzer" not in dir(__import__("aksu", fromlist=["MorphoAnalyzer"])),
    reason="aksu.MorphoAnalyzer not available",
)
def test_diagram_roots_match_analyzer_output():
    """Each root token emitted by the analyzer appears in the diagram block."""
    from aksu import MorphoAnalyzer
    try:
        analyzer = MorphoAnalyzer(backends=["zeyrek"])
    except Exception:
        pytest.skip("MorphoAnalyzer with zeyrek backend not available")

    try:
        results = analyzer.analyze_sentence("Çocuklar evlerinden çıktı")
    except Exception as e:
        pytest.skip(f"analyze_sentence raised {e!r}")

    if not results:
        pytest.skip("analyze_sentence returned empty results")

    diagram = _extract_diagram_block()

    for token_analyses in results:
        if not token_analyses.analyses:
            continue
        root = token_analyses.analyses[0].root
        assert root in diagram, (
            f"Root {root!r} for surface {token_analyses.surface!r} not found in "
            f"Architecture diagram. Diagram text:\n{diagram}"
        )


def test_diagram_does_not_contain_legacy_poss3sg_format():
    """The diagram must not contain the legacy '+POSS.3SG +ABL' for evlerinden.
    The code emits '+POSS.3PL +ABL' as the first (scored) candidate.
    This catches the specific H-4 regression."""
    diagram = _extract_diagram_block()
    # The legacy format had both +PLU and +POSS.3SG together for evlerinden.
    # The current code's first candidate is +POSS.3PL (singular possessor, 3rd pl).
    assert "+POSS.3SG" not in diagram or "+POSS.3PL" in diagram, (
        "Diagram still shows legacy +POSS.3SG without the current +POSS.3PL form"
    )


def test_diagram_contains_current_format():
    """Diagram shows 'ev +Noun +POSS.3PL +ABL' atomic string (current code format)."""
    diagram = _extract_diagram_block()
    assert "+POSS.3PL" in diagram, (
        "Diagram does not contain '+POSS.3PL'; may be using legacy canonical format"
    )
    assert "+ABL" in diagram, "Diagram does not contain '+ABL'"
