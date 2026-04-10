"""Tests for structured output formatting."""

from __future__ import annotations

import json

import pytest

from kokturk.core.datatypes import Morpheme, MorphologicalAnalysis, TokenAnalyses
from kokturk.core.output_formats import OutputFormatter


def _make_analyses(surface: str = "evlerinden") -> TokenAnalyses:
    """Helper: build a TokenAnalyses with Turkish characters."""
    analysis = MorphologicalAnalysis(
        surface=surface,
        root="ev",
        tags=("+PLU", "+POSS.3SG", "+ABL"),
        morphemes=(
            Morpheme(surface="ev", canonical="ev", category="inflectional"),
            Morpheme(surface="ler", canonical="+PLU", category="inflectional"),
        ),
        source="zeyrek",
        score=0.50,
    )
    return TokenAnalyses(surface=surface, analyses=(analysis,))


def _make_empty() -> TokenAnalyses:
    return TokenAnalyses(surface="xyz", analyses=())


class TestOutputFormatterInit:
    def test_valid_modes(self) -> None:
        for mode in ("text", "json", "minimal"):
            fmt = OutputFormatter(mode=mode)
            assert fmt.mode == mode

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid output mode"):
            OutputFormatter(mode="invalid")


class TestTextMode:
    def test_contains_surface_and_tags(self) -> None:
        fmt = OutputFormatter(mode="text")
        out = fmt.format(_make_analyses())
        assert "evlerinden" in out
        assert "+PLU" in out
        assert "ev" in out

    def test_empty_analyses(self) -> None:
        fmt = OutputFormatter(mode="text")
        out = fmt.format(_make_empty())
        assert "no analysis" in out


class TestJsonMode:
    def test_valid_json(self) -> None:
        fmt = OutputFormatter(mode="json")
        out = fmt.format(_make_analyses())
        parsed = json.loads(out)
        assert parsed["surface"] == "evlerinden"
        assert len(parsed["analyses"]) == 1

    def test_root_and_tags(self) -> None:
        fmt = OutputFormatter(mode="json")
        parsed = json.loads(fmt.format(_make_analyses()))
        a = parsed["analyses"][0]
        assert a["root"] == "ev"
        assert "+PLU" in a["tags"]
        assert a["source"] == "zeyrek"
        assert a["score"] == pytest.approx(0.50)

    def test_turkish_chars_preserved(self) -> None:
        """ensure_ascii=False must preserve Turkish ğ, ş, ç, ö, ü, ı."""
        ta = _make_analyses("göğüs")
        fmt = OutputFormatter(mode="json")
        out = fmt.format(ta)
        assert "göğüs" in out
        # Must NOT be escaped to \\u sequences
        assert "\\u" not in out

    def test_empty_analyses(self) -> None:
        fmt = OutputFormatter(mode="json")
        parsed = json.loads(fmt.format(_make_empty()))
        assert parsed["analyses"] == []


class TestMinimalMode:
    def test_no_ansi_codes(self) -> None:
        fmt = OutputFormatter(mode="minimal")
        out = fmt.format(_make_analyses())
        assert "\033" not in out
        assert "\x1b" not in out

    def test_tab_separated(self) -> None:
        fmt = OutputFormatter(mode="minimal")
        out = fmt.format(_make_analyses())
        assert "\t" in out

    def test_empty_analyses(self) -> None:
        fmt = OutputFormatter(mode="minimal")
        out = fmt.format(_make_empty())
        assert "\t_" in out


class TestBatchFormatting:
    def test_json_batch_valid(self) -> None:
        fmt = OutputFormatter(mode="json")
        results = [_make_analyses(), _make_analyses("gördüm")]
        out = fmt.format_batch(results)
        parsed = json.loads(out)
        assert isinstance(parsed, list)
        assert len(parsed) == 2

    def test_text_batch_newline_separated(self) -> None:
        fmt = OutputFormatter(mode="text")
        out = fmt.format_batch([_make_analyses(), _make_analyses()])
        assert "\n" in out

    def test_empty_batch(self) -> None:
        fmt = OutputFormatter(mode="json")
        out = fmt.format_batch([])
        assert json.loads(out) == []

    def test_minimal_batch(self) -> None:
        fmt = OutputFormatter(mode="minimal")
        out = fmt.format_batch([_make_analyses(), _make_empty()])
        lines = out.split("\n")
        assert len(lines) == 2
