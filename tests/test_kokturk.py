"""Tests for the kök-türk (kokturk) library wrapper."""
from __future__ import annotations

import pytest


def test_atomizer_import():
    from kokturk import Atomizer  # noqa: F401


def test_analysis_result_import():
    from kokturk import MorphologicalAnalysis  # noqa: F401


def test_version():
    import kokturk
    assert kokturk.__version__ == "0.1.0"


def test_atomizer_analyze():
    from kokturk import Atomizer

    a = Atomizer(backend="zeyrek")
    result = a.analyze("ev")
    assert result is not None
    assert result.root == "ev"


def test_atomizer_analyze_all():
    from kokturk import Atomizer

    a = Atomizer(backend="zeyrek")
    results = a.analyze_all("ev")
    assert len(results) >= 1
    assert results[0].root == "ev"


def test_to_canonical():
    from kokturk import Atomizer

    a = Atomizer(backend="zeyrek")
    canonical = a.to_canonical("ev")
    assert "ev" in canonical


def test_analyze_batch():
    from kokturk import Atomizer

    a = Atomizer(backend="zeyrek")
    results = a.analyze_batch(["ev", "okul"])
    assert len(results) == 2
    assert all(r is not None for r in results)


def test_unknown_word_returns_string():
    from kokturk import Atomizer

    a = Atomizer(backend="zeyrek")
    result = a.to_canonical("xyzqwerty12345")
    # Unknown words get a fallback canonical form
    assert isinstance(result, str)
    assert len(result) > 0
