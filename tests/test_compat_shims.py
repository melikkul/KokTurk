"""Asserts the compat shims still resolve to aksu.kokturk / aksu.ariturk symbols
and emit DeprecationWarning.

NOTE: these tests install the shim packages from compat/ via sys.path manipulation
so they can run without a separate pip install of kokturk/ariturk.
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

# Add compat package src dirs to path so they resolve without being pip-installed
_COMPAT_KOKTURK = str(Path(__file__).parent.parent / "compat" / "kokturk" / "src")
_COMPAT_ARITURK = str(Path(__file__).parent.parent / "compat" / "ariturk" / "src")
if _COMPAT_KOKTURK not in sys.path:
    sys.path.insert(0, _COMPAT_KOKTURK)
if _COMPAT_ARITURK not in sys.path:
    sys.path.insert(0, _COMPAT_ARITURK)


def test_kokturk_shim_emits_warning_and_resolves():
    # Remove cached module if already imported so warning fires fresh
    sys.modules.pop("kokturk", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import kokturk
        from kokturk import MorphoAnalyzer, Atomizer
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations, "Expected DeprecationWarning from kokturk shim"
    assert "renamed to `aksu.kokturk`" in str(deprecations[0].message)
    import aksu.kokturk
    assert kokturk is aksu.kokturk
    assert MorphoAnalyzer is aksu.kokturk.MorphoAnalyzer
    assert Atomizer is aksu.kokturk.Atomizer


def test_ariturk_shim_emits_warning_and_resolves():
    sys.modules.pop("ariturk", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import ariturk
        from ariturk import TextCleaner, turkish_lower
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations, "Expected DeprecationWarning from ariturk shim"
    assert "renamed to `aksu.ariturk`" in str(deprecations[0].message)
    import aksu.ariturk
    assert ariturk is aksu.ariturk
    assert TextCleaner is aksu.ariturk.TextCleaner
    assert turkish_lower is aksu.ariturk.turkish_lower
