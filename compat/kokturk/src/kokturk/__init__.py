"""Deprecated alias for aksu.kokturk. Will be removed in aksu 2.0.

Existing imports keep working:
    from kokturk import MorphoAnalyzer, Atomizer
will now resolve to the corresponding symbols under aksu.kokturk.
"""
from __future__ import annotations
import sys as _sys
import warnings as _warnings

_warnings.warn(
    "The top-level `kokturk` package has been renamed to `aksu.kokturk`. "
    "Please update your imports to `from aksu import ...` or `from aksu.kokturk import ...`. "
    "This compatibility shim will be removed in aksu 2.0.",
    DeprecationWarning,
    stacklevel=2,
)

import aksu.kokturk as _impl  # noqa: E402
_sys.modules[__name__] = _impl
