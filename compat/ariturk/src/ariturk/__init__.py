"""Deprecated alias for aksu.ariturk. Will be removed in aksu 2.0.

Existing imports keep working:
    from ariturk import TextCleaner, turkish_lower
will now resolve to the corresponding symbols under aksu.ariturk.
"""
from __future__ import annotations
import sys as _sys
import warnings as _warnings

_warnings.warn(
    "The top-level `ariturk` package has been renamed to `aksu.ariturk`. "
    "Please update your imports to `from aksu import ...` or `from aksu.ariturk import ...`. "
    "This compatibility shim will be removed in aksu 2.0.",
    DeprecationWarning,
    stacklevel=2,
)

import aksu.ariturk as _impl  # noqa: E402
_sys.modules[__name__] = _impl
