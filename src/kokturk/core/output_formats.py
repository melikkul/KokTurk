"""Structured output formatting for CLI and API responses.

Supports three modes:

* **text** — human-readable output (default).
* **json** — structured JSON for machine parsing and screen readers.
* **minimal** — plain text without ANSI codes or animations.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from kokturk.core.datatypes import TokenAnalyses

_VALID_MODES = frozenset({"text", "json", "minimal"})


class OutputFormatter:
    """Format analysis results for different output channels.

    Args:
        mode: One of ``"text"`` (default), ``"json"``, or ``"minimal"``.

    Raises:
        ValueError: If *mode* is not one of the valid modes.
    """

    def __init__(self, mode: str = "text") -> None:
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Invalid output mode {mode!r}. Must be one of {sorted(_VALID_MODES)}."
            )
        self.mode = mode

    # ------------------------------------------------------------------
    # Single result
    # ------------------------------------------------------------------

    def format(self, result: TokenAnalyses) -> str:
        """Format a single :class:`TokenAnalyses` object."""
        if self.mode == "json":
            return self._format_json(result)
        if self.mode == "minimal":
            return self._format_minimal(result)
        return self._format_text(result)

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    def format_batch(self, results: Iterable[TokenAnalyses]) -> str:
        """Format multiple analysis results.

        In JSON mode the output is a JSON array.  In other modes each
        result is separated by a newline.
        """
        items = list(results)
        if self.mode == "json":
            records = [self._result_to_dict(r) for r in items]
            return json.dumps(records, ensure_ascii=False, indent=2)
        return "\n".join(self.format(r) for r in items)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _result_to_dict(result: TokenAnalyses) -> dict:
        """Convert a single result to a plain dict for JSON serialization."""
        return {
            "surface": result.surface,
            "analyses": [
                {
                    "root": a.root,
                    "tags": list(a.tags),
                    "source": a.source,
                    "score": a.score,
                }
                for a in result.analyses
            ],
        }

    def _format_json(self, result: TokenAnalyses) -> str:
        return json.dumps(self._result_to_dict(result), ensure_ascii=False)

    @staticmethod
    def _format_text(result: TokenAnalyses) -> str:
        if not result.analyses:
            return f"{result.surface}\t(no analysis)"
        lines = []
        for a in result.analyses:
            lines.append(f"{result.surface} \u2192 {a.to_str()} [{a.source}, {a.score:.2f}]")
        return "\n".join(lines)

    @staticmethod
    def _format_minimal(result: TokenAnalyses) -> str:
        best = result.best
        if best is None:
            return f"{result.surface}\t_"
        return f"{result.surface}\t{best.to_str()}"
