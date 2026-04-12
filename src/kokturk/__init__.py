"""kök-türk: Turkish Morphological Atomizer.

Decomposes Turkish surface forms into linguistic atoms:
root + ordered canonical morphological tags.

Example::

    >>> from kokturk import Atomizer
    >>> Atomizer().to_canonical("evlerinden")
    'ev +PLU +POSS.3SG +ABL'
"""
from __future__ import annotations

__version__ = "0.1.0"

from kokturk.core.analyzer import MorphoAnalyzer
from kokturk.core.datatypes import Morpheme, MorphologicalAnalysis, TokenAnalyses
from kokturk.core.grammar import GrammarChecker

__all__ = [
    "MorphoAnalyzer",
    "MorphologicalAnalysis",
    "Morpheme",
    "TokenAnalyses",
    "Atomizer",
    "GrammarChecker",
]


class Atomizer:
    """High-level API for Turkish morphological atomization.

    Args:
        backend: Backend name (``"zeyrek"``, ``"neural"``).
        model_path: Path to neural model checkpoint (for ``"neural"``).
        vocab_dir: Directory containing vocab JSON files.
    """

    def __init__(
        self,
        backend: str = "zeyrek",
        model_path: str | None = None,
        vocab_dir: str | None = None,
    ) -> None:
        kwargs: dict[str, object] = {}
        if model_path is not None:
            kwargs["model_path"] = model_path
        if vocab_dir is not None:
            kwargs["vocab_dir"] = vocab_dir
        self._analyzer = MorphoAnalyzer(backends=[backend], **kwargs)
        self.backend = backend

    def analyze(self, word: str) -> MorphologicalAnalysis | None:
        """Analyze a single Turkish word, returning the best parse."""
        result = self._analyzer.analyze(word)
        return result.best if result.analyses else None

    def analyze_all(self, word: str) -> list[MorphologicalAnalysis]:
        """Return all candidate analyses for a word."""
        result = self._analyzer.analyze(word)
        return list(result.analyses)

    def analyze_batch(self, words: list[str]) -> list[MorphologicalAnalysis | None]:
        """Analyze a list of words."""
        return [self.analyze(w) for w in words]

    def to_canonical(self, word: str) -> str:
        """Return canonical string directly.

        >>> Atomizer().to_canonical("evlerinden")
        'ev +PLU +POSS.3SG +ABL'
        """
        result = self.analyze(word)
        return result.to_str() if result else word
