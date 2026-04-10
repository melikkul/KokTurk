"""Morpheme boundary extraction wrapper for arı-türk."""
from __future__ import annotations

from resource.boundary_extractor import extract_boundaries as _extract


class BoundaryExtractor:
    """Extract morpheme boundaries from Turkish words.

    Example::

        >>> ext = BoundaryExtractor()
        >>> ext.extract("evlerinden", "ev +PLU +POSS.3SG +ABL")
        'ev|ler|i|nden'
    """

    def extract(self, surface: str, canonical: str) -> str:
        """Extract boundaries for a single word."""
        return _extract(surface, canonical)

    def extract_batch(self, pairs: list[tuple[str, str]]) -> list[str]:
        """Extract boundaries for a list of (surface, canonical) pairs."""
        return [self.extract(s, c) for s, c in pairs]
