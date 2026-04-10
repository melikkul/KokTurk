"""Bulk Zeyrek morphological analysis for resource generation."""
from __future__ import annotations
import logging
from pathlib import Path
import sys

logger = logging.getLogger(__name__)

# Add src to path if needed
_SRC = Path(__file__).parent.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def analyze_bulk(
    surfaces: list[str],
    batch_size: int = 1000,
) -> dict[str, str | None]:
    """Run Zeyrek on a list of surface forms and return canonical tags.

    Uses the project's MorphoAnalyzer with the zeyrek backend. For surfaces
    where Zeyrek has exactly one parse, returns the canonical tag string.
    For ambiguous surfaces (multiple parses), returns the highest-scoring parse.

    Args:
        surfaces: List of surface forms to analyze.
        batch_size: Process in batches of this size (for progress logging).

    Returns:
        Dict mapping surface → canonical_tags_string (or None if Zeyrek
        produces no parses for that surface).
    """
    try:
        from kokturk.core.analyzer import MorphoAnalyzer
    except ImportError:
        logger.error("kokturk not importable; check PYTHONPATH includes src/")
        return {}

    results: dict[str, str | None] = {}

    with MorphoAnalyzer(backends=["zeyrek"]) as analyzer:
        for i in range(0, len(surfaces), batch_size):
            batch = surfaces[i : i + batch_size]
            if i % 10_000 == 0:
                logger.info("Zeyrek bulk: %d / %d surfaces", i, len(surfaces))

            for surface in batch:
                try:
                    token_analyses = analyzer.analyze(surface)
                except Exception as exc:
                    logger.warning("Zeyrek failed on '%s': %s", surface, exc)
                    results[surface] = None
                    continue

                if not token_analyses.analyses:
                    results[surface] = None
                    continue

                # Use first (highest-scoring) analysis
                best = token_analyses.analyses[0]
                tag_parts = [best.root] + list(best.tags)
                results[surface] = " ".join(tag_parts)

    return results
