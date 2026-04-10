"""Synthetic inflection importer — wraps paradigm augmentation."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def import_synthetic_inflections(
    tag_frequency_json: Path | str,
    output_path: Path | str,
    roots: list[tuple[str, str]] | None = None,
    alternation_map: dict[str, str] | None = None,
) -> int:
    """Generate synthetic inflections targeting rare tags.

    Delegates to :func:`data.paradigm_augmentation.augment_corpus`. Returns
    the number of synthetic entries written.
    """
    from data.paradigm_augmentation import augment_corpus

    if roots is None:
        roots = [("ev", "Noun"), ("göz", "Noun"), ("kitap", "Noun"),
                 ("gel", "Verb"), ("git", "Verb")]
    report = augment_corpus(tag_frequency_json, output_path, roots,
                            alternation_map=alternation_map)
    logger.info(
        "Synthetic inflections: %d generated, %d kept, %d discarded",
        report.generated, report.kept, report.discarded_invalid,
    )
    return report.kept
