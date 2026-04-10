"""BOUN UD Treebank importer."""
from __future__ import annotations
import logging
from pathlib import Path

from resource.corpus_processor import parse_conllu_file
from resource.schema import MorphDatabase, MorphEntry

logger = logging.getLogger(__name__)


def import_boun(treebank_dir: Path, db: MorphDatabase) -> int:
    """Import the BOUN UD Treebank into the resource database.

    Reads all .conllu files found under treebank_dir.
    All entries are assigned source="boun" and tier="gold".

    Args:
        treebank_dir: Path to the BOUN treebank directory.
        db: Target MorphDatabase instance.

    Returns:
        Total number of entries inserted.

    Raises:
        FileNotFoundError: If treebank_dir does not exist.
    """
    if not treebank_dir.exists():
        raise FileNotFoundError(f"BOUN treebank not found at {treebank_dir}")

    conllu_files = sorted(treebank_dir.glob("*.conllu"))
    if not conllu_files:
        logger.warning("No .conllu files found in %s", treebank_dir)
        return 0

    total = 0
    for conllu_path in conllu_files:
        entries = parse_conllu_file(conllu_path, source="boun", tier="gold")
        inserted = db.bulk_insert(entries)
        logger.info("BOUN %s: %d entries parsed, %d inserted", conllu_path.name, len(entries), inserted)
        total += len(entries)  # count parsed, not just inserted (for reporting)

    return total
