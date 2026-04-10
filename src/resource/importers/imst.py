"""IMST UD Treebank importer."""
from __future__ import annotations
import logging
from pathlib import Path

from resource.corpus_processor import parse_conllu_file
from resource.schema import MorphDatabase

logger = logging.getLogger(__name__)

IMST_CLONE_URL = "https://github.com/UniversalDependencies/UD_Turkish-IMST.git"
IMST_DEFAULT_DIR = Path("data/external/imst_treebank")


def import_imst(treebank_dir: Path, db: MorphDatabase) -> int:
    """Import the IMST UD Treebank into the resource database.

    Reads all .conllu files found under treebank_dir.
    All entries are assigned source="imst" and tier="gold".

    Args:
        treebank_dir: Path to the IMST treebank directory.
        db: Target MorphDatabase instance.

    Returns:
        Total number of entries parsed.

    Raises:
        FileNotFoundError: If treebank_dir does not exist.
    """
    if not treebank_dir.exists():
        raise FileNotFoundError(
            f"IMST treebank not found at {treebank_dir}. "
            f"Clone with: git clone --depth 1 {IMST_CLONE_URL} {treebank_dir}"
        )

    conllu_files = sorted(treebank_dir.glob("*.conllu"))
    if not conllu_files:
        logger.warning("No .conllu files found in %s", treebank_dir)
        return 0

    total = 0
    for conllu_path in conllu_files:
        entries = parse_conllu_file(conllu_path, source="imst", tier="gold")
        db.bulk_insert(entries)
        logger.info("IMST %s: %d entries", conllu_path.name, len(entries))
        total += len(entries)

    return total
