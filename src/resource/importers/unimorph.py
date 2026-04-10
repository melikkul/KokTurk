"""UniMorph Turkish (tur) flat-file importer."""
from __future__ import annotations
import logging
from pathlib import Path

from resource.normalizer import normalize_canonical, normalize_surface
from resource.schema import MorphDatabase, MorphEntry
from resource.tag_mappings import unimorph_tags_to_canonical

logger = logging.getLogger(__name__)

UNIMORPH_CLONE_URL = "https://github.com/unimorph/tur.git"
UNIMORPH_DEFAULT_DIR = Path("data/external/unimorph_tur")

# Default file within the tur repository
_TUR_FILE = "tur"


def import_unimorph(tur_dir: Path, db: MorphDatabase) -> int:
    """Import UniMorph Turkish flat file into the resource database.

    UniMorph Turkish file format (tab-separated, 3 columns):
    ``lemma \\t form \\t semicolon_separated_tags``

    Example line: ``ev\\tev\\tN;NOM;SG``

    All entries are assigned source="unimorph" and an initial tier="bronze"
    (tier is re-evaluated in the generation runner after cross-source comparison).

    Args:
        tur_dir: Path to the cloned unimorph/tur repository.
        db: Target MorphDatabase instance.

    Returns:
        Number of entries parsed.

    Raises:
        FileNotFoundError: If tur_dir or the tur file does not exist.
    """
    tur_file = tur_dir / _TUR_FILE

    if not tur_dir.exists():
        raise FileNotFoundError(
            f"UniMorph tur not found at {tur_dir}. "
            f"Clone with: git clone --depth 1 {UNIMORPH_CLONE_URL} {tur_dir}"
        )

    if not tur_file.exists():
        # Try to find any plain text file
        candidates = list(tur_dir.glob("tur*"))
        text_files = [f for f in candidates if f.suffix in ("", ".tsv", ".txt")]
        if not text_files:
            raise FileNotFoundError(f"No tur data file found in {tur_dir}")
        tur_file = text_files[0]
        logger.info("Using UniMorph file: %s", tur_file)

    entries: list[MorphEntry] = []
    skipped = 0

    with open(tur_file, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue

            cols = line.split("\t")
            if len(cols) < 3:
                skipped += 1
                continue

            lemma, form, tag_string = cols[0].strip(), cols[1].strip(), cols[2].strip()

            if not lemma or not form or not tag_string:
                skipped += 1
                continue

            norm_surface = normalize_surface(form)
            norm_lemma = normalize_surface(lemma)

            # Extract POS from first tag (e.g., "N;NOM;SG" → "N")
            first_tag = tag_string.split(";")[0]
            pos_map = {"N": "NOUN", "V": "VERB", "ADJ": "ADJ", "ADV": "ADV",
                       "PRO": "PRON", "POST": "ADP", "CONJ": "CCONJ", "PROPN": "PROPN", "DET": "DET"}
            pos = pos_map.get(first_tag, "X")

            canonical = normalize_canonical(
                unimorph_tags_to_canonical(norm_lemma, norm_surface, tag_string)
            )

            entries.append(MorphEntry(
                surface=norm_surface,
                lemma=norm_lemma,
                canonical_tags=canonical,
                pos=pos,
                source="unimorph",
                confidence=0.9,
                frequency=1,
                tier="bronze",  # will be updated after cross-source comparison
            ))

            # Batch insert every 10K entries
            if len(entries) >= 10_000:
                db.bulk_insert(entries)
                entries = []

    if entries:
        db.bulk_insert(entries)

    logger.info("UniMorph import complete. Skipped %d malformed lines.", skipped)
    return len(entries)
