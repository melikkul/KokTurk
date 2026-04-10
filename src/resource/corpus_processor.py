"""Extract MorphEntry objects from CoNLL-U treebank files."""
from __future__ import annotations
import logging
from pathlib import Path

from resource.normalizer import normalize_surface, normalize_canonical
from resource.schema import MorphEntry
from resource.tag_mappings import ud_feats_to_canonical

logger = logging.getLogger(__name__)


def parse_conllu_file(
    path: Path,
    source: str = "boun",
    tier: str = "gold",
) -> list[MorphEntry]:
    """Parse a CoNLL-U file and return MorphEntry objects.

    Handles:
    - Multi-word tokens (IDs like "2-3"): skipped.
    - Empty nodes (IDs like "1.1"): skipped.
    - Comment lines (starting with #): skipped.
    - Missing FEATS column ("_"): treated as no features.

    Args:
        path: Path to the .conllu file.
        source: Source identifier for all entries (e.g., "boun", "imst").
        tier: Quality tier for all entries from this file.

    Returns:
        List of MorphEntry objects, one per token (excluding multi-word tokens
        and empty nodes).
    """
    entries: list[MorphEntry] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            # Skip comment and blank lines
            if line.startswith("#") or not line.strip():
                continue

            cols = line.split("\t")
            if len(cols) < 6:
                continue

            token_id = cols[0]
            # Skip multi-word tokens (e.g., "2-3") and empty nodes (e.g., "1.1")
            if "-" in token_id or "." in token_id:
                continue

            form = cols[1]
            lemma = cols[2] if cols[2] != "_" else form
            pos = cols[3]  # UPOS
            feats = cols[5] if len(cols) > 5 else "_"

            # Skip punctuation and unknown tokens
            if pos in ("PUNCT", "X", "_", "SYM"):
                continue

            norm_surface = normalize_surface(form)
            if not norm_surface:
                continue

            canonical = normalize_canonical(ud_feats_to_canonical(lemma.lower(), pos, feats))

            entries.append(MorphEntry(
                surface=norm_surface,
                lemma=lemma.lower(),
                canonical_tags=canonical,
                pos=pos,
                source=source,
                confidence=1.0,
                frequency=1,
                tier=tier,
            ))

    logger.debug("Parsed %d entries from %s", len(entries), path)
    return entries
