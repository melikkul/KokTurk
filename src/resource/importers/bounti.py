"""BounTi Turkish tweets importer (domain=social_media).

Accepts a local path to the BounTi dataset. Auto-download is best-effort via
``git clone``; on failure (typical on restricted compute nodes behind
firewall) the user is told exactly where to place the data.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

BOUNTI_DEFAULT_DIR = "data/external/bounti"
BOUNTI_CLONE_URL = "https://github.com/boun-tabi/BounTi.git"


def _iter_tweets(root: Path) -> Iterable[str]:
    for p in root.rglob("*.csv"):
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Crude: CSV may have quoted comma-separated columns; take
                # the longest column as the tweet text.
                fields = [c.strip(' "\'') for c in line.split(",")]
                if not fields:
                    continue
                text = max(fields, key=len)
                if len(text) >= 3:
                    yield text
    for p in root.rglob("*.txt"):
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if len(line) >= 3:
                yield line


def import_bounti(
    local_path: Path | str = BOUNTI_DEFAULT_DIR,
    db: object | None = None,
    try_clone: bool = True,
) -> int:
    """Import BounTi tweets.

    Args:
        local_path: Location of an already-downloaded BounTi repo/dataset.
        db: Optional ``MorphDatabase`` for inserting entries. When ``None``
            the importer just counts parseable rows (useful for tests).
        try_clone: When ``True`` and ``local_path`` is missing, attempt
            ``git clone``. On failure prints clear instructions.

    Returns:
        Number of entries handled.
    """
    path = Path(local_path)
    if not path.exists() and try_clone:
        logger.info("BounTi not found at %s; attempting git clone ...", path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth", "1", BOUNTI_CLONE_URL, str(path)],
                check=True, capture_output=True,
            )
        except Exception as e:  # noqa: BLE001
            logger.error(
                "BounTi auto-download failed (%s). Please manually "
                "download and place the dataset at %s",
                e, path,
            )
            return 0
    if not path.exists():
        logger.error(
            "BounTi dataset not found at %s. Place data manually.", path,
        )
        return 0

    n = 0
    for text in _iter_tweets(path):
        n += 1
        if db is not None:
            try:
                _insert_bronze(db, text, domain="social_media")
            except Exception:  # noqa: BLE001
                pass
    logger.info("BounTi imported %d tweets", n)
    return n


def _insert_bronze(db: object, surface: str, domain: str) -> None:
    """Best-effort insert honoring whatever schema ``db`` exposes."""
    from resource.schema import MorphEntry  # type: ignore
    entry = MorphEntry(
        surface=surface,
        lemma=surface,
        canonical_tags=f"{surface}",
        pos="NOUN",
        source="bounti",
        confidence=0.5,
        frequency=1,
        tier="bronze",
    )
    try:
        db.bulk_insert([entry])  # type: ignore[attr-defined]
    except Exception:
        pass
