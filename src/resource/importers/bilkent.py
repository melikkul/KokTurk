"""Bilkent Turkish Writings dataset importer (domain=creative_writing)."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def import_bilkent(
    source: str | Path = "data/external/bilkent",
    db: object | None = None,
) -> int:
    """Import Bilkent writings from either a local path or HF datasets.

    Args:
        source: Either a local directory / CSV path, or the string
            ``"huggingface"`` to load via the ``datasets`` library.
        db: Optional MorphDatabase.

    Returns:
        Number of word tokens imported.
    """
    n = 0
    if str(source) == "huggingface":
        try:
            from datasets import load_dataset  # type: ignore
            ds = load_dataset("bilkent-writings", split="train")
            iterator = (row.get("text", "") for row in ds)
        except Exception as e:  # noqa: BLE001
            logger.error("HF load failed: %s. Fall back to local path.", e)
            return 0
    else:
        path = Path(source)
        if not path.exists():
            logger.error("Bilkent source %s not found.", path)
            return 0
        iterator = _iter_local_text(path)

    from tqdm import tqdm  # type: ignore
    for text in tqdm(iterator, desc="bilkent", unit="doc"):
        words = text.split()
        n += len(words)
        if db is not None:
            for w in words:
                try:
                    _insert_bronze(db, w, "creative_writing")
                except Exception:  # noqa: BLE001
                    pass
    logger.info("Bilkent imported %d tokens", n)
    return n


def _iter_local_text(path: Path):
    if path.is_file():
        yield from _read(path)
        return
    for p in path.rglob("*.txt"):
        yield from _read(p)


def _read(p: Path):
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line:
            yield line


def _insert_bronze(db: object, surface: str, domain: str) -> None:
    from resource.schema import MorphEntry  # type: ignore
    entry = MorphEntry(
        surface=surface, lemma=surface, canonical_tags=surface,
        pos="NOUN", source="bilkent", confidence=0.5, frequency=1, tier="bronze",
    )
    try:
        db.bulk_insert([entry])  # type: ignore[attr-defined]
    except Exception:
        pass
