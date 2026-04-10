"""Trendyol e-commerce reviews importer (domain=ecommerce).

Local CSV path only — no auto-download (Kaggle requires authentication).
Preprocesses reviews: strips HTML entities, emoji, URLs, product codes, and
inline star ratings. Skips reviews shorter than 3 Turkish words.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

_HTML_ENTITY_RE = re.compile(r"&[a-zA-Z#0-9]+;")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F600-\U0001F64F]",
)
_STAR_RATING_RE = re.compile(
    r"(\d+\s*(?:yıldız|star|\*))|[⭐★☆]+", re.IGNORECASE,
)
_PRODUCT_CODE_RE = re.compile(
    r"\b(?:XS|S|M|L|XL|XXL|\d+\s*numara)\b", re.IGNORECASE,
)
_WORD_RE = re.compile(r"[a-zA-ZçğıöşüÇĞİÖŞÜ]+")


def clean_review(text: str) -> str:
    """Strip HTML entities, emoji, URLs, ratings, product codes, collapse ws."""
    if not text:
        return ""
    text = _HTML_ENTITY_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _STAR_RATING_RE.sub(" ", text)
    text = _PRODUCT_CODE_RE.sub(" ", text)
    text = _EMOJI_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _iter_rows(csv_path: Path) -> Iterable[str]:
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # Try to locate a "review"/"comment"/"text" column; default to longest.
        text_col = None
        if header:
            for i, col in enumerate(header):
                if col.lower().strip() in {"review", "comment", "text", "content"}:
                    text_col = i
                    break
        for row in reader:
            if not row:
                continue
            text = row[text_col] if text_col is not None and text_col < len(row) \
                else max(row, key=len)
            yield text


def import_trendyol(
    csv_path: Path | str,
    db: object | None = None,
    min_words: int = 3,
) -> int:
    """Import Trendyol reviews from a local CSV file.

    Returns the number of usable review tokens emitted after cleaning.
    Reviews with fewer than ``min_words`` Turkish words are discarded.
    """
    p = Path(csv_path)
    if not p.exists():
        logger.error(
            "Trendyol CSV not found at %s. Kaggle requires authentication; "
            "download manually and pass the local CSV path.", p,
        )
        return 0

    from tqdm import tqdm  # type: ignore

    n = 0
    for raw in tqdm(_iter_rows(p), desc="trendyol", unit="rev"):
        cleaned = clean_review(raw)
        if not cleaned:
            continue
        words = _WORD_RE.findall(cleaned)
        if len(words) < min_words:
            continue
        n += len(words)
        if db is not None:
            for w in words:
                try:
                    _insert_bronze(db, w, "ecommerce")
                except Exception:  # noqa: BLE001
                    pass
    logger.info("Trendyol imported %d tokens", n)
    return n


def _insert_bronze(db: object, surface: str, domain: str) -> None:
    from resource.schema import MorphEntry  # type: ignore
    entry = MorphEntry(
        surface=surface, lemma=surface, canonical_tags=surface,
        pos="NOUN", source="trendyol", confidence=0.5, frequency=1, tier="bronze",
    )
    try:
        db.bulk_insert([entry])  # type: ignore[attr-defined]
    except Exception:
        pass
