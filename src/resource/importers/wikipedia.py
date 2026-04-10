"""Turkish Wikipedia importer for Generation 1 resource expansion.

Streams the HuggingFace ``wikipedia/20231101.tr`` dataset, extracts unique
Turkish surface forms not already in the database, runs Zeyrek on them, and
inserts new bronze-tier MorphEntry rows.

Requires: ``datasets>=2.19.0`` (already in requirements/train.txt).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Matches runs of 2+ Turkish characters (including dotless-i, cedilla, etc.)
TURKISH_WORD_RE = re.compile(
    r"[a-zA-Z\u00e7\u011f\u0131\u00f6\u015f\u00fc\u00c7\u011e\u0130\u00d6\u015e\u00dc]{2,}"
)

_CHUNK_SIZE = 5_000  # surfaces per Zeyrek batch call


def tokenize_turkish_text(text: str) -> list[str]:
    """Extract lowercase Turkish word tokens from raw Wikipedia text.

    Filters short tokens (< 2 chars) and strips leading/trailing punctuation.

    Args:
        text: Raw Wikipedia article text.

    Returns:
        List of lowercase Turkish word strings.
    """
    return [m.lower() for m in TURKISH_WORD_RE.findall(text)]


def import_wikipedia(
    db: object,  # MorphDatabase — typed loosely to avoid circular import
    max_articles: int | None = None,
    dataset_name: str = "wikipedia",
    language: str = "20231101.tr",
) -> int:
    """Stream Turkish Wikipedia, extract new surfaces, and analyse with Zeyrek.

    Only processes surfaces NOT already in the database to avoid redundant work.
    After inserting new zeyrek entries, re-evaluates tiers for any surface that
    now has ≥2 sources.

    Args:
        db: Open MorphDatabase instance.
        max_articles: If set, stop after processing this many Wikipedia articles
            (useful for debugging / smoke-testing).
        dataset_name: HuggingFace dataset name (default ``"wikipedia"``).
        language: Dataset config identifier (default ``"20231101.tr"``).

    Returns:
        Number of new MorphEntry rows inserted.
    """
    from datasets import load_dataset  # type: ignore[import]

    from resource.importers.zeyrek_bulk import analyze_bulk
    from resource.quality_check import tier_from_entries
    from resource.schema import MorphEntry

    # --------------------------------------------------------------------- #
    # 1. Load existing surfaces to skip
    # --------------------------------------------------------------------- #
    print("  Loading existing surfaces from DB ...")
    existing: set[str] = set(db.get_all_surfaces())
    print(f"  {len(existing):,} surfaces already in DB — will skip them")

    # --------------------------------------------------------------------- #
    # 2. Stream Wikipedia and collect new unique surfaces
    # --------------------------------------------------------------------- #
    print(f"  Streaming {dataset_name}/{language} ...")
    try:
        dataset = load_dataset(
            dataset_name, language, split="train", streaming=True, trust_remote_code=True
        )
    except RuntimeError:
        # datasets>=3.0 dropped script-based datasets; use wikimedia/wikipedia
        print("  Falling back to wikimedia/wikipedia ...")
        dataset = load_dataset(
            "wikimedia/wikipedia", f"20231101.tr", split="train", streaming=True,
        )

    new_surfaces: set[str] = set()
    n_articles = 0

    for article in dataset:
        text = article.get("text", "")
        tokens = tokenize_turkish_text(text)
        for tok in tokens:
            if tok not in existing and tok not in new_surfaces:
                new_surfaces.add(tok)

        n_articles += 1
        if n_articles % 10_000 == 0:
            logger.info("Processed %d articles, %d new surfaces so far", n_articles, len(new_surfaces))
            print(f"    articles={n_articles:,}  new surfaces={len(new_surfaces):,}")

        if max_articles is not None and n_articles >= max_articles:
            break

    print(f"  Finished streaming: {n_articles:,} articles, {len(new_surfaces):,} new surfaces")

    # --------------------------------------------------------------------- #
    # 3. Batch-analyse with Zeyrek
    # --------------------------------------------------------------------- #
    surface_list = sorted(new_surfaces)
    total_inserted = 0
    n_chunks = (len(surface_list) + _CHUNK_SIZE - 1) // _CHUNK_SIZE

    print(f"  Running Zeyrek on {len(surface_list):,} new surfaces in {n_chunks} chunks ...")

    for chunk_idx in range(n_chunks):
        chunk = surface_list[chunk_idx * _CHUNK_SIZE : (chunk_idx + 1) * _CHUNK_SIZE]
        results = analyze_bulk(chunk)

        entries: list[MorphEntry] = []
        for surface, canonical in results.items():
            if canonical is None:
                continue
            parts = canonical.split()
            lemma = parts[0] if parts else surface
            pos = "NOUN"
            for part in parts[1:]:
                if part in ("+Noun", "+Verb", "+Adj", "+Adv", "+Num", "+Pron"):
                    pos = part.lstrip("+").upper()
                    break
            entries.append(
                MorphEntry(
                    surface=surface,
                    lemma=lemma,
                    canonical_tags=canonical,
                    pos=pos,
                    source="zeyrek",
                    confidence=0.9,
                    frequency=1,
                    tier="bronze",
                )
            )

        inserted = db.bulk_insert(entries)
        total_inserted += inserted
        if (chunk_idx + 1) % 10 == 0 or chunk_idx == n_chunks - 1:
            print(f"    chunk {chunk_idx+1}/{n_chunks}  inserted={total_inserted:,}")

    # --------------------------------------------------------------------- #
    # 4. Re-evaluate tiers for surfaces that now have ≥2 sources
    # --------------------------------------------------------------------- #
    print("  Re-evaluating tiers for newly multi-source surfaces ...")
    updated = 0
    # Only check new surfaces (the existing ones were already evaluated in Gen 0)
    for surface in new_surfaces:
        entries = db.query_surface(surface)
        if len(entries) <= 1:
            continue
        tier, agreement = tier_from_entries(entries)
        for entry in entries:
            db.update_tier(surface, entry.source, tier, agreement)
        updated += 1

    print(f"  Tier upgrades applied to {updated:,} surfaces")
    return total_inserted
