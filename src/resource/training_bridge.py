"""Export MorphDatabase entries as training data for TieredCorpusDataset.

Tier mapping:
  DB "gold"   → training tier "gold"
  DB "silver" → training tier "silver-auto"
  DB "bronze" → training tier "silver-agreed"
"""
from __future__ import annotations
import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

_TIER_MAP: dict[str, str] = {
    "gold": "gold",
    "silver": "silver-auto",
    "bronze": "silver-agreed",
}


def export_training_data(
    db_path: str,
    output_path: str,
    max_bronze: int = 100_000,
    min_frequency: int = 2,
    seed: int = 42,
) -> None:
    """Export resource DB entries as JSONL for TieredCorpusDataset.

    Strategy:
    - ALL gold entries (highest quality, UD-sourced).
    - ALL silver entries (multi-source agreement).
    - Up to max_bronze bronze entries by frequency (most useful).

    Output format (one JSON object per line)::

        {"surface": "evlerinden", "label": "ev +PLU +POSS.3SG +ABL",
         "tier": "gold", "pos": "NOUN"}

    Args:
        db_path: Path to the SQLite resource database.
        output_path: Output JSONL file path.
        max_bronze: Maximum number of bronze-tier entries to include.
        min_frequency: Minimum frequency filter for all tiers.
        seed: Random seed for reproducible sampling of bronze entries.
    """
    from resource.schema import MorphDatabase

    rng = random.Random(seed)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}

    with MorphDatabase(db_path) as db, open(output_path, "w", encoding="utf-8") as out:
        # Write gold entries (all)
        gold_count = _write_tier(db, out, "gold", min_frequency, limit=None)
        counts["gold"] = gold_count

        # Write silver entries (all)
        silver_count = _write_tier(db, out, "silver", min_frequency, limit=None)
        counts["silver-auto"] = silver_count

        # Write bronze entries (capped + sampled by frequency)
        bronze_count = _write_tier(db, out, "bronze", min_frequency, limit=max_bronze)
        counts["silver-agreed"] = bronze_count

    total = sum(counts.values())
    print(f"Exported {total:,} training samples:")
    for tier, count in counts.items():
        print(f"  {tier:14s}: {count:,}")


def _write_tier(
    db: object,
    out: object,
    db_tier: str,
    min_frequency: int,
    limit: int | None,
) -> int:
    """Write entries of a specific tier to the output file.

    Args:
        db: MorphDatabase instance.
        out: Open file handle for writing.
        db_tier: Database tier to query ("gold", "silver", "bronze").
        min_frequency: Minimum frequency filter.
        limit: Maximum number of entries to write (None = no limit).

    Returns:
        Number of entries written.
    """
    training_tier = _TIER_MAP.get(db_tier, "silver-auto")

    # Deduplicate: for each surface, prefer gold > imst > boun > unimorph > zeyrek
    # We query all entries for the tier and deduplicate by surface
    from resource.schema import MorphEntry

    entries: list[MorphEntry] = []

    # Use direct DB query for efficiency
    cursor = db._conn.execute(
        "SELECT surface, lemma, canonical_tags, pos, source, confidence, frequency, tier "
        "FROM morphemes WHERE tier=? AND frequency>=? ORDER BY frequency DESC",
        (db_tier, min_frequency),
    )

    seen_surfaces: set[str] = set()
    for row in cursor.fetchall():
        entry = MorphEntry(*row)
        if entry.surface in seen_surfaces:
            continue
        seen_surfaces.add(entry.surface)
        entries.append(entry)
        if limit and len(entries) >= limit:
            break

    count = 0
    for entry in entries:
        record = {
            "surface": entry.surface,
            "label": entry.canonical_tags,
            "tier": training_tier,
            "pos": entry.pos,
        }
        out.write(json.dumps(record, ensure_ascii=False) + "\n")  # type: ignore[union-attr]
        count += 1

    return count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export resource DB to training JSONL")
    parser.add_argument("--db", default="data/resource/tr_gold_morph.db")
    parser.add_argument("--output", default="data/resource/training_export.jsonl")
    parser.add_argument("--max-bronze", type=int, default=100_000)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()

    export_training_data(args.db, args.output, args.max_bronze, args.min_frequency)
