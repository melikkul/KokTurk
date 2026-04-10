"""MorphEntry dataclass and SQLite-backed MorphDatabase."""
from __future__ import annotations
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MorphEntry:
    """A single morphological form-analysis pair.

    Args:
        surface: Inflected surface form (e.g., "evlerinden").
        lemma: Dictionary form / lemma (e.g., "ev").
        canonical_tags: Full canonical label (e.g., "ev +PLU +POSS.3SG +ABL").
        pos: Part-of-speech tag (e.g., "NOUN", "VERB").
        source: Origin of the entry: "boun" | "imst" | "unimorph" | "zeyrek".
        confidence: Confidence score 0.0–1.0.
        frequency: Observed corpus frequency (default 1).
        tier: Quality tier: "gold" | "silver" | "bronze".
    """
    surface: str
    lemma: str
    canonical_tags: str
    pos: str
    source: str
    confidence: float
    frequency: int
    tier: str


class MorphDatabase:
    """SQLite-backed database for MorphEntry objects.

    Schema: single `morphemes` table with columns matching MorphEntry fields.
    Unique index on (surface, source) — same surface from different sources
    is stored as separate rows.

    Args:
        db_path: Path to SQLite database file. Created if it does not exist.
    """

    _CREATE_TABLE = """
        CREATE TABLE IF NOT EXISTS morphemes (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            surface    TEXT NOT NULL,
            lemma      TEXT NOT NULL,
            canonical_tags TEXT NOT NULL,
            pos        TEXT NOT NULL,
            source     TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 1.0,
            frequency  INTEGER NOT NULL DEFAULT 1,
            tier       TEXT NOT NULL DEFAULT 'bronze',
            UNIQUE(surface, source)
        )
    """
    _CREATE_INDEX = """
        CREATE INDEX IF NOT EXISTS idx_surface ON morphemes(surface)
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute(self._CREATE_TABLE)
        self._conn.execute(self._CREATE_INDEX)
        self._conn.commit()

    # ------------------------------------------------------------------ #
    # Write operations
    # ------------------------------------------------------------------ #

    def insert(self, entry: MorphEntry) -> None:
        """Insert a single entry; silently ignore if (surface, source) already exists."""
        self._conn.execute(
            "INSERT OR IGNORE INTO morphemes "
            "(surface, lemma, canonical_tags, pos, source, confidence, frequency, tier) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (entry.surface, entry.lemma, entry.canonical_tags, entry.pos,
             entry.source, entry.confidence, entry.frequency, entry.tier),
        )
        self._conn.commit()

    def bulk_insert(self, entries: list[MorphEntry]) -> int:
        """Insert many entries; returns number of rows actually inserted."""
        rows = [
            (e.surface, e.lemma, e.canonical_tags, e.pos,
             e.source, e.confidence, e.frequency, e.tier)
            for e in entries
        ]
        cursor = self._conn.executemany(
            "INSERT OR IGNORE INTO morphemes "
            "(surface, lemma, canonical_tags, pos, source, confidence, frequency, tier) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        return cursor.rowcount

    def update_tier(self, surface: str, source: str, tier: str, confidence: float) -> None:
        """Update tier and confidence for a specific (surface, source) row."""
        self._conn.execute(
            "UPDATE morphemes SET tier=?, confidence=? WHERE surface=? AND source=?",
            (tier, confidence, surface, source),
        )
        self._conn.commit()

    # ------------------------------------------------------------------ #
    # Read operations
    # ------------------------------------------------------------------ #

    def query_surface(self, surface: str) -> list[MorphEntry]:
        """Return all entries for a given surface form (from all sources)."""
        cursor = self._conn.execute(
            "SELECT surface, lemma, canonical_tags, pos, source, confidence, frequency, tier "
            "FROM morphemes WHERE surface=?",
            (surface,),
        )
        return [MorphEntry(*row) for row in cursor.fetchall()]

    def get_all_surfaces(self) -> list[str]:
        """Return sorted list of all unique surface forms."""
        cursor = self._conn.execute("SELECT DISTINCT surface FROM morphemes ORDER BY surface")
        return [row[0] for row in cursor.fetchall()]

    def export_jsonl(
        self,
        path: str | Path,
        tier_filter: list[str] | None = None,
        min_frequency: int = 1,
    ) -> int:
        """Export entries as JSONL for training pipeline.

        Args:
            path: Output file path.
            tier_filter: If given, only export entries in these tiers.
            min_frequency: Minimum frequency to include.

        Returns:
            Number of entries written.
        """
        query = "SELECT surface, lemma, canonical_tags, pos, source, confidence, frequency, tier FROM morphemes"
        params: list[object] = []

        conditions = [f"frequency >= {min_frequency}"]
        if tier_filter:
            placeholders = ",".join("?" * len(tier_filter))
            conditions.append(f"tier IN ({placeholders})")
            params.extend(tier_filter)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        cursor = self._conn.execute(query, params)
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(path, "w", encoding="utf-8") as f:
            for row in cursor.fetchall():
                entry = MorphEntry(*row)
                f.write(json.dumps({
                    "surface": entry.surface,
                    "lemma": entry.lemma,
                    "canonical_tags": entry.canonical_tags,
                    "pos": entry.pos,
                    "source": entry.source,
                    "confidence": entry.confidence,
                    "frequency": entry.frequency,
                    "tier": entry.tier,
                }, ensure_ascii=False) + "\n")
                count += 1
        return count

    def get_stats(self) -> dict[str, object]:
        """Return statistics dict: total, by_tier, by_source, unique_surfaces, unique_lemmas."""
        total = self._conn.execute("SELECT COUNT(*) FROM morphemes").fetchone()[0]
        unique_surfaces = self._conn.execute("SELECT COUNT(DISTINCT surface) FROM morphemes").fetchone()[0]
        unique_lemmas = self._conn.execute("SELECT COUNT(DISTINCT lemma) FROM morphemes").fetchone()[0]

        by_tier: dict[str, int] = {}
        for row in self._conn.execute("SELECT tier, COUNT(*) FROM morphemes GROUP BY tier"):
            by_tier[row[0]] = row[1]

        by_source: dict[str, int] = {}
        for row in self._conn.execute("SELECT source, COUNT(*) FROM morphemes GROUP BY source"):
            by_source[row[0]] = row[1]

        return {
            "total": total,
            "unique_surfaces": unique_surfaces,
            "unique_lemmas": unique_lemmas,
            "by_tier": by_tier,
            "by_source": by_source,
        }

    def export_multi_schema(self, output_dir: str | Path) -> dict[str, int]:
        """Export resource in 3 schemas + stats JSON.

        Files written:
          1. ``tr_gold_morph_canonical.tsv`` — surface, lemma, canonical_tags,
             morpheme_boundaries, frequency, tier
          2. ``tr_gold_morph_ud.tsv`` — surface, lemma, upos, ud_features,
             frequency, tier
          3. ``tr_gold_morph_unimorph.tsv`` — lemma, surface, unimorph_features
             (UniMorph 3-column format)
          4. ``tr_gold_morph_stats.json`` — summary statistics

        Returns:
            Dict with counts per schema file.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        rows = self._conn.execute(
            "SELECT surface, lemma, canonical_tags, pos, source, confidence, "
            "frequency, tier, COALESCE(morpheme_boundaries, '') "
            "FROM morphemes ORDER BY frequency DESC"
        ).fetchall()

        # 1. Canonical TSV
        canon_path = out / "tr_gold_morph_canonical.tsv"
        with open(canon_path, "w", encoding="utf-8") as f:
            f.write("surface\tlemma\tcanonical_tags\tmorpheme_boundaries\tfrequency\ttier\n")
            for r in rows:
                f.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[8]}\t{r[6]}\t{r[7]}\n")

        # 2. UD TSV (simplified: POS + canonical tags as-is)
        ud_path = out / "tr_gold_morph_ud.tsv"
        with open(ud_path, "w", encoding="utf-8") as f:
            f.write("surface\tlemma\tupos\tud_features\tfrequency\ttier\n")
            for r in rows:
                f.write(f"{r[0]}\t{r[1]}\t{r[3]}\t{r[2]}\t{r[6]}\t{r[7]}\n")

        # 3. UniMorph 3-column (lemma, form, features)
        um_path = out / "tr_gold_morph_unimorph.tsv"
        with open(um_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(f"{r[1]}\t{r[0]}\t{r[2]}\n")

        # 4. Stats JSON
        stats = self.get_stats()
        with_freq = self._conn.execute(
            "SELECT COUNT(*) FROM morphemes WHERE frequency > 1"
        ).fetchone()[0]
        with_bounds = self._conn.execute(
            "SELECT COUNT(*) FROM morphemes WHERE morpheme_boundaries != '' AND morpheme_boundaries IS NOT NULL"
        ).fetchone()[0]
        stats["with_frequency"] = with_freq
        stats["with_boundaries"] = with_bounds

        stats_path = out / "tr_gold_morph_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        return {
            "canonical": len(rows),
            "ud": len(rows),
            "unimorph": len(rows),
        }

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> MorphDatabase:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
