"""Tests for MorphEntry dataclass and MorphDatabase."""
from __future__ import annotations
import json
import tempfile
from pathlib import Path
import pytest

from resource.schema import MorphDatabase, MorphEntry

@pytest.fixture
def db(tmp_path):
    """Temporary in-memory-style database for testing."""
    return MorphDatabase(tmp_path / "test.db")


def _make_entry(surface="ev", source="boun", tier="gold", tags="ev +Noun") -> MorphEntry:
    return MorphEntry(
        surface=surface, lemma="ev", canonical_tags=tags,
        pos="NOUN", source=source, confidence=1.0, frequency=1, tier=tier,
    )


class TestMorphEntry:
    def test_frozen(self):
        entry = _make_entry()
        with pytest.raises(Exception):  # frozen=True prevents attribute assignment
            entry.surface = "changed"  # type: ignore[misc]

    def test_fields(self):
        entry = _make_entry()
        assert entry.surface == "ev"
        assert entry.tier == "gold"


class TestMorphDatabase:
    def test_insert_and_query(self, db):
        entry = _make_entry()
        db.insert(entry)
        results = db.query_surface("ev")
        assert len(results) == 1
        assert results[0].canonical_tags == "ev +Noun"

    def test_duplicate_insert_ignored(self, db):
        """Same (surface, source) insert twice — only one row stored."""
        e = _make_entry()
        db.insert(e)
        db.insert(e)  # second insert should be silently ignored
        results = db.query_surface("ev")
        assert len(results) == 1

    def test_different_sources_both_stored(self, db):
        """Same surface from two different sources → two rows."""
        e1 = _make_entry(source="boun")
        e2 = _make_entry(source="zeyrek")
        db.insert(e1)
        db.insert(e2)
        results = db.query_surface("ev")
        assert len(results) == 2
        sources = {r.source for r in results}
        assert sources == {"boun", "zeyrek"}

    def test_get_stats(self, db):
        for i, tier in enumerate(("gold", "gold", "silver", "bronze")):
            db.insert(_make_entry(surface=f"s_{i}", tier=tier))
        stats = db.get_stats()
        assert stats["total"] == 4
        assert stats["by_tier"]["gold"] == 2

    def test_export_jsonl(self, db, tmp_path):
        db.insert(_make_entry(surface="ev", tier="gold"))
        db.insert(_make_entry(surface="git", tier="bronze", source="zeyrek"))
        out = tmp_path / "out.jsonl"
        count = db.export_jsonl(out, tier_filter=["gold"])
        assert count == 1
        with open(out) as f:
            rec = json.loads(f.readline())
        assert rec["surface"] == "ev"

    def test_update_tier(self, db):
        db.insert(_make_entry(source="unimorph", tier="bronze"))
        db.update_tier("ev", "unimorph", "silver", 0.95)
        results = db.query_surface("ev")
        assert results[0].tier == "silver"
