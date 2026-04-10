"""Tests for resource training data export."""
from __future__ import annotations
import json
from pathlib import Path
import pytest

from resource.schema import MorphDatabase, MorphEntry
from resource.training_bridge import export_training_data


@pytest.fixture
def populated_db(tmp_path):
    db = MorphDatabase(tmp_path / "test.db")

    # Add gold entries
    for i in range(5):
        db.insert(MorphEntry(
            surface=f"gold_{i}", lemma=f"root_{i}",
            canonical_tags=f"root_{i} +Noun", pos="NOUN",
            source="boun", confidence=1.0, frequency=3, tier="gold",
        ))
    # Add silver entries
    for i in range(3):
        db.insert(MorphEntry(
            surface=f"silver_{i}", lemma=f"sroot_{i}",
            canonical_tags=f"sroot_{i} +Noun +PLU", pos="NOUN",
            source="unimorph", confidence=0.95, frequency=2, tier="silver",
        ))
    # Add bronze entries
    for i in range(4):
        db.insert(MorphEntry(
            surface=f"bronze_{i}", lemma=f"broot_{i}",
            canonical_tags=f"broot_{i} +Verb", pos="VERB",
            source="zeyrek", confidence=0.8, frequency=1, tier="bronze",
        ))
    return db


class TestExportTrainingData:
    def test_export_creates_file(self, populated_db, tmp_path):
        out = tmp_path / "train.jsonl"
        export_training_data(str(populated_db.db_path), str(out))
        assert out.exists()

    def test_tier_mapping(self, populated_db, tmp_path):
        """DB tiers must map correctly to training tiers."""
        out = tmp_path / "train.jsonl"
        export_training_data(str(populated_db.db_path), str(out), min_frequency=1)

        records = [json.loads(line) for line in out.read_text().splitlines()]
        tiers_found = {r["tier"] for r in records}

        # gold → "gold", silver → "silver-auto", bronze → "silver-agreed"
        assert "gold" in tiers_found
        assert "silver-auto" in tiers_found
        assert "silver-agreed" in tiers_found

    def test_record_format(self, populated_db, tmp_path):
        """Each record must have surface, label, tier, pos."""
        out = tmp_path / "train.jsonl"
        export_training_data(str(populated_db.db_path), str(out), min_frequency=1)

        records = [json.loads(line) for line in out.read_text().splitlines()]
        assert len(records) > 0
        for rec in records:
            assert "surface" in rec
            assert "label" in rec
            assert "tier" in rec
            assert "pos" in rec

    def test_max_bronze_limit(self, populated_db, tmp_path):
        """max_bronze limits bronze-tier entries."""
        out = tmp_path / "train.jsonl"
        export_training_data(str(populated_db.db_path), str(out), max_bronze=2, min_frequency=1)

        records = [json.loads(line) for line in out.read_text().splitlines()]
        silver_agreed = [r for r in records if r["tier"] == "silver-agreed"]
        assert len(silver_agreed) <= 2
