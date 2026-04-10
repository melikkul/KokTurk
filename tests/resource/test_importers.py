"""Tests for resource importers."""
from __future__ import annotations
import tempfile
from pathlib import Path
import pytest

from resource.schema import MorphDatabase, MorphEntry
from resource.corpus_processor import parse_conllu_file
from resource.importers.boun import import_boun
from resource.tag_mappings import ud_feats_to_canonical, unimorph_tags_to_canonical


# Minimal CoNLL-U fixture
CONLLU_FIXTURE = """\
# sent_id = test_1
# text = Evlerinden gitti.
1\tevlerinden\tev\tNOUN\t_\tCase=Abl|Number=Plur|Number[psor]=Sing|Person=3|Person[psor]=3\t2\tobl\t_\t_
2\tgitti\tgit\tVERB\t_\tAspect=Perf|Mood=Ind|Number=Sing|Person=3|Polarity=Pos|Tense=Past\t0\troot\t_\tSpaceAfter=No
3\t.\t.\tPUNCT\t_\t_\t2\tpunct\t_\t_

"""


@pytest.fixture
def conllu_file(tmp_path):
    f = tmp_path / "test.conllu"
    f.write_text(CONLLU_FIXTURE, encoding="utf-8")
    return f


@pytest.fixture
def db(tmp_path):
    return MorphDatabase(tmp_path / "test.db")


class TestParseConllu:
    def test_returns_morph_entries(self, conllu_file):
        entries = parse_conllu_file(conllu_file, source="boun", tier="gold")
        # Should have 2 entries (evlerinden and gitti), not the PUNCT
        assert len(entries) == 2

    def test_skips_punct(self, conllu_file):
        entries = parse_conllu_file(conllu_file, source="boun", tier="gold")
        surfaces = [e.surface for e in entries]
        assert "." not in surfaces

    def test_entry_source(self, conllu_file):
        entries = parse_conllu_file(conllu_file, source="imst", tier="gold")
        assert all(e.source == "imst" for e in entries)

    def test_entry_tier(self, conllu_file):
        entries = parse_conllu_file(conllu_file, source="boun", tier="gold")
        assert all(e.tier == "gold" for e in entries)


class TestImportBoun:
    def test_raises_if_dir_missing(self, db, tmp_path):
        with pytest.raises(FileNotFoundError):
            import_boun(tmp_path / "nonexistent", db)

    def test_imports_to_db(self, conllu_file, db):
        # Create a directory with the conllu file
        conllu_dir = conllu_file.parent
        import_boun(conllu_dir, db)
        stats = db.get_stats()
        assert stats["total"] > 0
        assert stats["by_source"].get("boun", 0) > 0


class TestTagMappings:
    def test_ud_noun_loc(self):
        result = ud_feats_to_canonical("ev", "NOUN", "Case=Loc|Number=Sing|Person=3")
        assert "+LOC" in result
        assert result.startswith("ev")

    def test_ud_verb_past(self):
        result = ud_feats_to_canonical("git", "VERB", "Aspect=Perf|Tense=Past|Number=Sing|Person=3")
        assert "+PAST" in result

    def test_unimorph_noun_pl_abl(self):
        result = unimorph_tags_to_canonical("ev", "evlerden", "N;ABL;PL")
        assert "+ABL" in result
        assert "+PLU" in result

    def test_ud_feats_starts_with_lemma(self):
        result = ud_feats_to_canonical("kitap", "NOUN", "Case=Gen|Number=Sing")
        assert result.startswith("kitap")
