"""Tests for offline functionality in aksu.data.build.preprocess.

Covers _load_local_jsonl (used when --local-jsonl is passed on HPC without
internet) and the sources dataclass definitions, without requiring HuggingFace
network access.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aksu.data.build.preprocess import _load_local_jsonl
from aksu.data.build.sources import SOURCES, Source


class TestLoadLocalJsonl:
    def test_plain_text_lines(self, tmp_path: Path) -> None:
        f = tmp_path / "corpus.jsonl"
        f.write_text("Merhaba dünya\nEvde oturuyorum\n", encoding="utf-8")
        result = _load_local_jsonl(f)
        assert result == ["Merhaba dünya", "Evde oturuyorum"]

    def test_json_objects_with_text_field(self, tmp_path: Path) -> None:
        f = tmp_path / "corpus.jsonl"
        lines = [
            json.dumps({"text": "Birinci cümle.", "source": "oscar-tr"}),
            json.dumps({"text": "İkinci cümle.", "lang": "tr"}),
        ]
        f.write_text("\n".join(lines) + "\n", encoding="utf-8")
        result = _load_local_jsonl(f)
        assert result == ["Birinci cümle.", "İkinci cümle."]

    def test_json_objects_with_sentence_field(self, tmp_path: Path) -> None:
        f = tmp_path / "corpus.jsonl"
        f.write_text(json.dumps({"sentence": "Cümle var."}) + "\n", encoding="utf-8")
        result = _load_local_jsonl(f)
        assert result == ["Cümle var."]

    def test_json_string_objects(self, tmp_path: Path) -> None:
        f = tmp_path / "corpus.jsonl"
        f.write_text('"düz metin"\n"başka metin"\n', encoding="utf-8")
        result = _load_local_jsonl(f)
        assert result == ["düz metin", "başka metin"]

    def test_empty_lines_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "corpus.jsonl"
        f.write_text("cümle bir\n\n\ncümle iki\n", encoding="utf-8")
        result = _load_local_jsonl(f)
        assert result == ["cümle bir", "cümle iki"]

    def test_empty_text_field_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "corpus.jsonl"
        lines = [
            json.dumps({"text": "Geçerli cümle."}),
            json.dumps({"text": ""}),
        ]
        f.write_text("\n".join(lines) + "\n", encoding="utf-8")
        result = _load_local_jsonl(f)
        assert result == ["Geçerli cümle."]

    def test_custom_text_field(self, tmp_path: Path) -> None:
        f = tmp_path / "corpus.jsonl"
        f.write_text(json.dumps({"content": "özel alan"}) + "\n", encoding="utf-8")
        result = _load_local_jsonl(f, text_field="content")
        assert result == ["özel alan"]

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "corpus.jsonl"
        f.write_text("", encoding="utf-8")
        result = _load_local_jsonl(f)
        assert result == []

    def test_mixed_json_and_plain(self, tmp_path: Path) -> None:
        f = tmp_path / "corpus.jsonl"
        lines = [
            json.dumps({"text": "json satırı"}),
            "düz metin satırı",
        ]
        f.write_text("\n".join(lines) + "\n", encoding="utf-8")
        result = _load_local_jsonl(f)
        assert result == ["json satırı", "düz metin satırı"]


class TestSources:
    def test_sources_is_non_empty(self) -> None:
        assert len(SOURCES) > 0

    def test_all_sources_have_required_fields(self) -> None:
        for s in SOURCES:
            assert isinstance(s, Source)
            assert s.name
            assert s.url
            assert s.license

    def test_oscar_tr_exists(self) -> None:
        names = {s.name for s in SOURCES}
        assert "oscar-tr" in names

    def test_imst_not_redistributable(self) -> None:
        imst = next((s for s in SOURCES if "imst" in s.name.lower()), None)
        if imst is not None:
            assert not imst.redistribute, "IMST has NC license — must not redistribute"

    def test_redistributable_sources_have_permissive_licenses(self) -> None:
        non_permissive_markers = ["NC", "ND"]
        for s in SOURCES:
            if s.redistribute:
                for marker in non_permissive_markers:
                    assert marker not in s.license, (
                        f"Source {s.name!r} marked redistribute=True but license "
                        f"{s.license!r} contains {marker!r} (non-commercial/no-derivatives)"
                    )
