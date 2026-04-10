"""Tests for data.pii_scan."""

from __future__ import annotations

import json

import pytest

from data.pii_scan import (
    EMAIL_RE,
    IBAN_RE,
    PHONE_RE,
    TC_KIMLIK_RE,
    PIIFinding,
    count_proper_nouns,
    generate_pii_report,
    scan_corpus,
    scan_file,
)


# ------------------------------------------------------------------
# Regex pattern tests
# ------------------------------------------------------------------


class TestRegexPatterns:
    """Verify each PII regex matches expected strings."""

    def test_tc_kimlik_matches_11_digits(self):
        assert TC_KIMLIK_RE.search("12345678901") is not None

    def test_tc_kimlik_no_match_10_digits(self):
        assert TC_KIMLIK_RE.search("1234567890") is None

    def test_tc_kimlik_no_match_12_digits(self):
        # 12-digit number should NOT match as a single 11-digit group
        m = TC_KIMLIK_RE.search("123456789012")
        assert m is None

    def test_tc_kimlik_word_boundary(self):
        assert TC_KIMLIK_RE.search("TC: 12345678901 no") is not None

    def test_phone_plus90_format(self):
        assert PHONE_RE.search("+90 532 123 45 67") is not None

    def test_phone_plus90_compact(self):
        assert PHONE_RE.search("+905321234567") is not None

    def test_phone_zero_prefix(self):
        assert PHONE_RE.search("0532 123 45 67") is not None

    def test_phone_no_match_short(self):
        assert PHONE_RE.search("0532 123") is None

    def test_email_standard(self):
        assert EMAIL_RE.search("user@example.com") is not None

    def test_email_with_dots(self):
        assert EMAIL_RE.search("first.last@uni.edu.tr") is not None

    def test_email_no_match_incomplete(self):
        assert EMAIL_RE.search("user@") is None

    def test_iban_turkish_compact(self):
        assert IBAN_RE.search("TR330006100519786457841326") is not None

    def test_iban_turkish_with_spaces(self):
        assert IBAN_RE.search("TR33 0006 1005 1978 6457 8413 26") is not None

    def test_iban_no_match_non_tr(self):
        assert IBAN_RE.search("DE330006100519786457841326") is None


# ------------------------------------------------------------------
# scan_file tests
# ------------------------------------------------------------------


class TestScanFile:
    def test_finds_email_in_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("line one\ncontact: test@mail.com here\nline three\n")
        findings = scan_file(f)
        assert len(findings) == 1
        assert findings[0].pii_type == "email"
        assert findings[0].line_number == 2
        assert findings[0].matched_text == "test@mail.com"

    def test_finds_multiple_types(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("+905321234567\nsome text\ninfo@test.org\n")
        findings = scan_file(f)
        types = {fin.pii_type for fin in findings}
        assert "phone" in types
        assert "email" in types

    def test_empty_file_returns_empty(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        assert scan_file(f) == []

    def test_context_contains_match(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("call +905321234567 now\n")
        findings = scan_file(f)
        assert len(findings) >= 1
        assert "+905321234567" in findings[0].context


# ------------------------------------------------------------------
# scan_corpus tests
# ------------------------------------------------------------------


class TestScanCorpus:
    def test_recursive_scan(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "a.jsonl").write_text('{"email": "a@b.com"}\n')
        (tmp_path / "b.txt").write_text("contact a@b.com\n")
        findings = scan_corpus(tmp_path)
        files = {f.file_path for f in findings}
        assert len(files) == 2

    def test_skips_non_matching_extensions(self, tmp_path):
        (tmp_path / "script.py").write_text("email = 'a@b.com'\n")
        findings = scan_corpus(tmp_path)
        assert len(findings) == 0


# ------------------------------------------------------------------
# count_proper_nouns tests
# ------------------------------------------------------------------


class TestCountProperNouns:
    def test_counts_propn_tags(self, tmp_path):
        f = tmp_path / "corpus.jsonl"
        records = [
            {"surface": "Ankara", "label": "Ankara +Prop +NOM"},
            {"surface": "ev", "label": "ev +Noun +PLU"},
            {"surface": "Ankara", "label": "Ankara +PROPN +DAT"},
        ]
        f.write_text("\n".join(json.dumps(r) for r in records))
        counts = count_proper_nouns(f)
        assert counts["Ankara"] == 2
        assert "ev" not in counts

    def test_empty_corpus(self, tmp_path):
        f = tmp_path / "empty.jsonl"
        f.write_text("")
        assert count_proper_nouns(f) == {}


# ------------------------------------------------------------------
# generate_pii_report tests
# ------------------------------------------------------------------


class TestGenerateReport:
    def test_report_contains_summary(self):
        findings = [
            PIIFinding("f.txt", 1, "email", "a@b.com", "a@b.com", "high"),
        ]
        report = generate_pii_report(findings)
        assert "## Summary" in report
        assert "1" in report

    def test_report_writes_file(self, tmp_path):
        findings = [
            PIIFinding("f.txt", 1, "email", "a@b.com", "a@b.com", "high"),
        ]
        out = tmp_path / "report.md"
        generate_pii_report(findings, output_path=out)
        assert out.exists()
        assert out.read_text().strip() != ""

    def test_empty_findings_report(self):
        report = generate_pii_report([])
        assert "0" in report
        assert "No PII candidates" in report


# ------------------------------------------------------------------
# Frozen dataclass tests
# ------------------------------------------------------------------


class TestFrozenDataclass:
    def test_pii_finding_is_frozen(self):
        finding = PIIFinding("f.txt", 1, "email", "a@b.com", "ctx", "high")
        with pytest.raises(AttributeError):
            finding.pii_type = "phone"  # type: ignore[misc]
