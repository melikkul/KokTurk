"""Personal Information Identifier (PII) scanner for Turkish text corpora.

Scans training data files for potential PII using Turkish-specific heuristic
regex patterns. This is an audit tool, not a production NER system.

Usage::

    PYTHONPATH=src python -m data.pii_scan --corpus-dir data/

"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Data types
# ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PIIFinding:
    """A single PII match in a corpus file."""

    file_path: str
    line_number: int
    pii_type: str  # "tc_kimlik" | "phone" | "email" | "iban"
    matched_text: str
    context: str  # surrounding ~40 chars for human review
    severity: str  # "high" | "medium" | "low"


# ------------------------------------------------------------------
# Turkish-specific PII patterns
# ------------------------------------------------------------------

TC_KIMLIK_RE = re.compile(r"\b\d{11}\b")
PHONE_RE = re.compile(r"(?:\+90|0)\s*\d{3}\s*\d{3}\s*\d{2}\s*\d{2}")
EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
IBAN_RE = re.compile(
    r"\bTR\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{2}\b"
)

PII_PATTERNS: dict[str, tuple[re.Pattern[str], str]] = {
    "tc_kimlik": (TC_KIMLIK_RE, "high"),
    "phone": (PHONE_RE, "high"),
    "email": (EMAIL_RE, "high"),
    "iban": (IBAN_RE, "high"),
}


# ------------------------------------------------------------------
# Core scanning functions
# ------------------------------------------------------------------


def _extract_context(line: str, start: int, end: int, width: int = 20) -> str:
    """Return surrounding context around a match."""
    ctx_start = max(0, start - width)
    ctx_end = min(len(line), end + width)
    return line[ctx_start:ctx_end].strip()


def scan_file(file_path: str | Path) -> list[PIIFinding]:
    """Scan a single file for PII patterns.

    Reads line by line, applies each regex, returns findings.
    """
    file_path = Path(file_path)
    findings: list[PIIFinding] = []

    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        logger.warning("Cannot read %s, skipping", file_path)
        return findings

    for line_num, line in enumerate(text.splitlines(), start=1):
        for pii_type, (pattern, severity) in PII_PATTERNS.items():
            for match in pattern.finditer(line):
                findings.append(
                    PIIFinding(
                        file_path=str(file_path),
                        line_number=line_num,
                        pii_type=pii_type,
                        matched_text=match.group(),
                        context=_extract_context(
                            line, match.start(), match.end()
                        ),
                        severity=severity,
                    )
                )

    return findings


def scan_corpus(
    corpus_dir: str | Path,
    extensions: tuple[str, ...] = (".jsonl", ".tsv", ".txt", ".conllu"),
) -> list[PIIFinding]:
    """Recursively scan all files under *corpus_dir* matching *extensions*.

    Returns findings sorted by (file_path, line_number).
    """
    corpus_dir = Path(corpus_dir)
    findings: list[PIIFinding] = []

    for ext in extensions:
        for file_path in sorted(corpus_dir.rglob(f"*{ext}")):
            file_findings = scan_file(file_path)
            findings.extend(file_findings)

    findings.sort(key=lambda f: (f.file_path, f.line_number))
    return findings


# ------------------------------------------------------------------
# Proper noun counting
# ------------------------------------------------------------------


def count_proper_nouns(corpus_path: str | Path) -> dict[str, int]:
    """Count tokens tagged with +Prop or +PROPN in a JSONL corpus.

    Each line should be a JSON object with a ``"label"`` field containing
    the morphological analysis string.

    Returns:
        ``{surface_form: count}`` sorted by frequency descending.
    """
    counter: Counter[str] = Counter()
    path = Path(corpus_path)

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        logger.warning("Cannot read %s", path)
        return {}

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        label = record.get("label", "")
        if "+Prop" in label or "+PROPN" in label:
            surface = record.get("surface", record.get("text", ""))
            if surface:
                counter[surface] += 1

    return dict(counter.most_common())


# ------------------------------------------------------------------
# Report generation
# ------------------------------------------------------------------


def generate_pii_report(
    findings: list[PIIFinding],
    proper_noun_counts: dict[str, int] | None = None,
    output_path: str | Path | None = None,
) -> str:
    """Generate a markdown PII audit report.

    Returns the markdown string. Writes to *output_path* if provided.
    """
    lines: list[str] = []
    lines.append("# PII Audit Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"Total PII candidates found: **{len(findings)}**")
    lines.append("")

    # By type
    by_type: dict[str, int] = {}
    for f in findings:
        by_type[f.pii_type] = by_type.get(f.pii_type, 0) + 1

    if by_type:
        lines.append("| PII Type | Count | Severity |")
        lines.append("|----------|-------|----------|")
        for pii_type, count in sorted(by_type.items()):
            severity = PII_PATTERNS.get(pii_type, (None, "unknown"))[1]
            lines.append(f"| {pii_type} | {count} | {severity} |")
        lines.append("")

    # By file
    by_file: dict[str, int] = {}
    for f in findings:
        by_file[f.file_path] = by_file.get(f.file_path, 0) + 1

    if by_file:
        lines.append("## Findings by File")
        lines.append("")
        lines.append("| File | Count |")
        lines.append("|------|-------|")
        for fp, count in sorted(by_file.items()):
            lines.append(f"| {fp} | {count} |")
        lines.append("")

    # Proper nouns
    if proper_noun_counts:
        lines.append("## Top 20 Proper Nouns")
        lines.append("")
        lines.append("| Token | Count |")
        lines.append("|-------|-------|")
        for token, count in list(proper_noun_counts.items())[:20]:
            lines.append(f"| {token} | {count} |")
        lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")
    if len(findings) == 0:
        lines.append("No PII candidates detected. Continue monitoring with each data update.")
    else:
        lines.append("1. Review all high-severity findings manually before model release.")
        lines.append("2. Consider redacting or replacing identified PII in training data.")
        lines.append("3. Re-run this scan after any data pipeline changes.")
        lines.append("4. Note: TC Kimlik detection is heuristic (any 11-digit number matches). Manual verification required.")
    lines.append("")

    report = "\n".join(lines)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        logger.info("PII report written to %s", out)

    return report


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------


def run_pii_scan(
    corpus_dir: str | Path = "data",
    corpus_jsonl: str | Path | None = None,
    output_path: str | Path = "models/benchmark/pii_report.md",
) -> list[PIIFinding]:
    """High-level runner: scan corpus, optionally count proper nouns, generate report."""
    findings = scan_corpus(corpus_dir)
    logger.info("Found %d PII candidates in %s", len(findings), corpus_dir)

    proper_nouns: dict[str, int] | None = None
    if corpus_jsonl is not None:
        proper_nouns = count_proper_nouns(corpus_jsonl)
        logger.info("Counted %d unique proper nouns", len(proper_nouns))

    generate_pii_report(findings, proper_nouns, output_path)
    return findings


def main() -> None:  # pragma: no cover
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="PII scan for Turkish corpus data"
    )
    parser.add_argument(
        "--corpus-dir", type=Path, default=Path("data"),
        help="Root directory to scan",
    )
    parser.add_argument(
        "--corpus-jsonl", type=Path, default=None,
        help="JSONL file for proper noun counting",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("models/benchmark/pii_report.md"),
        help="Output path for PII report",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    findings = run_pii_scan(args.corpus_dir, args.corpus_jsonl, args.output)
    print(f"Found {len(findings)} PII candidates")


if __name__ == "__main__":
    main()
