"""CheckList-style behavioral tests for Turkish morphological analysis.

Three test classes are supported:

* **MFT** — minimum functionality: basic case, possessive, negation,
  and vowel-harmony paradigms must produce the right tag sequence.
* **INV** — invariance: swapping the root for a synonym must leave the
  tag sequence unchanged.
* **DIR** — directional: adding a single suffix must add exactly one
  new tag to the output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TestCase:
    kind: str
    input: str
    expected_tags: tuple[str, ...]
    phenomenon: str


@dataclass
class CheckListReport:
    per_phenomenon: dict[str, dict[str, int]] = field(default_factory=dict)


_CASES = ["+NOM", "+ACC", "+DAT", "+LOC", "+ABL", "+GEN"]
_PERSONS = ["+POSS.1SG", "+POSS.2SG", "+POSS.3SG", "+POSS.1PL", "+POSS.2PL", "+POSS.3PL"]


def generate_mft_tests() -> list[TestCase]:
    cases: list[TestCase] = []
    # 6 cases on "ev" (front harmony)
    for c in _CASES:
        cases.append(TestCase("MFT", f"ev{_case_suffix(c)}", ("+NOUN", c), "case"))
    # 6 possessives on "kitap" (book)
    for p in _PERSONS:
        cases.append(TestCase("MFT", f"kitap+{p}", ("+NOUN", p), "possessive"))
    # Vowel harmony plural: back vs front
    cases.append(TestCase("MFT", "arabalar", ("+NOUN", "+PLU"), "vowel_harmony"))
    cases.append(TestCase("MFT", "evler", ("+NOUN", "+PLU"), "vowel_harmony"))
    # Negation
    cases.append(TestCase("MFT", "gelme", ("+VERB", "+NEG"), "negation"))
    return cases


def _case_suffix(case_tag: str) -> str:
    return {"+NOM": "", "+ACC": "i", "+DAT": "e", "+LOC": "de", "+ABL": "den", "+GEN": "in"}[case_tag]


def generate_inv_tests() -> list[TestCase]:
    # Synonym swap: kitabı <-> defteri both should carry +POSS.3SG.
    return [
        TestCase("INV", "kitabı", ("+NOUN", "+POSS.3SG"), "synonym_swap"),
        TestCase("INV", "defteri", ("+NOUN", "+POSS.3SG"), "synonym_swap"),
    ]


def generate_dir_tests() -> list[TestCase]:
    # Adding +NEG must only add +NEG.
    return [
        TestCase("DIR", "geldi", ("+VERB", "+PAST"), "negation_add_base"),
        TestCase("DIR", "gelmedi", ("+VERB", "+NEG", "+PAST"), "negation_add_result"),
        TestCase("DIR", "ev", ("+NOUN",), "plural_add_base"),
        TestCase("DIR", "evler", ("+NOUN", "+PLU"), "plural_add_result"),
    ]


def run_checklist(predict_fn, output_path: str | Path | None = None) -> CheckListReport:
    """Run MFT/INV/DIR tests against ``predict_fn: str -> tuple[str, ...]``."""
    report = CheckListReport()
    all_cases = generate_mft_tests() + generate_inv_tests() + generate_dir_tests()
    for tc in all_cases:
        bucket = report.per_phenomenon.setdefault(
            tc.phenomenon, {"pass": 0, "fail": 0}
        )
        try:
            pred = tuple(predict_fn(tc.input))
        except Exception:
            bucket["fail"] += 1
            continue
        if pred == tc.expected_tags:
            bucket["pass"] += 1
        else:
            bucket["fail"] += 1
    if output_path is not None:
        _write_report(report, Path(output_path))
    return report


def _write_report(report: CheckListReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# CheckList Report\n", "| Phenomenon | Pass | Fail |", "|---|---|---|"]
    for name, counts in report.per_phenomenon.items():
        lines.append(f"| {name} | {counts['pass']} | {counts['fail']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
