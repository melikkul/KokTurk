"""Turkish morphological minimal pair challenge set.

Loads pairs from :file:`configs/eval/minimal_pairs.yaml` and scores a
``predict_fn: str -> tuple[str, ...]`` model against each phenomenon.

For every pair, the model is asked to analyze *both* surface forms and
must return the expected tag tuple for each side of the pair. A pair
passes only if both analyses are correct.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None


DEFAULT_CONFIG = Path("configs/eval/minimal_pairs.yaml")


@dataclass(frozen=True, slots=True)
class Pair:
    phenomenon: str
    acceptable_word: str
    acceptable_tags: tuple[str, ...]
    unacceptable_word: str
    unacceptable_tags: tuple[str, ...]


@dataclass
class MinimalPairReport:
    per_phenomenon: dict[str, dict[str, int]] = field(default_factory=dict)
    total_pass: int = 0
    total_fail: int = 0


def load_pairs(path: str | Path = DEFAULT_CONFIG) -> list[Pair]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load minimal_pairs.yaml")
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    pairs: list[Pair] = []
    for entry in data.get("pairs", []):
        pairs.append(
            Pair(
                phenomenon=entry["phenomenon"],
                acceptable_word=entry["acceptable"]["word"],
                acceptable_tags=tuple(entry["acceptable"]["tags"]),
                unacceptable_word=entry["unacceptable"]["word"],
                unacceptable_tags=tuple(entry["unacceptable"]["tags"]),
            )
        )
    return pairs


def evaluate_minimal_pairs(
    predict_fn,
    pairs: list[Pair] | None = None,
    config_path: str | Path = DEFAULT_CONFIG,
    output_path: str | Path | None = None,
) -> MinimalPairReport:
    if pairs is None:
        pairs = load_pairs(config_path)
    report = MinimalPairReport()
    for p in pairs:
        bucket = report.per_phenomenon.setdefault(p.phenomenon, {"pass": 0, "fail": 0})
        try:
            a = tuple(predict_fn(p.acceptable_word))
            b = tuple(predict_fn(p.unacceptable_word))
        except Exception:
            bucket["fail"] += 1
            report.total_fail += 1
            continue
        if a == p.acceptable_tags and b == p.unacceptable_tags:
            bucket["pass"] += 1
            report.total_pass += 1
        else:
            bucket["fail"] += 1
            report.total_fail += 1
    if output_path is not None:
        _write_report(report, Path(output_path))
    return report


def _write_report(report: MinimalPairReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Minimal Pair Report\n",
        f"Total pass: {report.total_pass}",
        f"Total fail: {report.total_fail}\n",
        "| Phenomenon | Pass | Fail |",
        "|---|---|---|",
    ]
    for name, counts in report.per_phenomenon.items():
        lines.append(f"| {name} | {counts['pass']} | {counts['fail']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
