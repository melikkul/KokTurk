"""Tag frequency analysis for the kokturk training corpus.

Counts every morphological tag (`+XYZ` token) across the training corpus,
produces a Zipfian frequency table, and classifies each tag into a frequency
bucket used downstream by stratified eval and paradigm augmentation.

Buckets (fraction of total tag occurrences):
    HIGH_FREQ  > 5%
    MID_FREQ   1%..5%
    LOW_FREQ   0.1%..1%
    RARE       < 0.1%

Corpus path resolution is tolerant: probes JSONL -> TSV -> canonical TSV and
fails loudly if none exist.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True, slots=True)
class TagStat:
    tag: str
    count: int
    percentage: float
    cumulative_percentage: float
    frequency_class: str


def _classify(pct: float) -> str:
    if pct > 5.0:
        return "HIGH_FREQ"
    if pct >= 1.0:
        return "MID_FREQ"
    if pct >= 0.1:
        return "LOW_FREQ"
    return "RARE"


def resolve_corpus_path(data_dir: Path | str = "data/gold") -> Path:
    """Probe known gold corpus locations and return the first that exists."""
    root = Path(data_dir)
    candidates = [
        root / "tr_gold_morph_v1.jsonl",
        root / "train.tsv",
        root / "tr_gold_morph_canonical.tsv",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"No gold corpus found under {root}. Tried: "
        + ", ".join(str(c) for c in candidates)
    )


def iter_labels(path: Path) -> Iterator[str]:
    """Yield canonical label strings (e.g. `ev +Noun +PLU +POSS.3SG`) from
    any supported corpus format.

    JSONL: one token per line with key `label`.
    TSV: tab-separated; assumes last column is the canonical label.
    """
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as f:
        if suffix == ".jsonl":
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                label = obj.get("label")
                if label:
                    yield label
        else:  # tsv
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                cols = line.split("\t")
                if len(cols) >= 2:
                    yield cols[-1]


def extract_tags(label: str) -> list[str]:
    """Extract `+TAG` tokens from a canonical label string."""
    return [tok for tok in label.split() if tok.startswith("+")]


def count_tags(path: Path) -> Counter[str]:
    c: Counter[str] = Counter()
    for label in iter_labels(path):
        c.update(extract_tags(label))
    return c


def build_frequency_table(counts: Counter[str]) -> list[TagStat]:
    total = sum(counts.values())
    if total == 0:
        return []
    stats: list[TagStat] = []
    cumulative = 0.0
    for tag, count in counts.most_common():
        pct = 100.0 * count / total
        cumulative += pct
        stats.append(
            TagStat(
                tag=tag,
                count=count,
                percentage=pct,
                cumulative_percentage=cumulative,
                frequency_class=_classify(pct),
            )
        )
    return stats


def format_markdown_table(stats: list[TagStat]) -> str:
    lines = [
        "| Rank | Tag | Count | % | Cum % | Class |",
        "| ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for i, s in enumerate(stats, 1):
        lines.append(
            f"| {i} | `{s.tag}` | {s.count} | {s.percentage:.3f} | "
            f"{s.cumulative_percentage:.2f} | {s.frequency_class} |"
        )
    return "\n".join(lines)


def write_json(stats: list[TagStat], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "total_tag_occurrences": sum(s.count for s in stats),
        "unique_tags": len(stats),
        "tags": [
            {
                "tag": s.tag,
                "count": s.count,
                "percentage": s.percentage,
                "cumulative_percentage": s.cumulative_percentage,
                "frequency_class": s.frequency_class,
            }
            for s in stats
        ],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def run(
    data_dir: Path | str = "data/gold",
    output_path: Path | str = "models/benchmark/tag_frequency.json",
) -> list[TagStat]:
    corpus = resolve_corpus_path(data_dir)
    counts = count_tags(corpus)
    stats = build_frequency_table(counts)
    write_json(stats, Path(output_path))
    return stats


def main() -> None:
    stats = run()
    print(format_markdown_table(stats))
    print(
        f"\nTotal tag occurrences: {sum(s.count for s in stats)}  "
        f"unique tags: {len(stats)}"
    )


if __name__ == "__main__":
    main()
