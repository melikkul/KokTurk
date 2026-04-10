"""Export corpus in multiple formats (TSV, CoNLL-U-like, stats JSON).

Usage:
    PYTHONPATH=src python src/data/export.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

CORPUS_PATH = Path("data/gold/tr_gold_morph_v1.jsonl")
OUTPUT_DIR = Path("data/gold")


def export_corpus(
    corpus_path: Path = CORPUS_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Export the corpus in TSV, CoNLL-U-like, and stats JSON formats."""
    records: list[dict[str, object]] = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    stem = corpus_path.stem

    # TSV: surface \t root \t tags
    tsv_path = output_dir / f"{stem}.tsv"
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("surface\troot\ttags\ttier\n")
        for r in records:
            label = str(r.get("label", ""))
            parts = label.split()
            root = parts[0] if parts else ""
            tags = " ".join(parts[1:]) if len(parts) > 1 else ""
            f.write(f"{r['surface']}\t{root}\t{tags}\t{r.get('tier', '')}\n")

    # CoNLL-U-like: grouped by sentence
    conllu_path = output_dir / f"{stem}.conllu"
    from collections import defaultdict

    by_sent: dict[str, list[dict[str, object]]] = defaultdict(list)
    for r in records:
        by_sent[str(r["sentence_id"])].append(r)

    with open(conllu_path, "w", encoding="utf-8") as f:
        for sid in sorted(by_sent.keys()):
            tokens = sorted(by_sent[sid], key=lambda x: int(x.get("token_idx", 0)))
            f.write(f"# sent_id = {sid}\n")
            for t in tokens:
                idx = int(t.get("token_idx", 0)) + 1
                surface = t.get("surface", "_")
                label = str(t.get("label", "_"))
                parts = label.split()
                root = parts[0] if parts else "_"
                tags_str = "|".join(parts[1:]) if len(parts) > 1 else "_"
                tier = t.get("tier", "_")
                f.write(f"{idx}\t{surface}\t{root}\t{tags_str}\t{tier}\n")
            f.write("\n")

    # Stats JSON
    stats_path = output_dir / f"{stem}_stats.json"
    tier_counts = Counter(str(r.get("tier", "?")) for r in records)
    root_counts = Counter()
    tag_counts: Counter[str] = Counter()
    for r in records:
        parts = str(r.get("label", "")).split()
        if parts:
            root_counts[parts[0]] += 1
        for p in parts[1:]:
            tag_counts[p] += 1

    stats = {
        "total_tokens": len(records),
        "tiers": dict(tier_counts),
        "unique_roots": len(root_counts),
        "unique_tags": len(tag_counts),
        "top50_roots": root_counts.most_common(50),
        "top30_tags": tag_counts.most_common(30),
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("Exported:")
    print(f"  TSV:     {tsv_path}")
    print(f"  CoNLL-U: {conllu_path}")
    print(f"  Stats:   {stats_path}")


def main() -> None:
    export_corpus()


if __name__ == "__main__":
    main()
