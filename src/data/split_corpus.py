"""Split the tiered corpus into train/val/test by sentence.

Preserves tier proportions across splits. Test split guaranteed
to contain gold tokens for unbiased evaluation.

Usage:
    PYTHONPATH=src python src/data/split_corpus.py
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

CORPUS_PATH = Path("data/gold/tr_gold_morph_v1.jsonl")
SPLITS_DIR = Path("data/splits")
SEED = 42


def split_corpus(
    corpus_path: Path = CORPUS_PATH,
    output_dir: Path = SPLITS_DIR,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    seed: int = SEED,
) -> dict[str, object]:
    """Split corpus into train/val/test by sentence.

    Groups tokens by sentence_id, then splits sentences 80/10/10.
    Stratifies to preserve gold/silver tier proportions.

    Returns:
        Statistics per split.
    """
    rng = random.Random(seed)

    # Group by sentence_id
    by_sentence: dict[str, list[dict[str, object]]] = defaultdict(list)
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            by_sentence[str(rec["sentence_id"])].append(rec)

    # Classify sentences: has_gold or silver_only
    gold_sents: list[str] = []
    silver_sents: list[str] = []
    for sid, tokens in by_sentence.items():
        if any(t.get("tier") == "gold" for t in tokens):
            gold_sents.append(sid)
        else:
            silver_sents.append(sid)

    rng.shuffle(gold_sents)
    rng.shuffle(silver_sents)

    def _split_list(items: list[str]) -> tuple[list[str], list[str], list[str]]:
        n = len(items)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]

    g_train, g_val, g_test = _split_list(gold_sents)
    s_train, s_val, s_test = _split_list(silver_sents)

    splits = {
        "train": g_train + s_train,
        "val": g_val + s_val,
        "test": g_test + s_test,
    }

    # Write splits
    output_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, object] = {}

    for split_name, sent_ids in splits.items():
        records: list[dict[str, object]] = []
        for sid in sent_ids:
            records.extend(by_sentence[sid])

        path = output_dir / f"{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        tier_counts = Counter(str(r.get("tier", "?")) for r in records)
        stats[split_name] = {
            "sentences": len(sent_ids),
            "tokens": len(records),
            "tiers": dict(tier_counts),
        }

    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    stats = split_corpus()

    print(f"\n{'='*65}")
    print("CORPUS SPLITS")
    print(f"{'='*65}")
    total_tok = 0
    for name in ("train", "val", "test"):
        s = stats[name]
        total_tok += s["tokens"]  # type: ignore[operator]
        tiers = s["tiers"]  # type: ignore[index]
        print(f"  {name:5s}  sents={s['sentences']:>6}  tokens={s['tokens']:>6}  "
              f"gold={tiers.get('gold', 0):>5}  "
              f"s-auto={tiers.get('silver-auto', 0):>5}  "
              f"s-agreed={tiers.get('silver-agreed', 0):>5}")
    print(f"  {'total':5s}  {'':>14}  tokens={total_tok:>6}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
