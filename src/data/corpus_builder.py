"""Three-tier corpus construction for morphological atomizer training.

Tier 1 (Gold):          Human/heuristic-annotated tokens (seed + AL batches)
Tier 2 (Silver-auto):   Unambiguous tokens (parse_count==1) with confidence >= 0.95
Tier 3 (Silver-agreed): Ambiguous tokens with confidence >= 0.98

Usage:
    PYTHONPATH=src python -c "
    from data.corpus_builder import CorpusBuilder
    cb = CorpusBuilder(); stats = cb.build()
    for k, v in stats.items(): print(f'{k}: {v}')
    "
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# POS-dependent confidence thresholds: (auto_threshold, agreed_threshold)
# Verbs: high ambiguity, derivational chains — require high confidence
# Nouns/Adj/Adv: moderate ambiguity — slightly relaxed
# Function words (Postp/Conj/Det): rarely ambiguous — relaxed
# Punctuation: always accept
POS_THRESHOLDS: dict[str, tuple[float, float]] = {
    "Verb": (0.97, 0.99),
    "Noun": (0.93, 0.97),
    "Adj": (0.93, 0.97),
    "Adv": (0.93, 0.97),
    "Pron": (0.90, 0.95),
    "Num": (0.90, 0.95),
    "Postp": (0.88, 0.93),
    "Conj": (0.88, 0.93),
    "Det": (0.88, 0.93),
    "Prop": (0.90, 0.95),
    "Interj": (0.88, 0.93),
    "Punc": (0.50, 0.50),
    "default": (0.95, 0.98),
}


def _extract_pos(label: str) -> str:
    """Extract POS category from a morphological label string.

    Labels have the form "root +POS +TAG1 +TAG2 ...".
    Returns the POS name (e.g., "Noun", "Verb") or "default".
    """
    for part in label.split():
        if part.startswith("+"):
            return part[1:]  # strip leading +
    return "default"


class CorpusBuilder:
    """Build a multi-tier corpus combining gold and silver annotations."""

    def __init__(
        self,
        gold_path: str = "data/gold/combined_gold.jsonl",
        weak_labels_path: str = "data/weak_labels/probabilistic_labels.jsonl",
        prelabeled_path: str = "data/prelabeled/candidates.jsonl",
        output_path: str = "data/gold/tr_gold_morph_v1.jsonl",
        auto_threshold: float = 0.95,
        agreed_threshold: float = 0.98,
        use_pos_thresholds: bool = True,
        pos_thresholds: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.gold_path = Path(gold_path)
        self.weak_labels_path = Path(weak_labels_path)
        self.prelabeled_path = Path(prelabeled_path)
        self.output_path = Path(output_path)
        self.auto_threshold = auto_threshold
        self.agreed_threshold = agreed_threshold
        self.use_pos_thresholds = use_pos_thresholds
        self.pos_thresholds = pos_thresholds or POS_THRESHOLDS

    def build(self) -> dict[str, object]:
        """Build the tiered corpus.

        Returns:
            Statistics dict with tier counts and total.
        """
        # 1. Load gold tokens — index by (sentence_id, token_idx)
        gold_keys: set[str] = set()
        gold_records: list[dict[str, object]] = []

        with open(self.gold_path, encoding="utf-8") as f:
            for line in f:
                sent = json.loads(line)
                sid = str(sent["sentence_id"])
                for token in sent["tokens"]:
                    gl = token.get("gold_label")
                    if gl is not None:
                        key = f"{sid}_{token['token_idx']}"
                        gold_keys.add(key)
                        gold_records.append({
                            "sentence_id": sid,
                            "token_idx": token["token_idx"],
                            "surface": token["surface"],
                            "label": gl,
                            "tier": "gold",
                        })

        # 2. Load prelabeled candidates for parse_count
        parse_counts: dict[str, int] = {}
        with open(self.prelabeled_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                key = f"{rec['sentence_id']}_{rec['token_idx']}"
                parse_counts[key] = int(rec.get("parse_count", 0))

        # 3. Load weak labels, assign silver tiers
        silver_auto: list[dict[str, object]] = []
        silver_agreed: list[dict[str, object]] = []

        with open(self.weak_labels_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                key = f"{rec['sentence_id']}_{rec['token_idx']}"

                # Skip if already in gold
                if key in gold_keys:
                    continue

                label = rec.get("predicted_label", "")
                if not label:
                    continue

                confidence = float(rec.get("confidence", 0.0))
                pc = parse_counts.get(key, 0)

                entry = {
                    "sentence_id": str(rec["sentence_id"]),
                    "token_idx": rec["token_idx"],
                    "surface": rec["surface"],
                    "label": label,
                }

                # Determine thresholds (POS-specific or global)
                if self.use_pos_thresholds:
                    pos = _extract_pos(label)
                    fallback = (self.auto_threshold, self.agreed_threshold)
                    default = self.pos_thresholds.get("default", fallback)
                    auto_thr, agreed_thr = self.pos_thresholds.get(
                        pos, default,
                    )
                else:
                    auto_thr = self.auto_threshold
                    agreed_thr = self.agreed_threshold

                # Tier 2: unambiguous + high confidence
                if pc == 1 and confidence >= auto_thr:
                    entry["tier"] = "silver-auto"
                    silver_auto.append(entry)
                # Tier 3: ambiguous but very high confidence
                elif pc > 1 and confidence >= agreed_thr:
                    entry["tier"] = "silver-agreed"
                    silver_agreed.append(entry)

        # 4. Combine all tiers
        all_records = gold_records + silver_auto + silver_agreed

        # 5. Save
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            for rec in all_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        stats: dict[str, object] = {
            "gold": len(gold_records),
            "silver_auto": len(silver_auto),
            "silver_agreed": len(silver_agreed),
            "total": len(all_records),
            "output": str(self.output_path),
        }
        logger.info("Corpus built: %s", stats)
        return stats
