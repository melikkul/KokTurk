"""Create stratified seed annotation dataset (200 sentences).

Selects sentences for initial manual annotation using stratified sampling
to ensure diversity in morphological complexity.

Usage:
    python src/data/create_seed.py
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

CORPUS_PATH = Path("data/processed/corpus.jsonl")
CANDIDATES_PATH = Path("data/prelabeled/candidates.jsonl")
LABELS_PATH = Path("data/weak_labels/probabilistic_labels.jsonl")
OUTPUT_PATH = Path("data/gold/seed/seed_200.jsonl")

SEED = 42
N_SENTENCES = 200
N_HIGH_AMBIGUITY = 30
N_LOW_AMBIGUITY = 30
N_HARD_CASES = 20


def load_jsonl(path: Path) -> list[dict[str, object]]:
    """Load a JSONL file into a list of dicts."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def compute_sentence_ambiguity(
    sentence: dict[str, object],
    candidates_by_sent: dict[str, list[dict[str, object]]],
) -> dict[str, float]:
    """Compute ambiguity statistics for a sentence."""
    sent_id = str(sentence["sentence_id"])
    token_candidates = candidates_by_sent.get(sent_id, [])
    tokens: list[str] = sentence["tokens"]  # type: ignore[assignment]
    n_tokens = len(tokens)

    if n_tokens == 0:
        return {"ambiguity_frac": 0.0, "max_parse_count": 0, "avg_parses": 0.0}

    parse_counts = [c.get("parse_count", 0) for c in token_candidates]
    ambiguous = sum(1 for pc in parse_counts if pc > 1)

    return {
        "ambiguity_frac": ambiguous / n_tokens,
        "max_parse_count": max(parse_counts) if parse_counts else 0,
        "avg_parses": sum(parse_counts) / len(parse_counts) if parse_counts else 0.0,
    }


def select_seed_sentences(
    corpus: list[dict[str, object]],
    candidates_by_sent: dict[str, list[dict[str, object]]],
    n_total: int = N_SENTENCES,
    seed: int = SEED,
) -> list[dict[str, object]]:
    """Select sentences using stratified sampling.

    Strata:
    - High ambiguity: >40% ambiguous tokens (at least N_HIGH_AMBIGUITY)
    - Low ambiguity: <20% ambiguous tokens (at least N_LOW_AMBIGUITY)
    - Hard cases: contains tokens with parse_count >= 5 (at least N_HARD_CASES)
    - Remaining: random

    Returns:
        List of selected sentence dicts with ambiguity metadata.
    """
    rng = random.Random(seed)

    # Compute ambiguity for all sentences
    scored: list[tuple[dict[str, object], dict[str, float]]] = []
    for sent in corpus:
        stats = compute_sentence_ambiguity(sent, candidates_by_sent)
        scored.append((sent, stats))

    # Categorize
    high_amb = [(s, st) for s, st in scored if st["ambiguity_frac"] > 0.4]
    low_amb = [(s, st) for s, st in scored if st["ambiguity_frac"] < 0.2]
    hard = [(s, st) for s, st in scored if st["max_parse_count"] >= 5]

    rng.shuffle(high_amb)
    rng.shuffle(low_amb)
    rng.shuffle(hard)

    selected_ids: set[str] = set()
    selected: list[tuple[dict[str, object], dict[str, float]]] = []

    # Pick from each stratum
    for pool, quota in [
        (high_amb, N_HIGH_AMBIGUITY),
        (low_amb, N_LOW_AMBIGUITY),
        (hard, N_HARD_CASES),
    ]:
        count = 0
        for s, st in pool:
            sid = str(s["sentence_id"])
            if sid not in selected_ids and count < quota:
                selected.append((s, st))
                selected_ids.add(sid)
                count += 1

    # Fill remaining with random
    remaining = [(s, st) for s, st in scored if str(s["sentence_id"]) not in selected_ids]
    rng.shuffle(remaining)
    for s, st in remaining:
        if len(selected) >= n_total:
            break
        selected.append((s, st))
        selected_ids.add(str(s["sentence_id"]))

    return [s for s, _ in selected[:n_total]]


def build_seed_dataset(
    selected_sentences: list[dict[str, object]],
    candidates_by_sent: dict[str, list[dict[str, object]]],
    labels_by_key: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    """Build the seed dataset with candidate parses ranked by confidence."""
    seed_data: list[dict[str, object]] = []

    for sent in selected_sentences:
        sent_id = str(sent["sentence_id"])
        tokens_data: list[dict[str, object]] = []
        tokens: list[str] = sent["tokens"]  # type: ignore[assignment]

        for idx, surface in enumerate(tokens):
            key = f"{sent_id}_{idx}"
            label_record = labels_by_key.get(key, {})

            # Get candidates for this token
            cands = [
                c for c in candidates_by_sent.get(sent_id, [])
                if c.get("token_idx") == idx
            ]
            analyses: list[dict[str, object]] = []
            if cands:
                analyses = cands[0].get("analyses", [])  # type: ignore[assignment]

            # Rank by confidence from label model
            confidence = float(label_record.get("confidence", 0.0))
            predicted_idx = int(label_record.get("predicted_idx", 0))

            ranked_parses: list[dict[str, object]] = []
            for i, a in enumerate(analyses):
                parse_conf = confidence if i == predicted_idx else (
                    (1.0 - confidence) / max(len(analyses) - 1, 1)
                )
                ranked_parses.append({
                    "root": a.get("root", surface),
                    "tags": a.get("tags", []),
                    "source": a.get("source", "unknown"),
                    "confidence": round(parse_conf, 4),
                })

            # Sort by confidence descending
            ranked_parses.sort(key=lambda p: p["confidence"], reverse=True)

            tokens_data.append({
                "surface": surface,
                "token_idx": idx,
                "candidate_parses": ranked_parses,
                "gold_label": None,
            })

        seed_data.append({
            "sentence_id": sent_id,
            "text": sent.get("text", ""),
            "tokens": tokens_data,
        })

    return seed_data


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s",
    )

    logger.info("Loading data...")
    corpus = load_jsonl(CORPUS_PATH)
    candidates = load_jsonl(CANDIDATES_PATH)
    labels = load_jsonl(LABELS_PATH)

    # Index candidates by sentence_id
    candidates_by_sent: dict[str, list[dict[str, object]]] = {}
    for c in candidates:
        sid = str(c["sentence_id"])
        candidates_by_sent.setdefault(sid, []).append(c)

    # Index labels by (sentence_id, token_idx)
    labels_by_key: dict[str, dict[str, object]] = {}
    for lb in labels:
        key = f"{lb['sentence_id']}_{lb['token_idx']}"
        labels_by_key[key] = lb

    logger.info("Selecting %d seed sentences...", N_SENTENCES)
    selected = select_seed_sentences(corpus, candidates_by_sent)

    logger.info("Building seed dataset...")
    seed_data = build_seed_dataset(selected, candidates_by_sent, labels_by_key)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for record in seed_data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Print stats
    total_tokens = sum(len(s["tokens"]) for s in seed_data)  # type: ignore[arg-type]
    ambiguous_tokens = sum(
        1 for s in seed_data
        for t in s["tokens"]  # type: ignore[union-attr]
        if len(t["candidate_parses"]) > 1  # type: ignore[index]
    )
    avg_parses = sum(
        len(t["candidate_parses"])  # type: ignore[index]
        for s in seed_data
        for t in s["tokens"]  # type: ignore[union-attr]
    ) / max(total_tokens, 1)

    # POS distribution from corpus
    pos_counter: Counter[str] = Counter()
    corpus_by_id = {str(s["sentence_id"]): s for s in corpus}
    for s in seed_data:
        csent = corpus_by_id.get(str(s["sentence_id"]), {})
        for pos in csent.get("pos_tags", []):  # type: ignore[union-attr]
            pos_counter[pos] += 1

    print(f"\n{'='*60}")
    print("SEED DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Sentences:             {len(seed_data)}")
    print(f"Total tokens:          {total_tokens}")
    print(f"Ambiguous tokens:      {ambiguous_tokens} ({ambiguous_tokens/max(total_tokens,1):.1%})")
    print(f"Avg parses per token:  {avg_parses:.2f}")
    print("\nPOS distribution (top 10):")
    for pos, count in pos_counter.most_common(10):
        print(f"  {pos:10s} {count:6d}")
    print(f"{'='*60}\n")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
