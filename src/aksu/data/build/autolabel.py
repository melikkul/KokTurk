"""Auto-labeling pipeline for TR-Gold-Morph unique tokens.

For each unique token from data/intermediate/unique_tokens.jsonl:
  1. Run Zeyrek → candidates list
  2. If 0 candidates → mark for DualHead generation (placeholder)
  3. If 1 candidate → accept, confidence=1.0, unanimous=True
  4. If >1 candidates → run v6 disambiguator ensemble (5 seeds)

Outputs data/intermediate/autolabeled.jsonl.

Usage (from project root):
    python -m aksu.data.build.autolabel --shard-start 0 --shard-end 50000
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path

import zeyrek

logger = logging.getLogger(__name__)

ENSEMBLE_SEEDS = [42, 123, 456, 789, 1337]


def _load_sentence_index(token_sents_path: Path) -> dict[str, list[str]]:
    """Load token → [sentence_id, ...] index."""
    index: dict[str, list[str]] = {}
    if not token_sents_path.exists():
        return index
    with token_sents_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            index[row["token"]] = row["sentence_ids"]
    return index


def _load_sentence_texts(sentences_path: Path) -> dict[str, str]:
    """Load sentence_id → sentence text."""
    texts: dict[str, str] = {}
    if not sentences_path.exists():
        return texts
    with sentences_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            texts[row["sentence_id"]] = row["text"]
    return texts


def _format_canonical(result) -> str:
    """Format a Zeyrek MorphAnalysisResult as our canonical string."""
    try:
        # Zeyrek returns list of MorphAnalysis; use first
        analyses = result if isinstance(result, list) else [result]
        if not analyses:
            return ""
        ana = analyses[0]
        # Access lemma and morpheme list
        lemma = getattr(ana, "lemma", None) or str(ana)
        morphemes = getattr(ana, "morphemes", [])
        if morphemes:
            tags = " ".join(f"+{m}" for m in morphemes if m)
            return f"{lemma} {tags}".strip()
        return lemma
    except Exception:
        return str(result)


def autolabel_tokens(
    tokens: list[dict],
    *,
    sentence_index: dict[str, list[str]],
    sentence_texts: dict[str, str],
    output_path: Path,
    rng_seed: int = 42,
) -> dict[str, int]:
    """Autolabel a list of unique token records and append to output_path."""
    rng = random.Random(rng_seed)
    analyzer = zeyrek.MorphAnalyzer()

    stats: Counter = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a", encoding="utf-8") as out:
        for row in tokens:
            token = row["token"]
            source_id = row.get("source", "unknown")

            try:
                results = analyzer.analyze(token)
            except Exception as e:
                logger.warning("Zeyrek error for %r: %s", token, e)
                results = []

            if not results:
                # No candidates — mark for DualHead generation
                record = {
                    "token": token,
                    "canonical": None,
                    "candidates": [],
                    "confidence": 0.0,
                    "ensemble_votes": 0,
                    "unanimous": False,
                    "source_id": source_id,
                    "method": "dualhead_pending",
                }
                stats["dualhead"] += 1
            elif len(results) == 1:
                record = {
                    "token": token,
                    "canonical": _format_canonical(results[0]),
                    "candidates": [_format_canonical(results[0])],
                    "confidence": 1.0,
                    "ensemble_votes": len(ENSEMBLE_SEEDS),
                    "unanimous": True,
                    "source_id": source_id,
                    "method": "unambiguous",
                }
                stats["unambiguous"] += 1
            else:
                # Multiple candidates — pick a sentence context for BERTurk
                sids = sentence_index.get(token, [])
                context = ""
                if sids:
                    chosen_sid = rng.choice(sids)
                    context = sentence_texts.get(chosen_sid, "")

                # Simplified scoring: take first candidate (placeholder until
                # BERTurk disambiguator models are loaded via SLURM)
                candidates_str = [_format_canonical(r) for r in results]
                record = {
                    "token": token,
                    "canonical": candidates_str[0] if candidates_str else None,
                    "candidates": candidates_str,
                    "confidence": 0.80,  # placeholder; real value from ensemble
                    "ensemble_votes": 0,  # placeholder; set by ensemble scorer
                    "unanimous": False,  # placeholder
                    "source_id": source_id,
                    "context_sentence": context[:200] if context else None,
                    "method": "zeyrek_first",  # upgraded to "ensemble" by SLURM scorer
                }
                stats["ambiguous"] += 1

            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    return dict(stats)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--unique-tokens", default="data/intermediate/unique_tokens.jsonl")
    ap.add_argument("--token-sentences", default="data/intermediate/token_sentences.jsonl")
    ap.add_argument("--sentences", default="data/intermediate/sentences.jsonl")
    ap.add_argument("--output", default="data/intermediate/autolabeled.jsonl")
    ap.add_argument("--shard-start", type=int, default=0)
    ap.add_argument("--shard-end", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tokens_path = Path(args.unique_tokens)
    if not tokens_path.exists():
        ap.error(f"unique_tokens file not found: {tokens_path}")

    with tokens_path.open(encoding="utf-8") as f:
        all_tokens = [json.loads(l) for l in f if l.strip()]

    shard = all_tokens[args.shard_start : args.shard_end]
    logger.info("Autolabeling %d tokens (shard %d:%s)", len(shard), args.shard_start, args.shard_end)

    logger.info("Loading sentence index...")
    sentence_index = _load_sentence_index(Path(args.token_sentences))
    sentence_texts = _load_sentence_texts(Path(args.sentences))
    logger.info("Index: %d tokens, %d sentences", len(sentence_index), len(sentence_texts))

    stats = autolabel_tokens(
        shard,
        sentence_index=sentence_index,
        sentence_texts=sentence_texts,
        output_path=Path(args.output),
        rng_seed=args.seed,
    )
    logger.info("Done: %s", stats)


if __name__ == "__main__":
    main()
