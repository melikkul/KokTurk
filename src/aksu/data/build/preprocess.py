"""Preprocessing pipeline for TR-Gold-Morph corpus harvest.

Produces three artifacts in data/intermediate/:
  tokens.jsonl       — one row per token occurrence
  sentences.jsonl    — one row per unique sentence (de-duplicated)
  token_sentences.jsonl — one row per unique token with up to 32 sentence refs

Usage:
    python -m aksu.data.build.preprocess --shard oscar-tr --max-tokens 3000000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

_MAX_SENTENCE_REFS = 32  # cap per token to avoid huge lists for high-freq types


def _sentence_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _tokenize_simple(text: str) -> list[str]:
    """Whitespace tokenizer; replaced by Stanza-tr when available."""
    return re.findall(r"\S+", text)


def preprocess_shard(
    texts: list[str],
    source_name: str,
    source_license: str,
    *,
    output_dir: Path,
    max_tokens: int | None = None,
) -> dict[str, int]:
    """Process a list of sentence strings into the three intermediate files.

    Returns stats dict with token_count, unique_sentence_count, unique_token_count.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    tokens_path = output_dir / "tokens.jsonl"
    sentences_path = output_dir / "sentences.jsonl"
    token_sents_path = output_dir / "token_sentences.jsonl"

    seen_sentences: set[str] = set()
    token_to_sids: dict[str, list[str]] = defaultdict(list)

    token_count = 0
    unique_sentence_count = 0

    try:
        import stanza
        pipeline = stanza.Pipeline("tr", processors="tokenize", verbose=False)
        use_stanza = True
    except (ImportError, Exception):
        use_stanza = False
        logger.warning("Stanza unavailable; using whitespace tokenizer")

    with tokens_path.open("a", encoding="utf-8") as tf, \
         sentences_path.open("a", encoding="utf-8") as sf:

        for text in texts:
            if not text or not text.strip():
                continue

            # Sentence segmentation
            if use_stanza:
                doc = pipeline(text)
                sents = [s.text for s in doc.sentences]
            else:
                sents = [text.strip()]

            for sent_text in sents:
                sid = _sentence_id(sent_text)

                if sid not in seen_sentences:
                    seen_sentences.add(sid)
                    sf.write(json.dumps({
                        "sentence_id": sid,
                        "text": sent_text,
                        "source": source_name,
                        "source_lic": source_license,
                    }, ensure_ascii=False) + "\n")
                    unique_sentence_count += 1

                tokens = _tokenize_simple(sent_text)
                for pos, token in enumerate(tokens):
                    if not token:
                        continue
                    tf.write(json.dumps({
                        "token": token,
                        "sentence_id": sid,
                        "source": source_name,
                        "source_lic": source_license,
                        "position": pos,
                    }, ensure_ascii=False) + "\n")
                    token_count += 1

                    refs = token_to_sids[token]
                    if len(refs) < _MAX_SENTENCE_REFS and sid not in refs:
                        refs.append(sid)

                if max_tokens and token_count >= max_tokens:
                    logger.info("max_tokens=%d reached", max_tokens)
                    break
            if max_tokens and token_count >= max_tokens:
                break

    # Write token→sentence index
    unique_token_count = len(token_to_sids)
    with token_sents_path.open("a", encoding="utf-8") as tsf:
        for token, sids in token_to_sids.items():
            tsf.write(json.dumps({
                "token": token,
                "sentence_ids": sids,
            }, ensure_ascii=False) + "\n")

    return {
        "token_count": token_count,
        "unique_sentence_count": unique_sentence_count,
        "unique_token_count": unique_token_count,
    }


def _load_local_jsonl(path: Path, text_field: str = "text") -> list[str]:
    """Read sentences from a pre-downloaded JSONL file (one JSON object per line)."""
    texts: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                texts.append(line)
                continue
            if isinstance(obj, str):
                texts.append(obj)
            elif isinstance(obj, dict):
                texts.append(obj.get(text_field) or obj.get("sentence") or obj.get("text") or "")
    return [t for t in texts if t.strip()]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True, help="Source name (e.g. oscar-tr)")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--output-dir", default="data/intermediate")
    ap.add_argument("--dry-run", action="store_true", help="Print stats only")
    ap.add_argument(
        "--local-jsonl",
        help="Path to a pre-downloaded JSONL file (one sentence/text per line). "
             "Use this on HPC nodes without internet access. "
             "If not given, streams from HuggingFace (requires internet).",
    )
    args = ap.parse_args()

    from aksu.data.build.sources import SOURCES
    source = next((s for s in SOURCES if s.name == args.shard), None)
    if source is None:
        ap.error(f"Unknown shard {args.shard!r}. Available: {[s.name for s in SOURCES]}")

    if args.local_jsonl:
        local_path = Path(args.local_jsonl)
        if not local_path.exists():
            ap.error(f"--local-jsonl path does not exist: {local_path}")
        logger.info("Loading shard %s from local file %s ...", source.name, local_path)
        texts = _load_local_jsonl(local_path)
    else:
        logger.info("Loading shard %s from HuggingFace %s ...", source.name, source.url)
        try:
            from datasets import load_dataset
            ds = load_dataset(source.url, split="train", streaming=True)
            texts = [
                row.get("text") or row.get("sentence") or ""
                for row in ds
                if row.get("text") or row.get("sentence")
            ]
        except Exception as e:
            logger.error("Could not load shard: %s", e)
            raise

    if args.dry_run:
        logger.info("DRY RUN: first 10 texts from shard:")
        for t in texts[:10]:
            logger.info("  %r", t[:80])
        return

    stats = preprocess_shard(
        texts,
        source_name=source.name,
        source_license=source.license,
        output_dir=Path(args.output_dir),
        max_tokens=args.max_tokens,
    )
    logger.info("Done: %s", stats)


if __name__ == "__main__":
    main()
