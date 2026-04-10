"""Corpus ingestion — parse CoNLL-U files from BOUN Treebank.

Extracts sentences and tokens into a standardized JSONL format for
downstream morphological analysis and weak supervision.

Usage:
    python src/data/ingest.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

BOUN_DIR = Path("data/external/boun_treebank")
OUTPUT_PATH = Path("data/processed/corpus.jsonl")
SPLITS = ["train", "dev", "test"]


def parse_conllu(path: Path) -> list[dict[str, object]]:
    """Parse a CoNLL-U file into a list of sentence dicts.

    Each sentence dict has:
        sentence_id: str
        text: str
        tokens: list[str]
        pos_tags: list[str]
        lemmas: list[str]

    Multiword tokens (lines with '-' in ID) are expanded into their
    components. Only lines with integer IDs are kept as tokens.
    """
    sentences: list[dict[str, object]] = []
    current_id = ""
    current_text = ""
    current_tokens: list[str] = []
    current_pos: list[str] = []
    current_lemmas: list[str] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith("# sent_id"):
                current_id = line.split("=", 1)[1].strip()
            elif line.startswith("# text"):
                current_text = line.split("=", 1)[1].strip()
            elif line == "":
                # End of sentence
                if current_tokens:
                    sentences.append({
                        "sentence_id": current_id,
                        "text": current_text,
                        "tokens": current_tokens,
                        "pos_tags": current_pos,
                        "lemmas": current_lemmas,
                    })
                current_tokens = []
                current_pos = []
                current_lemmas = []
            elif not line.startswith("#"):
                fields = line.split("\t")
                if len(fields) < 4:
                    continue
                token_id = fields[0]
                # Skip multiword token range lines (e.g., "2-3")
                if "-" in token_id or "." in token_id:
                    continue
                current_tokens.append(fields[1])  # FORM
                current_lemmas.append(fields[2])  # LEMMA
                current_pos.append(fields[3])  # UPOS

    # Handle last sentence if file doesn't end with blank line
    if current_tokens:
        sentences.append({
            "sentence_id": current_id,
            "text": current_text,
            "tokens": current_tokens,
            "pos_tags": current_pos,
            "lemmas": current_lemmas,
        })

    return sentences


def ingest_boun_treebank(
    boun_dir: Path = BOUN_DIR,
    output_path: Path = OUTPUT_PATH,
    min_sentence_length: int = 3,
    max_sentence_length: int = 200,
) -> list[dict[str, object]]:
    """Ingest all BOUN Treebank splits into a single corpus JSONL file.

    Args:
        boun_dir: Path to the BOUN Treebank directory.
        output_path: Path for the output JSONL file.
        min_sentence_length: Minimum number of tokens per sentence.
        max_sentence_length: Maximum number of tokens per sentence.

    Returns:
        List of all ingested sentence dicts.
    """
    all_sentences: list[dict[str, object]] = []

    for split in SPLITS:
        path = boun_dir / f"tr_boun-ud-{split}.conllu"
        if not path.exists():
            logger.warning("Missing split file: %s", path)
            continue
        sentences = parse_conllu(path)
        logger.info("Parsed %d sentences from %s", len(sentences), split)
        all_sentences.extend(sentences)

    # Filter by length
    filtered = [
        s for s in all_sentences
        if min_sentence_length <= len(s["tokens"]) <= max_sentence_length  # type: ignore[arg-type]
    ]
    logger.info(
        "Filtered: %d → %d sentences (min=%d, max=%d tokens)",
        len(all_sentences), len(filtered), min_sentence_length, max_sentence_length,
    )

    # Renumber sentence IDs to ensure uniqueness
    for i, sent in enumerate(filtered):
        sent["sentence_id"] = f"boun_{i:05d}"

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sent in filtered:
            f.write(json.dumps(sent, ensure_ascii=False) + "\n")

    total_tokens = sum(len(s["tokens"]) for s in filtered)  # type: ignore[arg-type]
    logger.info("Wrote %d sentences (%d tokens) to %s", len(filtered), total_tokens, output_path)

    return filtered


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sentences = ingest_boun_treebank()
    total_tokens = sum(len(s["tokens"]) for s in sentences)  # type: ignore[arg-type]
    print(f"Ingested {len(sentences)} sentences, {total_tokens} tokens")


if __name__ == "__main__":
    main()
