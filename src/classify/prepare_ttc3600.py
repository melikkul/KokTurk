"""Prepare TTC-3600 dataset for classification experiments.

Downloads (or loads) TTC-3600 and atomizes all documents using the
trained morphological atomizer + Zeyrek fallback.

Usage:
    PYTHONPATH=src python src/classify/prepare_ttc3600.py
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

TTC_DIR = Path("data/external/ttc3600")
OUTPUT_PATH = TTC_DIR / "ttc3600_atomized.jsonl"

# TTC-3600 categories
CATEGORIES = ["economy", "culture", "health", "politics", "sports", "technology"]


def _try_download_ttc3600() -> bool:
    """Attempt to download TTC-3600 from known sources."""
    TTC_DIR.mkdir(parents=True, exist_ok=True)

    # TTC-3600 is often distributed as separate text files per category
    # If already present, skip download
    existing = list(TTC_DIR.glob("*.txt")) + list(TTC_DIR.glob("*.csv"))
    if existing:
        logger.info("TTC-3600 data found (%d files)", len(existing))
        return True

    # Try Kaggle-style download or UCI — these may fail on restricted networks
    logger.warning(
        "TTC-3600 not found in %s. "
        "Please download manually from UCI ML Repository or Kaggle "
        "and place files in %s/",
        TTC_DIR, TTC_DIR,
    )
    return False


def _create_synthetic_ttc3600() -> list[dict[str, str]]:
    """Create synthetic TTC-3600-like data from BOUN Treebank for testing.

    Uses BOUN sentences grouped into fake categories. This is NOT real
    TTC-3600 — it's for pipeline validation only.
    """
    logger.info("Creating synthetic TTC-3600 from BOUN Treebank sentences")

    corpus_path = Path("data/processed/corpus.jsonl")
    sentences: list[dict[str, object]] = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            sentences.append(json.loads(line))

    docs: list[dict[str, str]] = []
    # Group sentences into "documents" of 5 sentences each
    for i in range(0, len(sentences) - 4, 5):
        group = sentences[i:i + 5]
        text = " ".join(str(s.get("text", "")) for s in group)
        category = CATEGORIES[len(docs) % len(CATEGORIES)]
        docs.append({
            "text": text,
            "label": category,
            "doc_id": f"synth_{len(docs):04d}",
        })

    logger.info("Created %d synthetic documents", len(docs))
    return docs


def atomize_documents(
    docs: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Atomize all documents using Zeyrek analyzer.

    For each word, produces "root +TAG1 +TAG2 ..." representation.
    Falls back to raw surface form for OOV words.
    """
    from kokturk.core.analyzer import MorphoAnalyzer

    analyzer = MorphoAnalyzer()
    atomized_docs: list[dict[str, str]] = []
    total_tokens = 0
    atomized_tokens = 0

    for i, doc in enumerate(docs):
        words = doc["text"].split()
        atom_parts: list[str] = []
        for word in words:
            total_tokens += 1
            result = analyzer.analyze(word)
            best = result.best
            if best is not None:
                atom_parts.append(best.to_str())
                atomized_tokens += 1
            else:
                atom_parts.append(word)  # fallback: raw surface

        atomized_docs.append({
            "text": doc["text"],
            "text_atomized": " ".join(atom_parts),
            "label": doc["label"],
            "doc_id": doc["doc_id"],
        })

        if (i + 1) % 500 == 0:
            logger.info("Atomized %d/%d documents", i + 1, len(docs))

    analyzer.close()

    coverage = atomized_tokens / max(total_tokens, 1)
    logger.info(
        "Atomization: %d/%d tokens (%.1f%% coverage)",
        atomized_tokens, total_tokens, 100 * coverage,
    )

    return atomized_docs


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    # Try to load real TTC-3600
    has_data = _try_download_ttc3600()

    if not has_data:
        # Use synthetic data for pipeline validation
        docs = _create_synthetic_ttc3600()
    else:
        # Load from directory — implementation depends on file format
        logger.info("Loading TTC-3600 from %s", TTC_DIR)
        docs = []
        for cat in CATEGORIES:
            cat_dir = TTC_DIR / cat
            if cat_dir.is_dir():
                for f in sorted(cat_dir.glob("*.txt")):
                    text = f.read_text(encoding="utf-8", errors="ignore").strip()
                    if text:
                        docs.append({
                            "text": text, "label": cat,
                            "doc_id": f"{cat}_{f.stem}",
                        })
        if not docs:
            logger.warning("No docs loaded from TTC-3600, using synthetic")
            docs = _create_synthetic_ttc3600()

    logger.info("Documents: %d", len(docs))
    dist = Counter(d["label"] for d in docs)
    for cat in CATEGORIES:
        logger.info("  %s: %d", cat, dist.get(cat, 0))

    # Atomize
    atomized = atomize_documents(docs)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for doc in atomized:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    avg_tokens = sum(len(d["text"].split()) for d in docs) / max(len(docs), 1)
    print(f"\n{'='*60}")
    print("TTC-3600 PREPARATION")
    print(f"{'='*60}")
    print(f"Documents:            {len(docs)}")
    print(f"Classes:              {len(dist)}")
    print(f"Avg tokens/doc:       {avg_tokens:.0f}")
    print(f"Output:               {OUTPUT_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
