"""Train Word2Vec embeddings on Turkish corpus.

Training data: data/splits/ JSONL + (optionally) CoNLL-U treebanks + resource JSONL.
Output: Gensim Word2Vec model saved in KeyedVectors binary format.

Usage (local):
    PYTHONPATH=src python src/resource/train_word2vec.py \
        --jsonl_dirs data/splits data/resource \
        --output models/word2vec/tr_word2vec_128.bin

Usage (TRUBA — see scripts/truba/submit_w2v.sh).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sentence iterators
# ---------------------------------------------------------------------------

class _JsonlSentences:
    """Yield token lists from JSONL files in a directory.

    Handles two JSONL formats:
    - Corpus splits: each line has ``surface`` + ``sentence_id`` + ``token_idx``
      — sentences are reconstructed by grouping on ``sentence_id``.
    - Resource exports: each line has only ``surface`` — treated as single-token
      sentences (no sentence structure).
    """

    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths

    def __iter__(self):  # type: ignore[return]
        for path in self.paths:
            sentences: dict[str, list[tuple[int, str]]] = {}
            singletons: list[str] = []

            with open(path, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    surface = rec.get("surface", "").lower().strip()
                    if not surface:
                        continue
                    sent_id = rec.get("sentence_id")
                    if sent_id is not None:
                        tok_idx = rec.get("token_idx", 0)
                        sentences.setdefault(sent_id, []).append((tok_idx, surface))
                    else:
                        singletons.append(surface)

            # Yield reconstructed sentences
            for sent_id, toks in sentences.items():
                toks.sort(key=lambda x: x[0])
                yield [t for _, t in toks]

            # Yield singletons as 1-token sentences
            for tok in singletons:
                yield [tok]


class _ConlluSentences:
    """Yield token lists from CoNLL-U files in a directory.

    Only yields FORM (column 1) for regular tokens (skips multi-word and empty nodes).
    """

    def __init__(self, dirs: list[Path]) -> None:
        self.dirs = dirs

    def __iter__(self):  # type: ignore[return]
        for d in self.dirs:
            for conllu_file in sorted(d.rglob("*.conllu")):
                sentence: list[str] = []
                with open(conllu_file, encoding="utf-8") as f:
                    for line in f:
                        line = line.rstrip()
                        if not line:
                            if sentence:
                                yield sentence
                                sentence = []
                        elif line.startswith("#"):
                            continue
                        else:
                            cols = line.split("\t")
                            if len(cols) >= 2 and "-" not in cols[0] and "." not in cols[0]:
                                form = cols[1].lower()
                                if form:
                                    sentence.append(form)
                if sentence:
                    yield sentence


class _ChainedSentences:
    """Chain multiple sentence iterators."""

    def __init__(self, iterables) -> None:
        self.iterables = iterables

    def __iter__(self):  # type: ignore[return]
        for it in self.iterables:
            yield from it


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_training_sentences(
    jsonl_dirs: list[Path],
    conllu_dirs: list[Path] | None = None,
) -> _ChainedSentences:
    """Build a lazy sentence iterator from JSONL and optional CoNLL-U directories.

    Args:
        jsonl_dirs: Directories containing ``*.jsonl`` files.
        conllu_dirs: Optional directories containing ``*.conllu`` files.

    Returns:
        Iterable of word lists (one list per sentence).
    """
    jsonl_files: list[Path] = []
    for d in jsonl_dirs:
        jsonl_files.extend(sorted(Path(d).glob("*.jsonl")))

    iterables = [_JsonlSentences(jsonl_files)]
    if conllu_dirs:
        iterables.append(_ConlluSentences([Path(d) for d in conllu_dirs]))

    return _ChainedSentences(iterables)


def train_word2vec(
    jsonl_dirs: list[Path],
    output_path: Path,
    conllu_dirs: list[Path] | None = None,
    dim: int = 128,
    window: int = 5,
    min_count: int = 5,
    workers: int = 8,
    epochs: int = 10,
) -> object:
    """Train a CBOW Word2Vec model and save to ``output_path``.

    Uses gensim's Word2Vec implementation.  The model is saved in
    KeyedVectors binary format (``wv.save_word2vec_format(..., binary=True)``).

    Args:
        jsonl_dirs: Directories with JSONL training data.
        output_path: Path to write the output model.
        conllu_dirs: Optional CoNLL-U directories to include.
        dim: Embedding dimension.
        window: Context window size.
        min_count: Minimum token frequency to include in vocab.
        workers: Number of parallel worker threads.
        epochs: Training epochs.

    Returns:
        Trained ``gensim.models.Word2Vec`` object.
    """
    try:
        from gensim.models import Word2Vec  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "gensim is required for Word2Vec training. "
            "Install with: pip install gensim>=4.3.0"
        ) from exc

    sentences = build_training_sentences(jsonl_dirs, conllu_dirs)

    print(f"Training Word2Vec: dim={dim}, window={window}, min_count={min_count}, "
          f"workers={workers}, epochs={epochs}")

    model = Word2Vec(
        sentences=sentences,
        vector_size=dim,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        sg=0,  # CBOW
        seed=42,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.wv.save_word2vec_format(str(output_path), binary=True)
    print(f"Saved Word2Vec to {output_path}  (vocab size: {len(model.wv):,})")
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Train Turkish Word2Vec embeddings")
    parser.add_argument(
        "--jsonl_dirs", nargs="+", required=True,
        help="Directories containing JSONL training data",
    )
    parser.add_argument(
        "--conllu_dirs", nargs="*", default=None,
        help="Optional directories with CoNLL-U files",
    )
    parser.add_argument(
        "--output", default="models/word2vec/tr_word2vec_128.bin",
        help="Output path for Word2Vec binary",
    )
    parser.add_argument("--dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--window", type=int, default=5, help="Context window size")
    parser.add_argument("--min_count", type=int, default=5, help="Minimum word frequency")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    args = parser.parse_args()

    train_word2vec(
        jsonl_dirs=[Path(d) for d in args.jsonl_dirs],
        output_path=Path(args.output),
        conllu_dirs=[Path(d) for d in args.conllu_dirs] if args.conllu_dirs else None,
        dim=args.dim,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        epochs=args.epochs,
    )
