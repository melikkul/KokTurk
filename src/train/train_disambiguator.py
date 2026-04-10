"""Train the BERTurk morphological disambiguation model.

Much faster than generation training because:
1. BERTurk is frozen (no gradient through 110M params)
2. Task is classification over ~4 candidates (not generation over 7.8K tokens)
3. 77% of tokens are unambiguous (trivially correct, still useful for training)

Expected: ~2-8 hours on CPU with pre-cached BERTurk embeddings.

Usage:
    PYTHONPATH=src python -m train.train_disambiguator \\
        --train-data data/splits/train.jsonl \\
        --val-data data/splits/val.jsonl \\
        --tag-vocab models/vocabs/tag_vocab.json \\
        --berturk-path models/berturk \\
        --output-dir models/v6/disambiguator
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train.datasets import Vocab
from train.disambiguation_dataset import (
    DisambiguationDataset,
    disambiguation_collate,
)
from train.reproducibility import seed_everything

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BERTurk embedding pre-caching
# ---------------------------------------------------------------------------

def pre_cache_bert_embeddings(
    dataset: DisambiguationDataset,
    bert_path: str,
    cache_path: Path | None = None,
    batch_size: int = 32,
    shared_bert: object | None = None,
    shared_tokenizer: object | None = None,
) -> torch.Tensor:
    """Pre-compute BERTurk per-word embeddings for all samples.

    Stores embeddings as a single dense tensor of shape (N, 768) indexed
    by sample index — much more memory-efficient than dict-of-tensors
    (avoids Python per-object overhead and pickle bloat).

    Since BERTurk is frozen, we only need to run it once. This eliminates
    the BERT forward pass during training (~10x speedup).

    Args:
        dataset: DisambiguationDataset to cache embeddings for.
        bert_path: Path to local BERTurk directory.
        cache_path: Optional path to save/load the cache.
        batch_size: BERTurk batch size.
        shared_bert: Pre-loaded BERT model (avoids reloading on second call).
        shared_tokenizer: Pre-loaded tokenizer.

    Returns:
        Tensor of shape (N, 768) where N = len(dataset).
    """
    if cache_path and cache_path.exists():
        logger.info("Loading BERTurk cache from %s", cache_path)
        cache = torch.load(cache_path, map_location="cpu")
        logger.info("  Loaded cache tensor: %s", tuple(cache.shape))
        return cache

    N = len(dataset)
    logger.info("Pre-caching BERTurk embeddings for %d samples...", N)

    if shared_bert is None or shared_tokenizer is None:
        from transformers import AutoModel, AutoTokenizer  # type: ignore[import]
        shared_tokenizer = AutoTokenizer.from_pretrained(bert_path)
        shared_bert = AutoModel.from_pretrained(bert_path)
        shared_bert.eval()
        for param in shared_bert.parameters():
            param.requires_grad = False

    model = shared_bert
    tokenizer = shared_tokenizer

    # Pre-allocate single dense tensor — avoids dict-of-tensors overhead
    cache = torch.zeros(N, 768, dtype=torch.float32)

    # Group samples by sentence to avoid duplicate BERT forward passes
    sentence_samples: dict[str, list[int]] = defaultdict(list)
    for i, sample in enumerate(dataset.samples):
        sentence_samples[sample["sentence_text"]].append(i)

    sentences = list(sentence_samples.keys())
    n_batches = (len(sentences) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        batch_sents = sentences[
            batch_idx * batch_size : (batch_idx + 1) * batch_size
        ]

        encoding = tokenizer(
            batch_sents,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
            )
        hidden = outputs.last_hidden_state  # (B, S, 768)

        for b, sent_text in enumerate(batch_sents):
            for sample_idx in sentence_samples[sent_text]:
                pos = dataset.samples[sample_idx]["target_position"]
                try:
                    word_ids_b = encoding.word_ids(b)
                    subword_idx = next(
                        (
                            i for i, wid in enumerate(word_ids_b)
                            if wid == int(pos)
                        ),
                        1,
                    )
                except Exception:
                    subword_idx = min(int(pos) + 1, hidden.size(1) - 1)

                cache[sample_idx] = hidden[b, subword_idx].cpu()

        if (batch_idx + 1) % 100 == 0:
            logger.info(
                "  Cached %d/%d sentence batches", batch_idx + 1, n_batches,
            )

    logger.info("  Cached tensor shape: %s (%.1f MB)",
                tuple(cache.shape),
                cache.element_size() * cache.nelement() / 1024 / 1024)

    # Save to disk if path provided
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_path)
        logger.info("  Saved cache to %s", cache_path)

    return cache


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def _get_cached_embeds(
    bert_cache: torch.Tensor,
    sample_indices: torch.Tensor,
) -> torch.Tensor:
    """Look up pre-cached BERTurk embeddings by sample index."""
    return bert_cache[sample_indices]


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    bert_cache: torch.Tensor | None = None,
) -> tuple[float, float]:
    """Train one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    # Keep BERTurk frozen (may be None when skip_bert_loading=True)
    if getattr(model, "bert", None) is not None:
        model.bert.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        optimizer.zero_grad()

        cached_embeds = None
        if bert_cache is not None:
            cached_embeds = _get_cached_embeds(
                bert_cache, batch["sample_indices"],
            )

        logits, loss = model(
            sentence_texts=batch["sentence_texts"],
            target_positions=batch["target_positions"],
            candidate_ids=batch["candidate_ids"],
            candidate_mask=batch["candidate_mask"],
            gold_indices=batch["gold_indices"],
            cached_bert_embeds=cached_embeds,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=5.0,
        )
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == batch["gold_indices"].to(device)).sum().item()
        total += len(batch["gold_indices"])

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    bert_cache: torch.Tensor | None = None,
) -> dict[str, float]:
    """Evaluate the model. Returns dict of metrics."""
    model.eval()

    correct = 0
    total = 0
    ambig_correct = 0
    ambig_total = 0

    for batch in loader:
        cached_embeds = None
        if bert_cache is not None:
            cached_embeds = _get_cached_embeds(
                bert_cache, batch["sample_indices"],
            )

        logits, _ = model(
            sentence_texts=batch["sentence_texts"],
            target_positions=batch["target_positions"],
            candidate_ids=batch["candidate_ids"],
            candidate_mask=batch["candidate_mask"],
            cached_bert_embeds=cached_embeds,
        )

        preds = logits.argmax(dim=-1)
        gold = batch["gold_indices"].to(device)

        correct += (preds == gold).sum().item()
        total += len(gold)

        # Track ambiguous-only accuracy (K > 1)
        ambig_mask = batch["num_candidates"] > 1
        if ambig_mask.any():
            ambig_correct += (
                preds[ambig_mask] == gold[ambig_mask]
            ).sum().item()
            ambig_total += ambig_mask.sum().item()

    return {
        "overall_em": correct / total if total > 0 else 0,
        "ambiguous_em": ambig_correct / ambig_total if ambig_total > 0 else 0,
        "total": total,
        "ambiguous_total": ambig_total,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train BERTurk morphological disambiguator",
    )
    parser.add_argument("--train-data", required=True, type=Path)
    parser.add_argument("--val-data", required=True, type=Path)
    parser.add_argument("--tag-vocab", required=True, type=Path)
    parser.add_argument("--berturk-path", default="models/berturk")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir", default="models/v6/disambiguator")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--pre-cache-bert", action="store_true", default=True,
        help="Pre-compute BERTurk embeddings (default: True)",
    )
    parser.add_argument(
        "--no-pre-cache-bert", dest="pre_cache_bert", action="store_false",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Directory for BERTurk cache (default: $SCRATCH or output-dir)",
    )
    parser.add_argument("--patience", type=int, default=7)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    seed_everything(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load tag vocab
    tag_vocab = Vocab.load(args.tag_vocab)
    logger.info("Tag vocab: %d tokens", len(tag_vocab))

    # Build datasets
    logger.info("Building training dataset...")
    train_ds = DisambiguationDataset(args.train_data, tag_vocab)
    logger.info("Building validation dataset...")
    val_ds = DisambiguationDataset(args.val_data, tag_vocab)

    # Pre-cache BERTurk embeddings
    train_cache: torch.Tensor | None = None
    val_cache: torch.Tensor | None = None

    if args.pre_cache_bert:
        import gc
        import os
        cache_dir = args.cache_dir
        if cache_dir is None:
            scratch = os.environ.get("SCRATCH_DIR")
            if scratch:
                cache_dir = Path(scratch) / "bert_cache"
            else:
                cache_dir = out_dir / "bert_cache"

        # Load BERTurk ONCE and share between train/val caching to avoid
        # loading 440 MB of weights twice.
        train_cache_path = cache_dir / "train_bert_cache.pt"
        val_cache_path = cache_dir / "val_bert_cache.pt"

        if train_cache_path.exists() and val_cache_path.exists():
            train_cache = pre_cache_bert_embeddings(
                train_ds, args.berturk_path, cache_path=train_cache_path,
            )
            val_cache = pre_cache_bert_embeddings(
                val_ds, args.berturk_path, cache_path=val_cache_path,
            )
        else:
            from transformers import (  # type: ignore[import]
                AutoModel,
                AutoTokenizer,
            )
            logger.info("Loading BERTurk once for pre-caching...")
            shared_tokenizer = AutoTokenizer.from_pretrained(args.berturk_path)
            shared_bert = AutoModel.from_pretrained(args.berturk_path)
            shared_bert.eval()
            for p in shared_bert.parameters():
                p.requires_grad = False

            train_cache = pre_cache_bert_embeddings(
                train_ds, args.berturk_path,
                cache_path=train_cache_path,
                shared_bert=shared_bert,
                shared_tokenizer=shared_tokenizer,
            )
            val_cache = pre_cache_bert_embeddings(
                val_ds, args.berturk_path,
                cache_path=val_cache_path,
                shared_bert=shared_bert,
                shared_tokenizer=shared_tokenizer,
            )

            # Release BERTurk memory — ~440 MB back before building model
            del shared_bert
            del shared_tokenizer
            gc.collect()
            logger.info("Released shared BERTurk from memory")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, collate_fn=disambiguation_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, collate_fn=disambiguation_collate,
    )

    # Build model
    model = _build_model(tag_vocab, args)

    # Only train non-BERTurk params
    trainable = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in trainable)
    logger.info("Trainable parameters: %s (BERTurk frozen)", f"{total_params:,}")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    # Training loop
    best_em = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, args.device,
            bert_cache=train_cache,
        )

        val_metrics = evaluate(
            model, val_loader, args.device,
            bert_cache=val_cache,
        )

        elapsed = time.time() - t0
        logger.info(
            "Epoch %3d | Loss: %.4f | Train: %.1f%% | "
            "Val: %.1f%% | Val(ambig): %.1f%% | %.1fs",
            epoch, train_loss,
            train_acc * 100,
            val_metrics["overall_em"] * 100,
            val_metrics["ambiguous_em"] * 100,
            elapsed,
        )

        if val_metrics["overall_em"] > best_em:
            best_em = val_metrics["overall_em"]
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_em": val_metrics["overall_em"],
                    "val_ambig_em": val_metrics["ambiguous_em"],
                    "tag_vocab_size": len(tag_vocab),
                },
                out_dir / "best_model.pt",
            )
            logger.info("  -> New best: %.1f%% EM", best_em * 100)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(
                    "Early stopping at epoch %d (patience=%d)",
                    epoch, args.patience,
                )
                break

    logger.info("Best validation EM: %.1f%%", best_em * 100)

    # Save training config
    config = {
        "train_data": str(args.train_data),
        "val_data": str(args.val_data),
        "tag_vocab": str(args.tag_vocab),
        "berturk_path": args.berturk_path,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "best_em": best_em,
        "tag_vocab_size": len(tag_vocab),
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def _build_model(
    tag_vocab: Vocab,
    args: argparse.Namespace,
) -> torch.nn.Module:
    """Build the disambiguator model.

    When --pre-cache-bert is enabled, BERT is NOT loaded into the model
    (saves ~440 MB of memory during training).
    """
    from kokturk.models.disambiguator import BERTurkDisambiguator

    model = BERTurkDisambiguator(
        tag_vocab_size=len(tag_vocab),
        bert_path=args.berturk_path,
        skip_bert_loading=args.pre_cache_bert,
    ).to(args.device)

    return model


if __name__ == "__main__":
    main()
