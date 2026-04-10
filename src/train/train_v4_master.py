"""Master training script for v4 morphological atomizer.

Combines all Week 3-4 innovations:
  - DualHeadAtomizer / ContextualDualHeadAtomizer / MorphAtomizer (baseline)
  - TAAC or fixed curriculum
  - Tier-weighted loss
  - Scheduled sampling (inverse sigmoid)
  - Context encoders: Word2Vec neighbourhood window, BiGRU sentence

Usage::

    # Full v4 model with context + TAAC
    python src/train/train_v4_master.py \\
        --model contextual_dual_head --context-type word2vec \\
        --w2v-path models/word2vec/tr_word2vec_128.bin \\
        --curriculum taac --output-dir models/v4 --seed 42

    # Baseline seq2seq for ablation
    python src/train/train_v4_master.py \\
        --model single_seq2seq --context-type none --curriculum fixed \\
        --output-dir models/ablations/baseline_seq2seq
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, SubsetRandomSampler

from kokturk.models.char_gru import MorphAtomizer
from kokturk.models.contextual_dual_head import ContextualDualHeadAtomizer
from kokturk.models.dual_head import DualHeadAtomizer
from train.curriculum import (
    GOLD,
    TAAC,
    compute_tier_weights,
    get_allowed_tiers,
    get_curriculum_phase,
    scheduled_sampling_ratio,
)
from train.datasets import (
    ContextualTieredDataset,
    TieredCorpusDataset,
    Vocab,
    build_word_vocab,
    contextual_collate,
)
from train.losses import build_loss
from train.rdrop import compute_rdrop_loss
from train.reproducibility import capture_environment, log_environment_to_mlflow, seed_everything

logger = logging.getLogger(__name__)

GRAD_CLIP_NORM = 5.0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Legacy wrapper — delegates to reproducibility.seed_everything."""
    seed_everything(seed)


def inject_tag_noise(
    tag_ids: torch.Tensor, noise_rate: float, tag_vocab_size: int, pad_idx: int = 0,
) -> torch.Tensor:
    """Randomly corrupt a fraction of non-PAD tag tokens."""
    if noise_rate <= 0:
        return tag_ids
    mask = (tag_ids != pad_idx) & (torch.rand_like(tag_ids.float()) < noise_rate)
    random_tags = torch.randint(1, tag_vocab_size, tag_ids.shape, device=tag_ids.device)
    return torch.where(mask, random_tags, tag_ids)


def build_tier_sampler(dataset: object, allowed_tiers: set) -> SubsetRandomSampler:
    """Return a SubsetRandomSampler filtered to the given tier set."""
    indices = [i for i, item in enumerate(dataset) if item[2] in allowed_tiers]
    if not indices:
        logger.warning("No samples found for tiers %s — using all samples", allowed_tiers)
        indices = list(range(len(dataset)))  # type: ignore[arg-type]
    return SubsetRandomSampler(indices)


def build_context_inputs(
    context_type: str,
    sent_word_ids: torch.Tensor,
    target_pos: torch.Tensor,
    device: torch.device,
    sentence_texts: list[str] | None = None,
) -> object:
    """Build context_inputs tensor(s) from a batch, based on context_type.

    Args:
        context_type: One of ``"word2vec"``, ``"bigru"``, or ``"berturk"``.
        sent_word_ids: ``(B, S)`` sentence word indices.
        target_pos: ``(B,)`` target word positions.
        device: Target device.
        sentence_texts: Raw sentence strings (required for ``"berturk"``).

    Returns:
        For ``"word2vec"``: ``(B, 4)`` neighbour index tensor.
        For ``"bigru"``: ``(word_ids, target_pos)`` tuple.
        For ``"berturk"``: ``(sentence_texts, target_pos)`` tuple.
    """
    if context_type == "berturk":
        # BERTurkContext.forward() takes (sentence_texts, target_word_positions)
        assert sentence_texts is not None, "sentence_texts required for berturk"
        positions = target_pos.tolist() if isinstance(target_pos, torch.Tensor) else target_pos
        return (sentence_texts, positions)

    sent_word_ids = sent_word_ids.to(device)
    target_pos = target_pos.to(device)

    if context_type == "word2vec":
        # Extract 4 neighbours: [t-2, t-1, t+1, t+2]; 0 = PAD at boundary.
        B, S = sent_word_ids.shape
        # Pad 2 positions on each side with PAD (0)
        padded = F.pad(sent_word_ids, (2, 2), value=0)   # (B, S+4)
        offsets = torch.tensor([[0, 1, 3, 4]], dtype=torch.long, device=device)  # (1, 4)
        pos_exp = target_pos.unsqueeze(1) + offsets  # (B, 4)
        return padded.gather(1, pos_exp)  # (B, 4)

    elif context_type == "bigru":
        return (sent_word_ids, target_pos)

    else:
        raise ValueError(f"Unknown context_type for build_context_inputs: {context_type!r}")


# ---------------------------------------------------------------------------
# Training & validation loops
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,  # type: ignore[type-arg]
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    model_type: str,
    context_type: str,
    teacher_forcing_ratio: float,
    tag_vocab_size: int,
    gold_noise_rate: float = 0.0,
    tag_loss_fn: nn.Module | None = None,
    rdrop_alpha: float = 0.0,
    ema: object | None = None,
) -> float:
    """Run one training epoch; returns mean loss.

    Args:
        tag_loss_fn: Per-element tag loss module returning shape ``(N,)``.
            When ``None`` falls back to ``F.cross_entropy(reduction='none')``
            for backward compatibility.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        optimizer.zero_grad()

        if model_type in ("single_seq2seq", "dual_head"):
            char_ids, tag_ids, tiers, root_idx = batch
            sent_word_ids = target_pos = sentence_texts = None
        else:
            char_ids, tag_ids, tiers, root_idx, sent_word_ids, target_pos, sentence_texts = batch

        char_ids = char_ids.to(device)
        tag_ids = tag_ids.to(device)
        tiers = tiers.to(device)
        root_idx = root_idx.to(device)

        # Noise injection: corrupt gold-tier tag tokens for robustness sweep
        if gold_noise_rate > 0:
            gold_mask = tiers == GOLD
            if gold_mask.any():
                noisy = inject_tag_noise(tag_ids[gold_mask], gold_noise_rate, tag_vocab_size)
                tag_ids = tag_ids.clone()
                tag_ids[gold_mask] = noisy

        tier_weights = compute_tier_weights(tiers)  # (B,)

        # ---- Forward ----
        if model_type == "single_seq2seq":
            logits = model(char_ids, tag_ids, teacher_forcing_ratio=teacher_forcing_ratio)
            B, L, V = logits.shape
            if rdrop_alpha > 0:
                logits_2 = model(
                    char_ids, tag_ids, teacher_forcing_ratio=teacher_forcing_ratio,
                )
                base_fn = tag_loss_fn if tag_loss_fn is not None else (
                    lambda lg, tg: F.cross_entropy(
                        lg, tg, ignore_index=0, reduction="none",
                    )
                )
                ce_per, kl_per, _ = compute_rdrop_loss(
                    logits.reshape(-1, V), logits_2.reshape(-1, V),
                    tag_ids.reshape(-1), base_fn,
                    alpha=rdrop_alpha, ignore_index=0,
                )
                tw = tier_weights.unsqueeze(1).expand(B, L).reshape(-1)
                non_pad = (tag_ids.reshape(-1) != 0).float()
                denom = non_pad.sum().clamp(min=1)
                ce_loss = (ce_per * tw * non_pad).sum() / denom
                kl_loss = (kl_per * non_pad).sum() / denom
                loss = ce_loss + rdrop_alpha * kl_loss
            else:
                if tag_loss_fn is not None:
                    losses = tag_loss_fn(logits.reshape(-1, V), tag_ids.reshape(-1))
                else:
                    losses = F.cross_entropy(
                        logits.reshape(-1, V), tag_ids.reshape(-1),
                        ignore_index=0, reduction="none",
                    )  # (B*L,)
                tw = tier_weights.unsqueeze(1).expand(B, L).reshape(-1)
                non_pad = (tag_ids.reshape(-1) != 0).float()
                loss = (losses * tw * non_pad).sum() / non_pad.sum().clamp(min=1)

        else:
            if model_type == "contextual_dual_head":
                context = build_context_inputs(
                    context_type, sent_word_ids, target_pos, device,  # type: ignore[arg-type]
                    sentence_texts=sentence_texts,
                )
                root_logits, tag_outputs = model(
                    char_ids, context,
                    tag_ids=tag_ids, gold_root=root_idx,
                    teacher_forcing_ratio=teacher_forcing_ratio,
                )
            else:  # dual_head
                root_logits, tag_outputs = model(
                    char_ids,
                    tag_ids=tag_ids, gold_root=root_idx,
                    teacher_forcing_ratio=teacher_forcing_ratio,
                )

            # Tag loss — tier weighted
            tag_targets = tag_ids[:, 2:]  # (B, decode_len)
            B, DL, V = tag_outputs.shape
            tw = tier_weights.unsqueeze(1).expand(B, DL).reshape(-1)
            non_pad = (tag_targets.reshape(-1) != 0).float()
            denom = non_pad.sum().clamp(min=1)

            if rdrop_alpha > 0:
                # Second forward pass through the same dual-head model.
                if model_type == "contextual_dual_head":
                    _, tag_outputs_2 = model(
                        char_ids, context,
                        tag_ids=tag_ids, gold_root=root_idx,
                        teacher_forcing_ratio=teacher_forcing_ratio,
                    )
                else:
                    _, tag_outputs_2 = model(
                        char_ids,
                        tag_ids=tag_ids, gold_root=root_idx,
                        teacher_forcing_ratio=teacher_forcing_ratio,
                    )
                base_fn = tag_loss_fn if tag_loss_fn is not None else (
                    lambda lg, tg: F.cross_entropy(
                        lg, tg, ignore_index=0, reduction="none",
                    )
                )
                ce_per, kl_per, _ = compute_rdrop_loss(
                    tag_outputs.reshape(-1, V),
                    tag_outputs_2.reshape(-1, V),
                    tag_targets.reshape(-1), base_fn,
                    alpha=rdrop_alpha, ignore_index=0,
                )
                ce_tag = (ce_per * tw * non_pad).sum() / denom
                kl_tag = (kl_per * non_pad).sum() / denom
                tag_loss = ce_tag + rdrop_alpha * kl_tag
            else:
                if tag_loss_fn is not None:
                    tag_losses = tag_loss_fn(
                        tag_outputs.reshape(-1, V), tag_targets.reshape(-1),
                    )
                else:
                    tag_losses = F.cross_entropy(
                        tag_outputs.reshape(-1, V), tag_targets.reshape(-1),
                        ignore_index=0, reduction="none",
                    )
                tag_loss = (tag_losses * tw * non_pad).sum() / denom

            # CONTRASTIVE-INJECTION-POINT:
            # When the model is configured with root_head_type="contrastive",
            # the dual-head module computes a margin loss internally and
            # exposes it via _cached_contrastive_loss. In that case the
            # standard CE root_loss MUST be skipped — Task 2's --loss-fn does
            # NOT apply here.
            cached_contrastive = getattr(model, "_cached_contrastive_loss", None)
            if cached_contrastive is not None:
                root_loss = cached_contrastive
            else:
                root_loss = F.cross_entropy(root_logits, root_idx)
            loss = tag_loss + model.root_loss_weight * root_loss

        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()
        if ema is not None:
            ema.update()  # type: ignore[attr-defined]

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def validate(
    model: nn.Module,
    loader: DataLoader,  # type: ignore[type-arg]
    device: torch.device,
    model_type: str,
    context_type: str,
) -> dict[str, float]:
    """Run validation; returns dict with loss, root_loss, tag_loss, root_acc, tag_em."""
    model.eval()
    total_loss = 0.0
    total_root_loss = 0.0
    total_tag_loss = 0.0
    correct_roots = 0
    total_roots = 0
    correct_tags = 0
    total_tags = 0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            if model_type in ("single_seq2seq", "dual_head"):
                char_ids, tag_ids, tiers, root_idx = batch
                sent_word_ids = target_pos = sentence_texts = None
            else:
                char_ids, tag_ids, tiers, root_idx, sent_word_ids, target_pos, sentence_texts = batch

            char_ids = char_ids.to(device)
            tag_ids = tag_ids.to(device)
            root_idx = root_idx.to(device)

            if model_type == "single_seq2seq":
                logits = model(char_ids, tag_ids, teacher_forcing_ratio=0.0)
                B, L, V = logits.shape
                loss = F.cross_entropy(
                    logits.reshape(-1, V), tag_ids.reshape(-1), ignore_index=0
                )

            else:
                if model_type == "contextual_dual_head":
                    context = build_context_inputs(
                        context_type, sent_word_ids, target_pos, device,  # type: ignore[arg-type]
                        sentence_texts=sentence_texts,
                    )
                    root_logits, tag_outputs = model(
                        char_ids, context,
                        tag_ids=tag_ids, gold_root=root_idx,
                        teacher_forcing_ratio=0.0,
                    )
                else:
                    root_logits, tag_outputs = model(
                        char_ids,
                        tag_ids=tag_ids, gold_root=root_idx,
                        teacher_forcing_ratio=0.0,
                    )

                tag_targets = tag_ids[:, 2:]
                B, DL, V = tag_outputs.shape
                tag_loss_val = F.cross_entropy(
                    tag_outputs.reshape(-1, V), tag_targets.reshape(-1), ignore_index=0
                )
                root_loss_val = F.cross_entropy(root_logits, root_idx)
                loss = tag_loss_val + model.root_loss_weight * root_loss_val
                total_root_loss += root_loss_val.item()
                total_tag_loss += tag_loss_val.item()

                # Root accuracy
                root_preds = root_logits.argmax(dim=-1)
                correct_roots += (root_preds == root_idx).sum().item()
                total_roots += B

                # Tag-level accuracy (non-PAD tokens)
                tag_preds = tag_outputs.argmax(dim=-1)  # (B, DL)
                mask = tag_targets != 0
                correct_tags += ((tag_preds == tag_targets) & mask).sum().item()
                total_tags += mask.sum().item()

            total_loss += loss.item()
            n_batches += 1

    n = max(n_batches, 1)
    return {
        "loss": total_loss / n,
        "root_loss": total_root_loss / n if total_root_loss > 0 else 0.0,
        "tag_loss": total_tag_loss / n if total_tag_loss > 0 else 0.0,
        "root_acc": correct_roots / max(total_roots, 1),
        "tag_acc": correct_tags / max(total_tags, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train v4 morphological atomizer")

    # Model
    p.add_argument("--model", default="contextual_dual_head",
                   choices=["single_seq2seq", "dual_head", "contextual_dual_head"],
                   help="Model architecture")
    p.add_argument("--context-type", default="word2vec",
                   choices=["word2vec", "bigru", "berturk", "none"],
                   help="Context encoder type (ignored for single_seq2seq and dual_head)")
    p.add_argument("--w2v-path", type=Path, default=None,
                   help="Path to Word2Vec .bin for word2vec context")
    p.add_argument("--berturk-path", type=Path, default=Path("models/berturk"),
                   help="Path to local BERTurk model directory")

    # Curriculum
    p.add_argument("--curriculum", default="taac", choices=["taac", "fixed"],
                   help="Curriculum scheduler")
    p.add_argument("--transition-mode", default="component",
                   choices=["loss", "component", "ensemble"],
                   help="TAAC transition criterion (ignored for fixed curriculum)")
    p.add_argument("--gold-noise-rate", type=float, default=0.0,
                   help="Fraction of gold-tier tag tokens to randomly corrupt (noise sweep)")

    # Data
    p.add_argument("--training-data", type=Path,
                   default=Path("data/splits/train.jsonl"))
    p.add_argument("--eval-data", type=Path, default=None,
                   help="Validation JSONL path (optional)")
    p.add_argument("--char-vocab", type=Path,
                   default=Path("models/vocabs/char_vocab.json"))
    p.add_argument("--tag-vocab", type=Path,
                   default=Path("models/vocabs/tag_vocab.json"))
    p.add_argument("--root-vocab", type=Path, default=None,
                   help="Root vocabulary JSON (required for dual_head / contextual_dual_head)")
    p.add_argument("--word-vocab", type=Path, default=None,
                   help="Word vocabulary JSON (required for contextual_dual_head)")

    # Architecture
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--context-dropout", type=float, default=0.3)

    # Training
    p.add_argument("--base-lr", type=float, default=5e-4)
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)

    # I/O
    # Contrastive root head (Task 6)
    p.add_argument("--root-head-type", default="mlp",
                   choices=["mlp", "contrastive"],
                   help="Root head type for dual-head models. Default 'mlp' "
                        "preserves backward compat with v2 checkpoints.")
    p.add_argument("--contrastive-margin", type=float, default=1.0,
                   help="Euclidean margin for ContrastiveRootHead.")
    p.add_argument("--root-vocab-path", type=Path, default=None,
                   help="Root vocab JSON; required for "
                        "--root-head-type contrastive.")

    # Loss function selection (Task 2)
    p.add_argument("--loss-fn", default="ce",
                   choices=["ce", "focal", "sce", "label_smooth"],
                   help="Tag-loss function. Root loss remains standard CE "
                        "unless the contrastive root head is active.")
    p.add_argument("--focal-gamma", type=float, default=2.0,
                   help="Focal Loss focusing parameter (gamma).")
    p.add_argument("--sce-alpha", type=float, default=1.0,
                   help="Symmetric CE forward-CE weight.")
    p.add_argument("--sce-beta", type=float, default=0.4,
                   help="Symmetric CE reverse-CE weight.")
    p.add_argument("--label-smoothing", type=float, default=0.1,
                   help="Label smoothing epsilon (also used as compound "
                        "smoothing inside FocalLoss when --loss-fn focal).")

    # Category C — regularization
    p.add_argument("--rdrop-alpha", type=float, default=0.0,
                   help="R-Drop KL weight. 0 disables. Research default 5.0.")
    p.add_argument("--variational-dropout", type=float, default=0.0,
                   help="Variational (locked) dropout probability on RNN I/O.")
    p.add_argument("--weight-dropout", type=float, default=0.0,
                   help="DropConnect probability on GRU weight_hh_l* matrices.")
    p.add_argument("--augment-keyboard", type=float, default=0.0,
                   help="Keyboard-proximity char noise probability.")
    p.add_argument("--augment-diacritic", type=float, default=0.0,
                   help="Diacritic strip/swap probability.")
    p.add_argument("--augment-stemcorrupt", type=float, default=0.0,
                   help="STEMCORRUPT sample probability.")
    p.add_argument("--ema-decay", type=float, default=0.0,
                   help="EMA decay. 0 disables. Recommended 0.999.")
    p.add_argument("--early-stop-metric", default="val_loss",
                   choices=["val_loss", "macro_f1"],
                   help="Early-stopping criterion.")
    p.add_argument("--early-stop-patience", type=int, default=5)
    p.add_argument("--optimizer", default="adam", choices=["adam", "adamw"],
                   help="Optimizer selection.")
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="Weight decay (AdamW decouples it from gradient).")

    # Category G — scalability
    p.add_argument("--bucket-batching", action="store_true", default=False,
                   help="Enable dynamic bucket batching to minimise padding waste. "
                        "Groups similar-length sequences together.")

    p.add_argument("--output-dir", type=Path, default=Path("models/v4"))
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Resume training from checkpoint")
    p.add_argument("--device", default=None,
                   help="Force device (cpu/cuda). Auto-detect if omitted.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    set_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ---- Output dir ----
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Vocabs ----
    char_vocab = Vocab.load(args.char_vocab)
    tag_vocab = Vocab.load(args.tag_vocab)
    logger.info("Vocabs: %d chars, %d tags", len(char_vocab), len(tag_vocab))

    # ---- Tag loss function (Task 2) ----
    tag_loss_fn = build_loss(
        args.loss_fn,
        num_classes=len(tag_vocab),
        ignore_index=0,  # PAD
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        sce_alpha=args.sce_alpha,
        sce_beta=args.sce_beta,
    ).to(device)
    logger.info(
        "Tag loss: %s (gamma=%.2f, eps=%.2f, sce_alpha=%.2f, sce_beta=%.2f)",
        args.loss_fn, args.focal_gamma, args.label_smoothing,
        args.sce_alpha, args.sce_beta,
    )
    try:  # optional MLflow logging
        import mlflow  # noqa: PLC0415
        mlflow.log_param("loss_fn", args.loss_fn)
        mlflow.log_param("focal_gamma", args.focal_gamma)
        mlflow.log_param("label_smoothing", args.label_smoothing)
        mlflow.log_param("sce_alpha", args.sce_alpha)
        mlflow.log_param("sce_beta", args.sce_beta)
    except Exception:  # noqa: BLE001
        pass

    root_vocab: dict[str, int] | None = None
    if args.root_vocab is not None and args.root_vocab.exists():
        with open(args.root_vocab, encoding="utf-8") as f:
            _rv_raw = json.load(f)
        # Vocab files are JSON arrays; convert to {token: idx} dict
        if isinstance(_rv_raw, list):
            root_vocab = {tok: idx for idx, tok in enumerate(_rv_raw)}
        else:
            root_vocab = _rv_raw
        logger.info("Root vocab: %d roots", len(root_vocab))
    elif args.model in ("dual_head", "contextual_dual_head"):
        raise ValueError(
            f"--root-vocab required for model={args.model!r}"
        )

    word_vocab: dict[str, int] | None = None
    if args.model == "contextual_dual_head":
        if args.word_vocab is not None and args.word_vocab.exists():
            with open(args.word_vocab, encoding="utf-8") as f:
                word_vocab = json.load(f)
            logger.info("Word vocab: %d words", len(word_vocab))
        else:
            logger.info("Building word vocab from training data ...")
            word_vocab = build_word_vocab(args.training_data)
            logger.info("Word vocab built: %d words", len(word_vocab))

    root_vocab_size = len(root_vocab) if root_vocab else 4  # dummy for single_seq2seq

    # ---- Context encoder ----
    context_encoder = None
    if args.model == "contextual_dual_head":
        if args.context_type == "word2vec":
            from kokturk.models.context_encoder import Word2VecContext
            vocab_size = len(word_vocab)  # type: ignore[arg-type]
            pretrained = None
            if args.w2v_path is not None and args.w2v_path.exists():
                try:
                    import gensim.models  # type: ignore[import]
                    logger.info("Loading Word2Vec from %s ...", args.w2v_path)
                    w2v = gensim.models.KeyedVectors.load_word2vec_format(
                        str(args.w2v_path), binary=True
                    )
                    embed_dim = w2v.vector_size
                    pretrained = torch.zeros(vocab_size, embed_dim)
                    for word, idx in word_vocab.items():  # type: ignore[union-attr]
                        if word in w2v.key_to_index:
                            pretrained[idx] = torch.tensor(w2v[word])
                    logger.info("Pre-trained Word2Vec loaded (dim=%d)", embed_dim)
                except Exception as e:
                    logger.warning("Could not load Word2Vec: %s — using random init", e)
            context_encoder = Word2VecContext(
                vocab_size=vocab_size,
                embed_dim=args.embed_dim * 2,   # typical w2v dim is larger
                gru_hidden_dim=args.hidden_dim // 2,
                pretrained_weights=pretrained,
            )

        elif args.context_type == "bigru":
            from kokturk.models.context_encoder import SentenceBiGRUContext
            context_encoder = SentenceBiGRUContext(
                vocab_size=len(word_vocab),  # type: ignore[arg-type]
                embed_dim=args.embed_dim * 2,
                hidden_dim=args.hidden_dim // 2,
                dropout=args.dropout,
            )
        elif args.context_type == "berturk":
            from kokturk.models.context_encoder import BERTurkContext
            bert_path = str(args.berturk_path) if args.berturk_path else "models/berturk"
            context_encoder = BERTurkContext(
                bert_path=bert_path,
                context_dim=args.hidden_dim,
            )
            logger.info("BERTurk context encoder from %s", bert_path)
        else:
            raise ValueError(
                f"context_type={args.context_type!r} not supported for contextual_dual_head"
            )

    # ---- Model ----
    if args.model == "single_seq2seq":
        model: nn.Module = MorphAtomizer(
            char_vocab_size=len(char_vocab),
            tag_vocab_size=len(tag_vocab),
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            variational_dropout=args.variational_dropout,
            weight_dropout=args.weight_dropout,
        )

    elif args.model == "dual_head":
        model = DualHeadAtomizer(
            char_vocab_size=len(char_vocab),
            tag_vocab_size=len(tag_vocab),
            root_vocab_size=root_vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            root_head_type=args.root_head_type,
            contrastive_margin=args.contrastive_margin,
            root_vocab_path=str(args.root_vocab_path) if args.root_vocab_path else
                            (str(args.root_vocab) if args.root_vocab else None),
            variational_dropout=args.variational_dropout,
            weight_dropout=args.weight_dropout,
        )

    else:  # contextual_dual_head
        model = ContextualDualHeadAtomizer(
            context_encoder=context_encoder,
            char_vocab_size=len(char_vocab),
            tag_vocab_size=len(tag_vocab),
            root_vocab_size=root_vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            context_dropout=args.context_dropout,
            variational_dropout=args.variational_dropout,
            weight_dropout=args.weight_dropout,
        )

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %s | Parameters: %d", args.model, n_params)

    # ---- Datasets ----
    use_contextual = args.model == "contextual_dual_head"

    if use_contextual:
        train_ds = ContextualTieredDataset(
            path=args.training_data,
            char_vocab=char_vocab,
            tag_vocab=tag_vocab,
            word_vocab=word_vocab,   # type: ignore[arg-type]
            root_vocab=root_vocab,
        )
        collate_fn = contextual_collate
        val_ds = ContextualTieredDataset(
            path=args.eval_data,
            char_vocab=char_vocab,
            tag_vocab=tag_vocab,
            word_vocab=word_vocab,   # type: ignore[arg-type]
            root_vocab=root_vocab,
        ) if args.eval_data else None

    else:
        # Build augmenter if any of the flags enabled
        train_augmenter = None
        if (args.augment_keyboard > 0 or args.augment_diacritic > 0
                or args.augment_stemcorrupt > 0):
            from data.char_augmentation import (
                CompositeAugmenter,
                DiacriticAugmenter,
                KeyboardAugmenter,
                StemCorruptAugmenter,
            )
            augs: list[tuple[object, float]] = []
            if args.augment_keyboard > 0:
                augs.append((KeyboardAugmenter(noise_prob=args.augment_keyboard), 1.0))
            if args.augment_diacritic > 0:
                augs.append((DiacriticAugmenter(prob=args.augment_diacritic), 1.0))
            if args.augment_stemcorrupt > 0:
                augs.append((StemCorruptAugmenter(corrupt_prob=args.augment_stemcorrupt), 1.0))
            train_augmenter = CompositeAugmenter(augs)
            logger.info("Train augmenter: %d augmenters active", len(augs))

        train_ds = TieredCorpusDataset(
            path=args.training_data,
            char_vocab=char_vocab,
            tag_vocab=tag_vocab,
            root_vocab=root_vocab,
            augmenter=train_augmenter,
        )
        collate_fn = None
        val_ds = TieredCorpusDataset(
            path=args.eval_data,
            char_vocab=char_vocab,
            tag_vocab=tag_vocab,
            root_vocab=root_vocab,
        ) if args.eval_data else None

    logger.info("Train dataset: %d samples | %s", len(train_ds), getattr(train_ds, "tier_counts", {}))

    # Validation loader (no tier filtering)
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size,
            shuffle=False, num_workers=0,
            collate_fn=collate_fn,
        )
        logger.info("Val dataset: %d samples", len(val_ds))

    # ---- Optimizer ----
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay,
        )
    logger.info("Optimizer: %s (weight_decay=%.4f)", args.optimizer, args.weight_decay)

    # ---- EMA ----
    ema = None
    if args.ema_decay > 0:
        from train.ema import EMAWeights
        ema = EMAWeights(model, decay=args.ema_decay)
        logger.info("EMA enabled (decay=%.4f)", args.ema_decay)

    # ---- Resume from checkpoint ----
    start_epoch = 1
    best_val_loss = float("inf")
    if args.checkpoint is not None and args.checkpoint.exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 1) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        logger.info("Resumed from checkpoint %s (epoch %d)", args.checkpoint, start_epoch - 1)

    # ---- Curriculum ----
    curriculum: TAAC | None = None
    if args.curriculum == "taac":
        curriculum = TAAC(
            epsilon=0.01, patience=2,
            min_epochs_per_phase=2, max_epochs_per_phase=10,
            transition_mode=args.transition_mode,
        )
        allowed_tiers = get_allowed_tiers("gold_only")
    else:
        allowed_tiers = get_allowed_tiers(
            get_curriculum_phase(start_epoch, total_epochs=args.max_epochs)
        )

    # ---- MLflow (optional) ----
    mlflow_run = None
    try:
        import mlflow  # type: ignore[import]
        mlflow.set_experiment("train_v4_master")
        mlflow_run = mlflow.start_run()
        mlflow.log_params({
            "model": args.model,
            "context_type": args.context_type,
            "curriculum": args.curriculum,
            "embed_dim": args.embed_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "base_lr": args.base_lr,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "n_params": n_params,
        })
        log_environment_to_mlflow(capture_environment())
    except Exception:
        pass

    # ---- Save CLI args ----
    with open(args.output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    # ---- Training loop ----
    logger.info("Starting training: max_epochs=%d, curriculum=%s", args.max_epochs, args.curriculum)
    # Early stopping: best score so far, epochs since last improvement.
    # For "macro_f1" we use val_tag_acc as a proxy (micro token accuracy).
    es_best = float("inf") if args.early_stop_metric == "val_loss" else -float("inf")
    es_bad_epochs = 0

    for epoch in range(start_epoch, args.max_epochs + 1):
        epoch_start = time.time()

        # Teacher forcing ratio (separate decay for root vs tags)
        tf_ratio = scheduled_sampling_ratio(epoch - 1, k=5.0)

        # Build DataLoader with tier-filtered sampler
        if (
            args.bucket_batching
            and hasattr(train_ds, "char_lengths")
            and collate_fn is None
        ):
            from train.bucket_batching import (
                BucketBatchSampler,
                analyze_batching_efficiency,
                dynamic_pad_collate,
            )

            tier_indices = [
                i for i, t in enumerate(train_ds.tiers)
                if t in allowed_tiers
            ]
            if not tier_indices:
                tier_indices = list(range(len(train_ds)))
            bucket_sampler = BucketBatchSampler(
                train_ds.char_lengths,
                batch_size=args.batch_size,
                indices=tier_indices,
                shuffle=True,
                seed=args.seed,
            )
            bucket_sampler.set_epoch(epoch)
            train_loader = DataLoader(
                train_ds, batch_sampler=bucket_sampler,
                num_workers=0, collate_fn=dynamic_pad_collate,
            )
            # Log efficiency stats on first epoch
            if epoch == start_epoch:
                eff = analyze_batching_efficiency(
                    train_ds.char_lengths, args.batch_size,
                )
                logger.info(
                    "Bucket batching: naive_pad=%.1f%% bucket_pad=%.1f%% "
                    "speedup=%.2fx",
                    eff["naive_pad_fraction"] * 100,
                    eff["bucket_pad_fraction"] * 100,
                    eff["speedup_estimate"],
                )
                try:
                    import mlflow  # type: ignore[import]
                    mlflow.log_metrics({
                        "naive_pad_fraction": eff["naive_pad_fraction"],
                        "bucket_pad_fraction": eff["bucket_pad_fraction"],
                    })
                except Exception:
                    pass
        else:
            sampler = build_tier_sampler(train_ds, allowed_tiers)
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size,
                sampler=sampler, num_workers=0,
                collate_fn=collate_fn,
            )

        # Train one epoch
        train_loss = train_epoch(
            model, train_loader, optimizer, device,
            model_type=args.model,
            context_type=args.context_type,
            teacher_forcing_ratio=tf_ratio,
            tag_vocab_size=len(tag_vocab),
            gold_noise_rate=args.gold_noise_rate,
            tag_loss_fn=tag_loss_fn,
            rdrop_alpha=args.rdrop_alpha,
            ema=ema,
        )

        # Validate (use EMA weights if enabled)
        val_metrics: dict[str, float] = {"loss": float("nan")}
        if val_loader is not None:
            if ema is not None:
                ema.apply()  # type: ignore[attr-defined]
            try:
                val_metrics = validate(
                    model, val_loader, device,
                    model_type=args.model,
                    context_type=args.context_type,
                )
            finally:
                if ema is not None:
                    ema.restore()  # type: ignore[attr-defined]
        val_loss = val_metrics["loss"]
        val_root_loss = val_metrics.get("root_loss", 0.0)
        val_tag_loss = val_metrics.get("tag_loss", 0.0)
        val_root_acc = val_metrics.get("root_acc", 0.0)
        val_tag_acc = val_metrics.get("tag_acc", 0.0)

        elapsed = time.time() - epoch_start

        # Curriculum step
        if args.curriculum == "taac":
            info = curriculum.step(  # type: ignore[union-attr]
                val_loss if not np.isnan(val_loss) else train_loss,
                root_loss=val_root_loss,
                tag_loss=val_tag_loss,
            )
            allowed_tiers = info["allowed_tiers"]
            lr_mult = info["lr_multiplier"]
            phase = info["phase"]
            if info["should_transition"]:
                logger.info("  [TAAC] Transitioned to phase: %s", phase)
        else:
            phase = get_curriculum_phase(epoch, total_epochs=args.max_epochs)
            allowed_tiers = get_allowed_tiers(phase)
            lr_mult = 1.0  # fixed curriculum uses base LR only

        # Update learning rate
        new_lr = args.base_lr * lr_mult
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr

        logger.info(
            "Epoch %3d/%d | phase=%-22s | tf=%.3f | train_loss=%.4f | "
            "val_loss=%.4f | root_acc=%.3f | tag_acc=%.3f | lr=%.1e | %.1fs",
            epoch, args.max_epochs, phase, tf_ratio, train_loss, val_loss,
            val_root_acc, val_tag_acc, new_lr, elapsed,
        )

        # Log to MLflow
        if mlflow_run is not None:
            try:
                import mlflow  # type: ignore[import]
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss if not np.isnan(val_loss) else 0.0,
                    "val_root_acc": val_root_acc,
                    "val_tag_acc": val_tag_acc,
                    "lr": new_lr,
                    "teacher_forcing_ratio": tf_ratio,
                }, step=epoch)
            except Exception:
                pass

        # Early stopping check
        if args.early_stop_metric == "val_loss":
            es_score = val_loss if not np.isnan(val_loss) else train_loss
            improved = es_score < es_best
        else:  # macro_f1 proxy
            es_score = val_tag_acc
            improved = es_score > es_best
        if improved:
            es_best = es_score
            es_bad_epochs = 0
        else:
            es_bad_epochs += 1

        # Save best checkpoint
        save_loss = val_loss if not np.isnan(val_loss) else train_loss
        if save_loss < best_val_loss:
            best_val_loss = save_loss
            ckpt_path = args.output_dir / "best_model.pt"
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "args": vars(args),
            }, ckpt_path)
            logger.info("  Saved best checkpoint → %s (loss=%.4f)", ckpt_path, best_val_loss)

        if es_bad_epochs >= args.early_stop_patience:
            logger.info(
                "Early stopping: no improvement in %s for %d epochs",
                args.early_stop_metric, es_bad_epochs,
            )
            break

    # ---- Final save ----
    final_path = args.output_dir / "final_model.pt"
    torch.save({
        "model": model.state_dict(),
        "epoch": args.max_epochs,
        "best_val_loss": best_val_loss,
        "args": vars(args),
    }, final_path)
    logger.info("Final model saved → %s", final_path)

    if mlflow_run is not None:
        try:
            import mlflow  # type: ignore[import]
            mlflow.end_run()
        except Exception:
            pass

    logger.info("Training complete. Best val loss: %.4f", best_val_loss)


if __name__ == "__main__":
    main()
