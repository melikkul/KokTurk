"""Train the GRU Seq2Seq morphological atomizer.

Supports both legacy mode (gold+weak labels) and v3 mode with:
- Curriculum training (4-phase tier scheduling)
- Tier-weighted loss (gold 5x, silver-auto 1x, silver-agreed 0.7x)
- Scheduled sampling (inverse sigmoid decay)
- Copy mechanism with root auxiliary head

Usage:
    # Legacy mode (backward compatible)
    python src/train/train_atomizer.py

    # v3 mode with all improvements
    python src/train/train_atomizer.py \
        --use-curriculum --use-tier-weights --use-scheduled-sampling \
        --use-copy-mechanism --output-dir models/atomizer_v3/ --epochs 30
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split

from kokturk.models.char_gru import MorphAtomizer
from train.datasets import MorphAtomizerDataset, TieredCorpusDataset, Vocab

logger = logging.getLogger(__name__)

# Defaults matching configs/train/atomizer_gru.yaml
GOLD_PATH = Path("data/gold/combined_gold.jsonl")
WEAK_PATH = Path("data/weak_labels/probabilistic_labels.jsonl")
CHAR_VOCAB_PATH = Path("models/vocabs/char_vocab.json")
TAG_VOCAB_PATH = Path("models/vocabs/tag_vocab.json")
OUTPUT_DIR = Path("models/draft_v1")

EMBED_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 64
EPOCHS = 30
LR = 0.001
WEIGHT_DECAY = 0.0001
GRAD_CLIP = 1.0
TF_START = 0.5
TF_END = 0.0
SEED = 42


def evaluate(
    model: MorphAtomizer,
    dataloader: DataLoader,  # type: ignore[type-arg]
    criterion: nn.Module,
    tag_vocab: Vocab,
    device: torch.device,
    is_copy_model: bool = False,
) -> dict[str, float]:
    """Evaluate model on a validation set.

    Returns:
        Dict with loss, exact_match, root_accuracy, tag_f1.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    exact_matches = 0
    root_matches = 0
    tag_tp = 0
    tag_fp = 0
    tag_fn = 0

    eos_idx = 2

    with torch.no_grad():
        for chars, tags, *_ in dataloader:
            chars = chars.to(device)
            tags = tags.to(device)

            if is_copy_model:
                logits, _root_logits = model(chars, tags, teacher_forcing_ratio=0.0)
            else:
                logits = model(chars, tags, teacher_forcing_ratio=0.0)
            # logits: (B, L, V), tags: (B, L)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tags.reshape(-1),
            )
            total_loss += loss.item() * chars.size(0)
            total_tokens += chars.size(0)

            preds = logits.argmax(dim=-1)  # (B, L)

            for b in range(chars.size(0)):
                # Extract until EOS
                pred_seq = []
                for idx in preds[b].tolist():
                    if idx == eos_idx:
                        break
                    if idx > 3:  # skip PAD/SOS/EOS/UNK
                        pred_seq.append(idx)

                gold_seq = []
                for idx in tags[b].tolist():
                    if idx == eos_idx:
                        break
                    if idx > 3:
                        gold_seq.append(idx)

                if pred_seq == gold_seq:
                    exact_matches += 1

                # Root = first non-special token
                pred_root = pred_seq[0] if pred_seq else -1
                gold_root = gold_seq[0] if gold_seq else -2
                if pred_root == gold_root:
                    root_matches += 1

                # Tag F1 (set-based)
                pred_tags = set(pred_seq[1:]) if len(pred_seq) > 1 else set()
                gold_tags = set(gold_seq[1:]) if len(gold_seq) > 1 else set()
                tag_tp += len(pred_tags & gold_tags)
                tag_fp += len(pred_tags - gold_tags)
                tag_fn += len(gold_tags - pred_tags)

    precision = tag_tp / max(tag_tp + tag_fp, 1)
    recall = tag_tp / max(tag_tp + tag_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "val_loss": total_loss / max(total_tokens, 1),
        "exact_match": exact_matches / max(total_tokens, 1),
        "root_accuracy": root_matches / max(total_tokens, 1),
        "tag_f1": f1,
    }


def train_model(
    epochs: int = EPOCHS,
    device_name: str = "cpu",
    seed: int = SEED,
    output_dir: str | Path = OUTPUT_DIR,
    embed_dim: int = EMBED_DIM,
    hidden_dim: int = HIDDEN_DIM,
    num_layers: int = NUM_LAYERS,
    dropout: float = DROPOUT,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    use_curriculum: bool = False,
    use_tier_weights: bool = False,
    use_scheduled_sampling: bool = False,
    use_copy_mechanism: bool = False,
    gold_path: str | Path = GOLD_PATH,
    weak_path: str | Path = WEAK_PATH,
    train_path: str | Path | None = None,
    val_path: str | Path | None = None,
) -> Path:
    """Train the GRU atomizer model.

    Args:
        epochs: Number of training epochs.
        device_name: "cpu" or "cuda".
        seed: Random seed.
        output_dir: Directory for model checkpoints.
        use_curriculum: Enable 4-phase curriculum training.
        use_tier_weights: Enable tier-weighted loss.
        use_scheduled_sampling: Use inverse sigmoid TF decay.
        use_copy_mechanism: Use MorphAtomizerCopy with root head.
        train_path: Path to train split (for tiered training).
        val_path: Path to val split (for tiered training).

    Returns:
        Path to the best model checkpoint.
    """
    torch.manual_seed(seed)

    device = torch.device(device_name)
    logger.info("Training on device: %s (seed=%d)", device, seed)

    # Load vocabs
    char_vocab = Vocab.load(CHAR_VOCAB_PATH)
    tag_vocab = Vocab.load(TAG_VOCAB_PATH)
    logger.info("Vocabs: %d chars, %d tags", len(char_vocab), len(tag_vocab))

    # Determine training mode: tiered (v3) or legacy
    use_tiered = train_path is not None or use_curriculum or use_tier_weights

    if use_tiered:
        t_path = Path(train_path) if train_path else Path("data/splits/train.jsonl")
        v_path = Path(val_path) if val_path else Path("data/splits/val.jsonl")
        train_ds = TieredCorpusDataset(t_path, char_vocab, tag_vocab, gold_weight=2.0)
        val_ds = TieredCorpusDataset(v_path, char_vocab, tag_vocab)
        logger.info("Tiered train: %d samples %s", len(train_ds), train_ds.tier_counts)
        logger.info("Tiered val: %d samples %s", len(val_ds), val_ds.tier_counts)

        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        # Legacy mode: gold + weak labels with random split
        max_weak = 8000 if device_name == "cpu" else 0
        dataset = MorphAtomizerDataset(
            Path(gold_path), Path(weak_path), char_vocab, tag_vocab,
            max_weak_samples=max_weak,
        )
        logger.info(
            "Dataset: %d samples (%d gold, %d weak)",
            len(dataset), dataset.n_gold, dataset.n_weak,
        )
        gen = torch.Generator().manual_seed(seed)
        n_val = max(int(0.1 * len(dataset)), 100)
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=gen)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    if use_copy_mechanism:
        from kokturk.models.char_gru_copy import MorphAtomizerCopy

        model = MorphAtomizerCopy(
            char_vocab_size=len(char_vocab),
            tag_vocab_size=len(tag_vocab),
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)
    else:
        model = MorphAtomizer(
            char_vocab_size=len(char_vocab),
            tag_vocab_size=len(tag_vocab),
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)
    logger.info("Model parameters: %d", model.count_parameters())

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    # Loss: reduction='none' if tier weights are used, else mean
    if use_tier_weights and use_tiered:
        criterion_none = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
        criterion_eval = nn.CrossEntropyLoss(ignore_index=0)
    else:
        criterion_none = None
        criterion_eval = nn.CrossEntropyLoss(ignore_index=0)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_em = 0.0
    best_path = output_dir / "best_model.pt"

    # Import curriculum if needed
    if use_curriculum or use_scheduled_sampling:
        from train.curriculum import (
            compute_tier_weights,
            get_allowed_tiers,
            get_curriculum_phase,
            get_learning_rate,
            scheduled_sampling_ratio,
        )

    for epoch in range(epochs):
        epoch_1 = epoch + 1  # 1-indexed for curriculum

        # Determine teacher forcing ratio
        if use_scheduled_sampling:
            tf_ratio = scheduled_sampling_ratio(epoch)
        else:
            tf_ratio = TF_START - (TF_START - TF_END) * epoch / max(epochs - 1, 1)

        # Curriculum: adjust learning rate and data subset
        if use_curriculum and use_tiered:
            phase = get_curriculum_phase(epoch_1)
            allowed_tiers = get_allowed_tiers(phase)
            new_lr = get_learning_rate(epoch_1)
            for pg in optimizer.param_groups:
                pg["lr"] = new_lr

            # Build sampler from precomputed tier indices
            indices = []
            for tier in allowed_tiers:
                indices.extend(train_ds.tier_indices.get(tier, []))
            if not indices:
                indices = list(range(len(train_ds)))
            sampler = SubsetRandomSampler(indices)
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, sampler=sampler, num_workers=0,
            )
            logger.info(
                "  Phase: %s  tiers=%s  samples=%d  lr=%.2e",
                phase, allowed_tiers, len(indices), new_lr,
            )
        elif use_tiered:
            # Full dataset, no curriculum
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
            )

        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            chars = batch[0].to(device)
            tags = batch[1].to(device)
            tiers = batch[2] if len(batch) > 2 else None

            optimizer.zero_grad()

            if use_copy_mechanism:
                logits, root_logits = model(chars, tags, teacher_forcing_ratio=tf_ratio)
                # Root auxiliary loss
                gold_root_idx = tags[:, 1]  # first tag after SOS
                root_loss = nn.functional.cross_entropy(
                    root_logits, gold_root_idx, ignore_index=0,
                )
            else:
                logits = model(chars, tags, teacher_forcing_ratio=tf_ratio)
                root_loss = torch.tensor(0.0, device=device)

            # Main loss
            if use_tier_weights and criterion_none is not None and tiers is not None:
                tiers_tensor = torch.tensor(
                    [t if isinstance(t, int) else t.item() for t in tiers],
                    device=device, dtype=torch.long,
                )
                per_token_loss = criterion_none(
                    logits.reshape(-1, logits.size(-1)),
                    tags.reshape(-1),
                )  # (B * L,)
                tier_w = compute_tier_weights(tiers_tensor)  # (B,)
                # Expand tier weights to per-token level
                seq_len = tags.size(1)
                token_weights = tier_w.unsqueeze(1).expand(-1, seq_len).reshape(-1)
                loss = (per_token_loss * token_weights).sum() / token_weights.sum()
            else:
                loss = criterion_eval(
                    logits.reshape(-1, logits.size(-1)),
                    tags.reshape(-1),
                )

            # Add root auxiliary loss
            if use_copy_mechanism:
                loss = loss + model.root_loss_weight * root_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0

        # Validate
        metrics = evaluate(
            model, val_loader, criterion_eval, tag_vocab, device,
            is_copy_model=use_copy_mechanism,
        )

        phase_str = ""
        if use_curriculum and use_tiered:
            phase_str = f"  phase={get_curriculum_phase(epoch_1)}"

        logger.info(
            "Epoch %2d/%d  loss=%.4f  val_loss=%.4f  EM=%.3f  "
            "root=%.3f  F1=%.3f  tf=%.2f%s  (%.1fs)",
            epoch_1, epochs, avg_loss, metrics["val_loss"],
            metrics["exact_match"], metrics["root_accuracy"],
            metrics["tag_f1"], tf_ratio, phase_str, elapsed,
        )

        if metrics["exact_match"] > best_em:
            best_em = metrics["exact_match"]
            ckpt = {
                "model_state_dict": model.state_dict(),
                "char_vocab_size": len(char_vocab),
                "tag_vocab_size": len(tag_vocab),
                "embed_dim": embed_dim,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "epoch": epoch_1,
                "best_em": best_em,
                "metrics": metrics,
                "seed": seed,
                "use_copy_mechanism": use_copy_mechanism,
            }
            if use_curriculum and use_tiered:
                ckpt["curriculum_phase"] = get_curriculum_phase(epoch_1)
            torch.save(ckpt, best_path)
            logger.info("  → New best model (EM=%.3f)", best_em)

    logger.info("Training complete. Best EM: %.3f", best_em)
    return best_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train morphological atomizer")
    parser.add_argument("--gold-path", default=str(GOLD_PATH))
    parser.add_argument("--weak-path", default=str(WEAK_PATH))
    parser.add_argument("--train-path", default=None, help="Tiered train split path")
    parser.add_argument("--val-path", default=None, help="Tiered val split path")
    parser.add_argument("--vocab-dir", default="models/vocabs/")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--embed-dim", type=int, default=EMBED_DIM)
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default=None, help="cpu or cuda (auto-detect if omitted)")
    parser.add_argument("--use-curriculum", action="store_true")
    parser.add_argument("--use-tier-weights", action="store_true")
    parser.add_argument("--use-scheduled-sampling", action="store_true")
    parser.add_argument("--use-copy-mechanism", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    best_path = train_model(
        epochs=args.epochs,
        device_name=device,
        seed=args.seed,
        output_dir=args.output_dir,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        lr=args.lr,
        use_curriculum=args.use_curriculum,
        use_tier_weights=args.use_tier_weights,
        use_scheduled_sampling=args.use_scheduled_sampling,
        use_copy_mechanism=args.use_copy_mechanism,
        gold_path=args.gold_path,
        weak_path=args.weak_path,
        train_path=args.train_path,
        val_path=args.val_path,
    )
    print(f"\nBest model saved to: {best_path}")


if __name__ == "__main__":
    main()
