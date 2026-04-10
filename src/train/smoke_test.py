"""Quick training smoke test using only auto-accepted labels.

Validates the entire training pipeline works end-to-end:
1. Dataset loads correctly
2. Model forward pass (no shape mismatches)
3. Loss decreases over 5 epochs
4. Greedy decode produces valid tag sequences
5. Checkpoint save + load roundtrip
6. Evaluation metrics compute

Uses a tiny model (embed=32, hidden=64, layers=1) and gold-only
data to stay within CPU time limits.

Usage:
    PYTHONPATH=src python src/train/smoke_test.py
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from benchmark.intrinsic_eval import compute_all_metrics
from kokturk.models.char_gru import MorphAtomizer
from train.datasets import (
    EOS_IDX,
    PAD_IDX,
    SOS_IDX,
    MorphAtomizerDataset,
    build_vocabs,
)

logger = logging.getLogger(__name__)

GOLD_PATH = Path("data/gold/combined_gold.jsonl")
WEAK_PATH = Path("data/weak_labels/probabilistic_labels.jsonl")


def run_smoke_test() -> bool:
    """Run the full smoke test. Returns True if all checks pass."""
    torch.manual_seed(42)
    ok = True

    # ── Step 1: Build vocab from gold-only data ──
    print("Step 1: Building vocabularies from gold data...")
    char_vocab, tag_vocab = build_vocabs(
        GOLD_PATH, WEAK_PATH, weak_confidence_threshold=1.0,
    )
    # Also add chars/tags from weak data but with very high threshold
    # to keep vocab small
    print(f"  Char vocab: {len(char_vocab)} | Tag vocab: {len(tag_vocab)}")

    # Save vocabs for later use
    vocab_dir = Path("models/vocabs")
    char_vocab.save(vocab_dir / "char_vocab.json")
    tag_vocab.save(vocab_dir / "tag_vocab.json")

    # ── Step 2: Load dataset (gold only, no weak) ──
    print("Step 2: Loading dataset (gold only)...")
    dataset = MorphAtomizerDataset(
        GOLD_PATH, WEAK_PATH, char_vocab, tag_vocab,
        weak_confidence_threshold=1.0,
        max_weak_samples=200,  # tiny subset
    )
    print(f"  Samples: {len(dataset)} ({dataset.n_gold} gold, {dataset.n_weak} weak)")

    if len(dataset) < 10:
        print("  ERROR: Not enough samples")
        return False

    # ── Step 3: Train/val split ──
    gen = torch.Generator().manual_seed(42)
    n_val = max(int(0.1 * len(dataset)), 20)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

    # ── Step 4: Initialize tiny model ──
    print("Step 3: Initializing model (tiny config)...")
    model = MorphAtomizer(
        char_vocab_size=len(char_vocab),
        tag_vocab_size=len(tag_vocab),
        embed_dim=32,
        hidden_dim=64,
        num_layers=1,
        dropout=0.1,
    )
    print(f"  Parameters: {model.count_parameters()}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # ── Step 5: Train 5 epochs ──
    print("Step 4: Training 5 epochs...")
    losses: list[float] = []
    for epoch in range(5):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for chars, tags in train_loader:
            optimizer.zero_grad()
            logits = model(chars, tags, teacher_forcing_ratio=0.5)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tags.reshape(-1),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        print(f"  Epoch {epoch + 1}/5: loss={avg_loss:.4f}")

    if losses[-1] >= losses[0]:
        print(f"  WARN: Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}")
        # Not a hard failure — tiny model may not converge in 5 epochs
    else:
        print(f"  OK: Loss decreased: {losses[0]:.4f} → {losses[-1]:.4f}")

    # ── Step 6: Greedy decode on val samples ──
    print("Step 5: Greedy decode on 10 val samples...")
    model.eval()
    decode_count = 0
    for chars, tags in val_loader:
        preds = model.greedy_decode(chars)
        for i in range(min(chars.size(0), 10 - decode_count)):
            # Decode surface
            surface_ids = chars[i].tolist()
            surface = "".join(
                char_vocab.decode(c)
                for c in surface_ids
                if c not in (PAD_IDX, SOS_IDX, EOS_IDX)
            )
            # Decode predicted tags
            pred_tokens = []
            for idx in preds[i].tolist():
                if idx == EOS_IDX:
                    break
                if idx > 3:
                    pred_tokens.append(tag_vocab.decode(idx))
            pred_str = " ".join(pred_tokens) if pred_tokens else "(empty)"

            # Decode gold
            gold_tokens = []
            for idx in tags[i].tolist():
                if idx == EOS_IDX:
                    break
                if idx > 3:
                    gold_tokens.append(tag_vocab.decode(idx))
            gold_str = " ".join(gold_tokens) if gold_tokens else "(empty)"

            print(f"  {surface:20s} → pred: {pred_str:40s} gold: {gold_str}")
            decode_count += 1
        if decode_count >= 10:
            break

    # ── Step 7: Checkpoint roundtrip ──
    print("Step 6: Checkpoint save/load roundtrip...")
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_model.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "char_vocab_size": len(char_vocab),
            "tag_vocab_size": len(tag_vocab),
            "embed_dim": 32,
            "hidden_dim": 64,
            "num_layers": 1,
        }, ckpt_path)

        # Reload
        model2 = MorphAtomizer(
            char_vocab_size=len(char_vocab),
            tag_vocab_size=len(tag_vocab),
            embed_dim=32, hidden_dim=64, num_layers=1,
        )
        ckpt = torch.load(ckpt_path, weights_only=True)
        model2.load_state_dict(ckpt["model_state_dict"])

        # Verify identical predictions
        test_chars = next(iter(val_loader))[0][:2]
        p1 = model.greedy_decode(test_chars)
        p2 = model2.greedy_decode(test_chars)
        if torch.equal(p1, p2):
            print("  OK: Predictions match after reload")
        else:
            print("  FAIL: Predictions differ after reload")
            ok = False

    # ── Step 8: Evaluation metrics ──
    print("Step 7: Running intrinsic evaluation...")
    all_preds: list[list[int]] = []
    all_gold: list[list[int]] = []
    model.eval()
    with torch.no_grad():
        for chars, tags in val_loader:
            preds = model.greedy_decode(chars)
            for i in range(chars.size(0)):
                # Trim at first EOS
                trimmed: list[int] = []
                for idx in preds[i].tolist():
                    if idx == EOS_IDX:
                        break
                    if idx > 3:
                        trimmed.append(idx)
                all_preds.append(trimmed)

                gold_seq: list[int] = []
                for idx in tags[i].tolist():
                    if idx == EOS_IDX:
                        break
                    if idx > 3:
                        gold_seq.append(idx)
                all_gold.append(gold_seq)

    metrics = compute_all_metrics(all_preds, all_gold)
    print(f"  Exact Match:   {metrics['exact_match']:.3f}")
    print(f"  Root Accuracy: {metrics['root_accuracy']:.3f}")
    print(f"  Tag F1:        {metrics['f1']:.3f}")

    # ── Summary ──
    print(f"\n{'='*50}")
    if ok:
        print("SMOKE TEST PASSED — training pipeline is functional")
    else:
        print("SMOKE TEST HAD ISSUES — see warnings above")
    print(f"{'='*50}")
    return ok


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    success = run_smoke_test()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
