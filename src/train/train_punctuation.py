"""Noktalama restorasyonu modelini eğit.

train_disambiguator.py ile aynı kalıp:
- BERTurk gömmelerini önceden hesapla
- Küçük sınıflandırma başlığını eğit (~200K parametre)
- ~5-10 dk CPU eğitimi

Kullanım::

    PYTHONPATH=src python -m train.train_punctuation \\
        --train-data data/gold/tr_gold_morph_v1.conllu \\
        --val-data data/gold/tr_gold_morph_v1.conllu \\
        --berturk-path models/berturk \\
        --output-dir models/punctuation
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from kokturk.models.punctuation_restorer import PUNCT_LABELS, PunctuationRestorer
from train.punctuation_dataset import PunctuationDataset, punctuation_collate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Eğitim döngüsü
# ---------------------------------------------------------------------------


def train_epoch(
    model: PunctuationRestorer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Bir epoch eğitim. Döner: (ortalama loss, doğruluk)."""
    model.train()
    if model.bert is not None:
        model.bert.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        optimizer.zero_grad()

        all_embeddings: list[torch.Tensor] = []
        max_words = batch["labels"].shape[1]

        for sentence in batch["sentences"]:
            emb = model.get_word_embeddings(sentence)
            if emb.shape[0] < max_words:
                pad = torch.zeros(max_words - emb.shape[0], 768)
                emb = torch.cat([emb, pad], dim=0)
            else:
                emb = emb[:max_words]
            all_embeddings.append(emb)

        embeddings = torch.stack(all_embeddings).to(device)
        labels = batch["labels"].to(device)

        logits, loss = model(embeddings, labels)
        if loss is None:
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 5.0
        )
        optimizer.step()

        total_loss += loss.item()

        mask = labels != -1
        preds = logits.argmax(dim=-1)
        correct += (preds[mask] == labels[mask]).sum().item()
        total += mask.sum().item()

    avg_loss = total_loss / max(len(loader), 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def evaluate(
    model: PunctuationRestorer,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Doğrulama kümesi üzerinde doğruluk hesapla."""
    model.eval()
    correct = 0
    total = 0

    class_correct: dict[int, int] = {}
    class_total: dict[int, int] = {}

    with torch.no_grad():
        for batch in loader:
            all_embeddings: list[torch.Tensor] = []
            max_words = batch["labels"].shape[1]

            for sentence in batch["sentences"]:
                emb = model.get_word_embeddings(sentence)
                if emb.shape[0] < max_words:
                    pad = torch.zeros(max_words - emb.shape[0], 768)
                    emb = torch.cat([emb, pad], dim=0)
                else:
                    emb = emb[:max_words]
                all_embeddings.append(emb)

            embeddings = torch.stack(all_embeddings).to(device)
            labels = batch["labels"].to(device)

            logits, _ = model(embeddings, labels)
            preds = logits.argmax(dim=-1)

            mask = labels != -1
            correct += (preds[mask] == labels[mask]).sum().item()
            total += mask.sum().item()

            for pred, gold in zip(preds[mask], labels[mask], strict=True):
                g = gold.item()
                class_total[g] = class_total.get(g, 0) + 1
                if pred.item() == g:
                    class_correct[g] = class_correct.get(g, 0) + 1

    accuracy = correct / max(total, 1)

    # Sınıf bazlı doğruluk
    inv = {v: k for k, v in PUNCT_LABELS.items()}
    print("  Sınıf bazlı doğruluk:")
    for cls_id in sorted(class_total):
        cls_name = inv.get(cls_id, "?")
        cls_acc = class_correct.get(cls_id, 0) / class_total[cls_id]
        print(
            f"    {cls_name:12s}: {cls_acc:.1%}"
            f" ({class_correct.get(cls_id, 0)}/{class_total[cls_id]})"
        )

    return accuracy


# ---------------------------------------------------------------------------
# Ana giriş noktası
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Noktalama restorasyonu modeli eğitimi"
    )
    parser.add_argument("--train-data", required=True, help="CoNLL-U veya JSONL")
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--berturk-path", default="models/berturk")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", default="models/punctuation")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO)

    print("Veri kümeleri yükleniyor...")
    train_ds = PunctuationDataset(args.train_data)
    val_ds = PunctuationDataset(args.val_data)
    print(f"  Eğitim: {len(train_ds)} cümle")
    print(f"  Doğrulama: {len(val_ds)} cümle")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=punctuation_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        collate_fn=punctuation_collate,
    )

    device = torch.device(args.device)
    model = PunctuationRestorer(args.berturk_path).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    param_count = sum(p.numel() for p in trainable)
    print(f"Eğitilebilir: {param_count:,} parametre")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d} | Loss: {train_loss:.4f}"
            f" | Eğitim: {train_acc:.1%} | Doğrulama: {val_acc:.1%}"
            f" | {elapsed:.0f}s"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                },
                f"{args.output_dir}/best_model.pt",
            )
            print(f"  ✓ En iyi model kaydedildi (doğruluk: {val_acc:.1%})")

    print(f"\nEn iyi doğrulama doğruluğu: {best_acc:.1%}")


if __name__ == "__main__":
    main()
