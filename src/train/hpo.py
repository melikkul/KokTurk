"""Optuna-based hyperparameter optimization for the morphological atomizer.

Uses TPE sampler + HyperbandPruner (ASHA) for multi-fidelity search. Each
trial trains a small :class:`MorphAtomizer` for ``max_epochs`` epochs and
reports validation loss at every epoch to drive pruning. The final
objective value is tag-level accuracy (a macro-F1 proxy — per-tag F1 is
expensive and not needed for trial ranking).

Heavy dependencies (``optuna``, ``torch``) are imported lazily so the
module can be importable without optuna installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["create_study", "objective", "run_hpo"]


def create_study(
    study_name: str = "morpho_hpo",
    storage: str | None = None,
    direction: str = "maximize",
) -> Any:
    """Create an Optuna study with TPE sampler + HyperbandPruner.

    Args:
        study_name: Optuna study name.
        storage: SQLite URL (e.g. ``"sqlite:///models/hpo/optuna.db"``).
            ``None`` uses in-memory storage.
        direction: ``"maximize"`` (default) or ``"minimize"``.
    """
    import optuna

    pruner = optuna.pruners.HyperbandPruner(
        min_resource=2, max_resource=15, reduction_factor=3,
    )
    sampler = optuna.samplers.TPESampler(seed=42)
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )


def _build_tiny_dataset(train_path: str, val_path: str):
    """Build TieredCorpusDataset pair from JSONL paths."""
    from pathlib import Path as _P

    from train.datasets import TieredCorpusDataset, Vocab

    # Reuse project vocabs
    char_vocab = Vocab.load(_P("models/vocabs/char_vocab.json"))
    tag_vocab = Vocab.load(_P("models/vocabs/tag_vocab.json"))
    train_ds = TieredCorpusDataset(
        path=_P(train_path), char_vocab=char_vocab, tag_vocab=tag_vocab,
    )
    val_ds = TieredCorpusDataset(
        path=_P(val_path), char_vocab=char_vocab, tag_vocab=tag_vocab,
    )
    return char_vocab, tag_vocab, train_ds, val_ds


def objective(
    trial: Any,
    train_path: str,
    val_path: str,
    max_epochs: int = 15,
    device: str = "cpu",
) -> float:
    """Single Optuna trial.

    Search space is the one listed in Cat C plan — see Task 5 docstring.
    Reports intermediate ``val_loss`` at every epoch for ASHA pruning and
    returns final val tag accuracy (macro-F1 proxy) on success.
    """
    import optuna
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from kokturk.models.char_gru import MorphAtomizer
    from train.losses import build_loss

    embed_dim = trial.suggest_int("embedding_dim", 64, 256, step=32)
    hidden_dim = trial.suggest_int("hidden_dim", 128, 512, step=64)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.6, step=0.05)
    vdrop = trial.suggest_float("variational_dropout", 0.0, 0.4, step=0.05)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.05, step=0.01)
    weight_decay = trial.suggest_float("weight_decay", 1e-3, 5e-2, log=True)
    loss_fn = trial.suggest_categorical("loss_fn", ["ce", "focal", "label_smooth"])
    focal_gamma = (
        trial.suggest_float("focal_gamma", 1.0, 3.0) if loss_fn == "focal" else 2.0
    )
    rdrop_alpha = trial.suggest_categorical("rdrop_alpha", [0.0, 5.0, 10.0])

    char_vocab, tag_vocab, train_ds, val_ds = _build_tiny_dataset(train_path, val_path)

    model = MorphAtomizer(
        char_vocab_size=len(char_vocab),
        tag_vocab_size=len(tag_vocab),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        variational_dropout=vdrop,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    tag_loss = build_loss(
        loss_fn, num_classes=len(tag_vocab), ignore_index=0,
        focal_gamma=focal_gamma, label_smoothing=label_smoothing,
    ).to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    from train.rdrop import compute_rdrop_loss

    best_val_acc = 0.0
    for epoch in range(max_epochs):
        model.train()
        for chars, tags, _, _ in train_loader:
            chars = chars.to(device)
            tags = tags.to(device)
            logits = model(chars, tags, teacher_forcing_ratio=0.5)
            B, L, V = logits.shape
            if rdrop_alpha > 0:
                logits_2 = model(chars, tags, teacher_forcing_ratio=0.5)
                ce, kl, _ = compute_rdrop_loss(
                    logits.reshape(-1, V), logits_2.reshape(-1, V),
                    tags.reshape(-1), tag_loss,
                    alpha=rdrop_alpha, ignore_index=0,
                )
                non_pad = (tags.reshape(-1) != 0).float()
                denom = non_pad.sum().clamp(min=1)
                loss = (ce * non_pad).sum() / denom + rdrop_alpha * (kl * non_pad).sum() / denom
            else:
                per = tag_loss(logits.reshape(-1, V), tags.reshape(-1))
                non_pad = (tags.reshape(-1) != 0).float()
                loss = (per * non_pad).sum() / non_pad.sum().clamp(min=1)
            optim.zero_grad()
            loss.backward()
            optim.step()

        # Validate
        model.eval()
        val_loss_total = 0.0
        correct = 0
        total = 0
        n_batches = 0
        with torch.no_grad():
            for chars, tags, _, _ in val_loader:
                chars = chars.to(device)
                tags = tags.to(device)
                logits = model(chars, tags, teacher_forcing_ratio=0.0)
                B, L, V = logits.shape
                vl = F.cross_entropy(
                    logits.reshape(-1, V), tags.reshape(-1), ignore_index=0,
                )
                val_loss_total += float(vl)
                preds = logits.argmax(dim=-1)
                mask = tags != 0
                correct += int(((preds == tags) & mask).sum())
                total += int(mask.sum())
                n_batches += 1
        val_loss = val_loss_total / max(n_batches, 1)
        val_acc = correct / max(total, 1)
        best_val_acc = max(best_val_acc, val_acc)

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_acc


def run_hpo(
    n_trials: int = 50,
    train_path: str = "data/gold/train.tsv",
    val_path: str = "data/gold/val.tsv",
    max_epochs_per_trial: int = 15,
    device: str = "cpu",
    study_name: str = "morpho_hpo",
    storage: str = "sqlite:///models/hpo/optuna.db",
    output_json: str = "models/hpo/best_params.json",
) -> dict:
    """Run a full HPO campaign and persist best parameters.

    Returns the best trial's ``params`` dict.
    """
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    study = create_study(
        study_name=study_name, storage=storage, direction="maximize",
    )
    study.optimize(
        lambda t: objective(t, train_path, val_path, max_epochs_per_trial, device),
        n_trials=n_trials,
        catch=(Exception,),
    )
    best = dict(study.best_trial.params)
    with open(output_json, "w") as f:
        json.dump({
            "best_value": study.best_trial.value,
            "best_params": best,
            "n_trials": len(study.trials),
        }, f, indent=2)
    return best


def main() -> None:  # pragma: no cover
    import argparse

    p = argparse.ArgumentParser(description="Run Optuna HPO")
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--train-path", default="data/gold/train.jsonl")
    p.add_argument("--val-path", default="data/gold/val.jsonl")
    p.add_argument("--max-epochs", type=int, default=15)
    p.add_argument("--device", default="cpu")
    p.add_argument("--study-name", default="morpho_hpo")
    p.add_argument("--storage", default="sqlite:///models/hpo/optuna.db")
    args = p.parse_args()

    best = run_hpo(
        n_trials=args.n_trials,
        train_path=args.train_path,
        val_path=args.val_path,
        max_epochs_per_trial=args.max_epochs,
        device=args.device,
        study_name=args.study_name,
        storage=args.storage,
    )
    print("Best params:", best)


if __name__ == "__main__":  # pragma: no cover
    main()
