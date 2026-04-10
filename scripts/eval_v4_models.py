"""Evaluate v4 ablation / TAAC / BERTurk checkpoints on test split.

Loads a checkpoint saved by train_v4_master.py, reconstructs the model,
greedy-decodes the test set, and reports EM / Root Acc / Tag F1 via
string-level comparison of canonical labels.

Usage::

    PYTHONPATH=src python scripts/eval_v4_models.py <checkpoint> [<label>]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kokturk.models.char_gru import MorphAtomizer  # noqa: E402
from kokturk.models.contextual_dual_head import ContextualDualHeadAtomizer  # noqa: E402
from kokturk.models.dual_head import DualHeadAtomizer  # noqa: E402
from train.datasets import (  # noqa: E402
    EOS_IDX,
    ContextualTieredDataset,
    TieredCorpusDataset,
    Vocab,
    contextual_collate,
)


def _load_vocab_dict(path: Path) -> dict[str, int]:
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return {tok: i for i, tok in enumerate(obj)}
    return obj


def _build_model(args: dict, char_vocab_size: int, tag_vocab_size: int,
                 root_vocab_size: int, word_vocab_size: int | None):
    model_type = args["model"]
    embed_dim = args["embed_dim"]
    hidden_dim = args["hidden_dim"]
    num_layers = args["num_layers"]
    dropout = args.get("dropout", 0.3)

    if model_type == "single_seq2seq":
        return MorphAtomizer(
            char_vocab_size=char_vocab_size,
            tag_vocab_size=tag_vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    if model_type == "dual_head":
        return DualHeadAtomizer(
            char_vocab_size=char_vocab_size,
            tag_vocab_size=tag_vocab_size,
            root_vocab_size=root_vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    context_type = args.get("context_type", "word2vec")
    context_dropout = args.get("context_dropout", 0.3)
    if context_type == "word2vec":
        from kokturk.models.context_encoder import Word2VecContext
        ctx = Word2VecContext(
            vocab_size=word_vocab_size, embed_dim=embed_dim,
            gru_hidden_dim=hidden_dim // 2,
        )
    elif context_type == "bigru":
        from kokturk.models.context_encoder import SentenceBiGRUContext
        ctx = SentenceBiGRUContext(
            vocab_size=word_vocab_size, embed_dim=embed_dim * 2,
            hidden_dim=hidden_dim // 2, dropout=dropout,
        )
    elif context_type == "berturk":
        from kokturk.models.context_encoder import BERTurkContext
        bert_path = args.get("berturk_path") or "models/berturk"
        ctx = BERTurkContext(bert_path=str(bert_path), context_dim=hidden_dim)
    else:
        raise ValueError(f"Unknown context_type {context_type}")

    return ContextualDualHeadAtomizer(
        context_encoder=ctx,
        char_vocab_size=char_vocab_size,
        tag_vocab_size=tag_vocab_size,
        root_vocab_size=root_vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        context_dropout=context_dropout,
    )


def _tag_ids_to_string(tag_ids: list[int], tag_vocab_inv: list[str]) -> str:
    """Convert tag id sequence into a canonical string like 'ev +PLU +ABL'.

    Skips SOS/PAD/UNK; stops at EOS. Index 0 is expected to be the root token,
    subsequent indices are tag tokens. But in our layout tag_ids starts at SOS,
    so we need to skip tokens ≤ 3 (PAD, SOS, EOS, UNK).
    """
    tokens: list[str] = []
    for idx in tag_ids:
        if idx == EOS_IDX:
            break
        if idx <= 3:  # PAD, SOS, EOS, UNK
            continue
        if idx < len(tag_vocab_inv):
            tokens.append(tag_vocab_inv[idx])
    return " ".join(tokens)


def _get_root(canonical: str) -> str:
    return canonical.split()[0] if canonical.strip() else ""


def _get_tags(canonical: str) -> list[str]:
    parts = canonical.split()
    return parts[1:] if len(parts) > 1 else []


def _tag_f1(pred_tags: list[str], gold_tags: list[str]) -> float:
    if not pred_tags and not gold_tags:
        return 1.0
    pred_set = set(pred_tags)
    gold_set = set(gold_tags)
    if not pred_set or not gold_set:
        return 0.0
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate(checkpoint_path: str, label: str = "") -> dict[str, float]:
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    args = ckpt["args"]
    model_type = args["model"]

    char_vocab = Vocab.load(Path(args["char_vocab"]))
    tag_vocab = Vocab.load(Path(args["tag_vocab"]))

    # Build inverse vocabs for string rendering
    tag_vocab_inv: list[str] = [""] * len(tag_vocab)
    for tok, idx in tag_vocab.token2idx.items():
        if idx < len(tag_vocab_inv):
            tag_vocab_inv[idx] = tok

    root_vocab: dict[str, int] | None = None
    root_vocab_inv: list[str] | None = None
    if args.get("root_vocab"):
        root_vocab = _load_vocab_dict(Path(args["root_vocab"]))
        root_vocab_inv = [""] * len(root_vocab)
        for tok, idx in root_vocab.items():
            if idx < len(root_vocab_inv):
                root_vocab_inv[idx] = tok

    word_vocab: dict[str, int] | None = None
    if args.get("word_vocab") and Path(args["word_vocab"]).exists():
        with open(args["word_vocab"], encoding="utf-8") as f:
            word_vocab = json.load(f)

    model = _build_model(
        args,
        len(char_vocab), len(tag_vocab),
        len(root_vocab) if root_vocab else 4,
        len(word_vocab) if word_vocab else None,
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    test_path = Path("data/splits/test.jsonl")
    if model_type == "contextual_dual_head":
        test_ds = ContextualTieredDataset(
            path=test_path, char_vocab=char_vocab, tag_vocab=tag_vocab,
            word_vocab=word_vocab, root_vocab=root_vocab,
        )
        loader = DataLoader(test_ds, batch_size=32, shuffle=False,
                            collate_fn=contextual_collate)
    else:
        test_ds = TieredCorpusDataset(
            path=test_path, char_vocab=char_vocab, tag_vocab=tag_vocab,
            root_vocab=root_vocab,
        )
        loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # We need gold labels as strings too. For that we load the raw JSONL.
    gold_labels_raw: list[str] = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            gold_labels_raw.append(json.loads(line).get("label", ""))

    all_preds_str: list[str] = []
    gold_cursor = 0

    with torch.no_grad():
        for batch in loader:
            if model_type == "contextual_dual_head":
                char_ids, tag_ids, _, _, sent_ids, target_pos, texts = batch
            else:
                char_ids, tag_ids, _, _ = batch

            B = char_ids.size(0)

            if model_type == "single_seq2seq":
                pred_tokens = model.greedy_decode(char_ids)
                # Tensor of tag ids — convert to strings
                pred_lists = pred_tokens.tolist() if isinstance(pred_tokens, torch.Tensor) else pred_tokens
                for row in pred_lists:
                    all_preds_str.append(_tag_ids_to_string(row, tag_vocab_inv))

            elif model_type == "dual_head":
                labels = model.greedy_decode(
                    char_ids,
                    root_vocab_inv=root_vocab_inv,
                    tag_vocab_inv=tag_vocab_inv,
                )
                all_preds_str.extend(labels)

            else:  # contextual_dual_head
                if args.get("context_type") == "word2vec":
                    import torch.nn.functional as F
                    padded = F.pad(sent_ids, (2, 2), value=0)
                    offsets = torch.tensor([[0, 1, 3, 4]], dtype=torch.long)
                    pos_exp = target_pos.unsqueeze(1) + offsets
                    context = padded.gather(1, pos_exp)
                elif args.get("context_type") == "berturk":
                    context = (list(texts), target_pos.tolist())
                else:
                    context = (sent_ids, target_pos)
                labels = model.greedy_decode(
                    char_ids, context,
                    root_vocab_inv=root_vocab_inv,
                    tag_vocab_inv=tag_vocab_inv,
                )
                all_preds_str.extend(labels)

            gold_cursor += B

    # Match gold to predictions (skip gold records with empty label)
    # ContextualTieredDataset may reorder by sentence_id; safer to align via dataset samples
    gold_labels: list[str] = []
    for surface, label in test_ds.samples:
        gold_labels.append(label)

    n = min(len(all_preds_str), len(gold_labels))
    all_preds_str = all_preds_str[:n]
    gold_labels = gold_labels[:n]

    # Compute metrics
    em = 0
    root_correct = 0
    f1_sum = 0.0
    for pred, gold in zip(all_preds_str, gold_labels, strict=True):
        if pred == gold:
            em += 1
        if _get_root(pred) == _get_root(gold):
            root_correct += 1
        f1_sum += _tag_f1(_get_tags(pred), _get_tags(gold))

    metrics = {
        "exact_match": em / max(n, 1),
        "root_accuracy": root_correct / max(n, 1),
        "tag_f1": f1_sum / max(n, 1),
        "n": n,
        "best_val_loss": ckpt.get("best_val_loss", 0.0),
    }

    print(f"\n=== {label or checkpoint_path} ===")
    print(f"  n        = {metrics['n']}")
    print(f"  EM       = {metrics['exact_match']:.1%}")
    print(f"  Root Acc = {metrics['root_accuracy']:.1%}")
    print(f"  Tag F1   = {metrics['tag_f1']:.1%}")
    print(f"  val_loss = {metrics['best_val_loss']:.4f}")
    return metrics


if __name__ == "__main__":
    ckpt = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else ""
    evaluate(ckpt, label)
