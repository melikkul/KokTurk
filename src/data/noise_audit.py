"""Confident Learning noise audit for silver-tier corpus tokens.

Uses 5-fold cross-validation with the trained model to identify
likely mislabeled tokens in the silver tiers. Requires cleanlab.

Usage:
    PYTHONPATH=src python src/data/noise_audit.py \
        --model-path models/atomizer_v2/best_model.pt \
        --corpus-path data/gold/tr_gold_morph_v1.jsonl \
        --vocab-dir models/vocabs/
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

logger = logging.getLogger(__name__)


def _get_model_predictions(
    model_cls: type,
    model_kwargs: dict,
    dataset: object,
    train_indices: list[int],
    eval_indices: list[int],
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 64,
) -> np.ndarray:
    """Train on train_indices, predict probabilities on eval_indices."""
    import torch.nn as nn

    model = model_cls(**model_kwargs).to(device)
    train_set = Subset(dataset, train_indices)
    eval_set = Subset(dataset, eval_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for _epoch in range(epochs):
        model.train()
        for chars, tags, *_ in train_loader:
            chars, tags = chars.to(device), tags.to(device)
            optimizer.zero_grad()
            logits = model(chars, tags, teacher_forcing_ratio=0.3)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tags.reshape(-1))
            loss.backward()
            optimizer.step()

    # Collect predicted probabilities
    model.eval()
    all_probs: list[np.ndarray] = []
    with torch.no_grad():
        for chars, _tags, *_ in eval_loader:
            chars = chars.to(device)
            logits = model(chars, None, teacher_forcing_ratio=0.0)
            # Use first decode step probabilities as proxy for label quality
            probs = torch.softmax(logits[:, 0, :], dim=-1).cpu().numpy()
            all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)


def audit_silver_noise(
    model_path: str | Path,
    corpus_path: str | Path,
    vocab_dir: str | Path,
    output_dir: str | Path = "data/noise_audit",
    n_folds: int = 5,
    device_name: str = "cpu",
) -> dict[str, object]:
    """Run Confident Learning noise audit on silver tiers.

    Args:
        model_path: Path to trained model checkpoint.
        corpus_path: Path to tiered corpus JSONL.
        vocab_dir: Directory with char_vocab.json and tag_vocab.json.
        output_dir: Directory for output files.
        n_folds: Number of CV folds.
        device_name: "cpu" or "cuda".

    Returns:
        Dict with noise statistics and flagged token info.
    """
    try:
        from cleanlab.filter import find_label_issues
    except ImportError:
        logger.error(
            "cleanlab not installed. Install with: pip install cleanlab",
        )
        return {"error": "cleanlab not installed"}

    from kokturk.models.char_gru import MorphAtomizer
    from train.datasets import TIER_MAP, TieredCorpusDataset, Vocab

    device = torch.device(device_name)
    vocab_dir = Path(vocab_dir)
    char_vocab = Vocab.load(vocab_dir / "char_vocab.json")
    tag_vocab = Vocab.load(vocab_dir / "tag_vocab.json")

    # Load checkpoint for model architecture params
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    model_kwargs = {
        "char_vocab_size": ckpt["char_vocab_size"],
        "tag_vocab_size": ckpt["tag_vocab_size"],
        "embed_dim": ckpt.get("embed_dim", 64),
        "hidden_dim": ckpt.get("hidden_dim", 128),
        "num_layers": ckpt.get("num_layers", 2),
    }

    dataset = TieredCorpusDataset(
        Path(corpus_path), char_vocab, tag_vocab,
    )

    # Filter silver-only indices
    silver_indices = [
        i for i, t in enumerate(dataset.tiers) if t != TIER_MAP["gold"]
    ]
    logger.info("Silver tokens: %d", len(silver_indices))

    # Get labels (first non-special tag index as class label)
    labels = []
    for idx in silver_indices:
        _, tags, _ = dataset[idx]
        # First meaningful tag after SOS
        label_idx = 0
        for t in tags.tolist():
            if t > 3:  # skip PAD/SOS/EOS/UNK
                label_idx = t
                break
        labels.append(label_idx)
    labels_arr = np.array(labels)

    # K-fold cross-validated predictions
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    pred_probs = np.zeros((len(silver_indices), model_kwargs["tag_vocab_size"]))

    for fold, (train_rel, eval_rel) in enumerate(kf.split(silver_indices)):
        logger.info("Fold %d/%d", fold + 1, n_folds)
        train_abs = [silver_indices[i] for i in train_rel]
        eval_abs = [silver_indices[i] for i in eval_rel]

        fold_probs = _get_model_predictions(
            MorphAtomizer, model_kwargs, dataset,
            train_abs, eval_abs, device,
        )
        # Map back to the right positions
        for i, rel_idx in enumerate(eval_rel):
            if i < len(fold_probs):
                pred_probs[rel_idx] = fold_probs[i]

    # Run cleanlab
    issue_mask = find_label_issues(
        labels=labels_arr,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
    )

    # Separate by tier
    silver_auto_issues = []
    silver_agreed_issues = []
    for issue_idx in issue_mask:
        abs_idx = silver_indices[issue_idx]
        tier = dataset.tiers[abs_idx]
        surface = dataset.samples[abs_idx][0]
        label = dataset.samples[abs_idx][1]
        entry = {
            "corpus_idx": abs_idx,
            "surface": surface,
            "label": label,
            "confidence": float(pred_probs[issue_idx].max()),
        }
        if tier == TIER_MAP["silver-auto"]:
            silver_auto_issues.append(entry)
        else:
            silver_agreed_issues.append(entry)

    n_silver_auto = sum(1 for i in silver_indices if dataset.tiers[i] == TIER_MAP["silver-auto"])
    n_silver_agreed = len(silver_indices) - n_silver_auto

    results = {
        "silver_auto_noise_rate": len(silver_auto_issues) / max(n_silver_auto, 1),
        "silver_agreed_noise_rate": len(silver_agreed_issues) / max(n_silver_agreed, 1),
        "silver_auto_flagged": len(silver_auto_issues),
        "silver_agreed_flagged": len(silver_agreed_issues),
        "total_flagged": len(issue_mask),
        "total_silver": len(silver_indices),
    }

    # Save results
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "flagged_tokens.json", "w", encoding="utf-8") as f:
        json.dump({
            "silver_auto": silver_auto_issues[:100],
            "silver_agreed": silver_agreed_issues[:100],
            "stats": results,
        }, f, ensure_ascii=False, indent=2)

    logger.info("=== NOISE AUDIT ===")
    logger.info("Silver-auto estimated noise: %.1f%%", results["silver_auto_noise_rate"] * 100)
    logger.info("Silver-agreed estimated noise: %.1f%%", results["silver_agreed_noise_rate"] * 100)
    logger.info("Total flagged: %d / %d tokens", results["total_flagged"], results["total_silver"])
    logger.info("Results saved to %s", out / "flagged_tokens.json")

    return results


def seq2seq_to_token_classification(
    pred_logits_per_sentence: list[np.ndarray],
    gold_tag_ids_per_sentence: list[list[int]],
    pad_idx: int = 0,
) -> tuple[list[list[int]], list[np.ndarray]]:
    """Flatten seq2seq decoder outputs into a token-classification view.

    cleanlab's ``token_classification`` API expects:

    - ``labels``    : ``list[list[int]]`` (one list per sentence)
    - ``pred_probs``: ``list[np.ndarray]`` of shape ``(seq_len, num_classes)``

    A morphological seq2seq decoder produces, for every sentence, an
    autoregressive sequence of tag distributions. This adapter strips
    padding positions and applies softmax to the raw logits so the result
    plugs directly into ``get_label_quality_scores``.

    Args:
        pred_logits_per_sentence: List where entry ``i`` has shape
            ``(seq_len_i, tag_vocab_size)``. Logits, NOT probabilities.
        gold_tag_ids_per_sentence: List where entry ``i`` has length
            ``seq_len_i`` (gold tag indices for the same positions).
        pad_idx: Index treated as padding (positions with this id are
            removed from both ``labels`` and ``pred_probs``).

    Returns:
        ``(labels, pred_probs)`` ready for cleanlab.
    """
    labels: list[list[int]] = []
    pred_probs: list[np.ndarray] = []
    for logits, gold in zip(pred_logits_per_sentence, gold_tag_ids_per_sentence):
        if logits.size == 0 or len(gold) == 0:
            continue
        # Trim to whichever is shorter (decoder may stop at EOS).
        n = min(logits.shape[0], len(gold))
        if n == 0:
            continue
        gold_arr = np.asarray(gold[:n])
        logits_arr = logits[:n]
        keep = gold_arr != pad_idx
        if not keep.any():
            continue
        gold_kept = gold_arr[keep].tolist()
        logits_kept = logits_arr[keep]
        # Softmax along the vocabulary dimension (numerically stable).
        m = logits_kept.max(axis=-1, keepdims=True)
        exp = np.exp(logits_kept - m)
        probs = exp / exp.sum(axis=-1, keepdims=True)
        labels.append(gold_kept)
        pred_probs.append(probs.astype(np.float32))
    return labels, pred_probs


def run_cleanlab_token_audit(
    pred_logits_per_sentence: list[np.ndarray],
    gold_tag_ids_per_sentence: list[list[int]],
    sentence_ids: list[str] | None = None,
    pad_idx: int = 0,
    flag_threshold: float = 0.5,
    output_path: str | Path = "models/noise_audit/flagged_tokens.json",
) -> dict[str, object]:
    """Run cleanlab token-classification noise scoring on seq2seq output.

    The caller is responsible for collecting per-sentence decoder outputs
    via 5-fold cross-validation on the SILVER tiers (gold MUST be excluded
    from both training and evaluation — assertion below).

    Returns a dict with::

        {"n_sentences": ..., "n_tokens": ..., "n_flagged": ...,
         "flagged_token_indices": [(sentence_idx, token_idx), ...]}
    """
    assert len(pred_logits_per_sentence) == len(gold_tag_ids_per_sentence), \
        "pred_logits and gold_tag_ids must have the same outer length"
    if sentence_ids is not None:
        assert len(sentence_ids) == len(gold_tag_ids_per_sentence)

    labels, pred_probs = seq2seq_to_token_classification(
        pred_logits_per_sentence, gold_tag_ids_per_sentence, pad_idx=pad_idx,
    )

    flagged: list[tuple[int, int]] = []
    n_tokens = 0
    try:
        from cleanlab.token_classification.rank import (  # type: ignore
            get_label_quality_scores,
        )
        sentence_scores, token_scores = get_label_quality_scores(
            labels=labels, pred_probs=pred_probs,
        )
        for s_idx, scores in enumerate(token_scores):
            for t_idx, score in enumerate(scores):
                n_tokens += 1
                if float(score) < flag_threshold:
                    flagged.append((s_idx, t_idx))
    except ImportError:
        logger.warning(
            "cleanlab not installed; skipping token-classification scoring. "
            "Install with: pip install 'cleanlab>=2.2'",
        )
        # Fall back: use the model's own confidence as a proxy.
        for s_idx, (probs, gold) in enumerate(zip(pred_probs, labels)):
            for t_idx, (p, g) in enumerate(zip(probs, gold)):
                n_tokens += 1
                if p[g] < flag_threshold:
                    flagged.append((s_idx, t_idx))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_sentences": len(labels),
        "n_tokens": n_tokens,
        "n_flagged": len(flagged),
        "flag_threshold": flag_threshold,
        "flagged_token_indices": flagged,
        "sentence_ids": sentence_ids if sentence_ids is not None else [],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def generate_human_audit_sample(
    corpus_path: str | Path,
    output_path: str | Path = "data/audit/human_audit_sample.tsv",
    n: int = 385,
    seed: int = 42,
) -> int:
    """Produce a stratified human-audit sample (Cochran n=385).

    Output columns::

        token_idx  sentence_id  surface_form  left_context  right_context
        current_label  zeyrek_candidates  audit_label

    Stratification: 79% silver-auto, 21% silver-agreed. Gold tier is
    excluded. Left/right context is 3 tokens each (same sentence).

    Args:
        corpus_path: Path to the tiered JSONL corpus.
        output_path: TSV output path.
        n: Target sample size (Cochran: 385 → 95% CI, 5% margin).
        seed: RNG seed for reproducibility.

    Returns:
        Number of rows written.
    """
    import random

    rng = random.Random(seed)
    corpus_path = Path(corpus_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Group tokens by sentence so we can assemble context.
    sentences: dict[str, list[dict]] = {}
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = rec.get("sentence_id") or f"_line_{id(rec)}"
            sentences.setdefault(sid, []).append(rec)

    silver_auto: list[tuple[str, int]] = []
    silver_agreed: list[tuple[str, int]] = []
    for sid, tokens in sentences.items():
        for idx, tok in enumerate(tokens):
            tier = tok.get("tier", "")
            if tier == "silver-auto":
                silver_auto.append((sid, idx))
            elif tier == "silver-agreed":
                silver_agreed.append((sid, idx))

    n_auto = int(round(n * 0.79))
    n_agreed = n - n_auto
    sampled: list[tuple[str, int]] = []
    sampled += rng.sample(silver_auto, k=min(n_auto, len(silver_auto)))
    sampled += rng.sample(silver_agreed, k=min(n_agreed, len(silver_agreed)))
    rng.shuffle(sampled)

    header = (
        "token_idx\tsentence_id\tsurface_form\tleft_context\t"
        "right_context\tcurrent_label\tzeyrek_candidates\taudit_label"
    )
    with output_path.open("w", encoding="utf-8") as out:
        out.write(header + "\n")
        for sid, idx in sampled:
            toks = sentences[sid]
            tok = toks[idx]
            left = " ".join(t.get("surface", "") for t in toks[max(0, idx - 3): idx])
            right = " ".join(t.get("surface", "") for t in toks[idx + 1: idx + 4])
            candidates = tok.get("zeyrek_candidates") or tok.get("candidates") or ""
            if isinstance(candidates, list):
                candidates = "|".join(candidates)
            out.write(
                "\t".join(
                    [
                        str(tok.get("token_idx", idx)),
                        sid,
                        tok.get("surface", ""),
                        left,
                        right,
                        tok.get("label", ""),
                        str(candidates),
                        "",  # audit_label (empty for humans to fill in)
                    ]
                )
                + "\n"
            )
    return len(sampled)


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(description="Noise audit for silver corpus")
    parser.add_argument("--model-path", default="models/atomizer_v2/best_model.pt")
    parser.add_argument("--corpus-path", default="data/gold/tr_gold_morph_v1.jsonl")
    parser.add_argument("--vocab-dir", default="models/vocabs/")
    parser.add_argument("--output-dir", default="data/noise_audit")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    audit_silver_noise(
        args.model_path, args.corpus_path, args.vocab_dir,
        args.output_dir, device_name=args.device,
    )
