#!/usr/bin/env python3
"""Evaluate ALL 25 trained model checkpoints on the FULL test set (8140 tokens).

Auto-detects model architecture from checkpoint keys, handles both checkpoint
formats (model_state_dict vs model key), and outputs a clean TSV table.

Usage:
    PYTHONPATH=src python scripts/eval_full_test.py

Environment variables:
    MAX_TEST_SAMPLES: cap on test samples (default: unlimited = full test set)
    BATCH_SIZE: inference batch size (default: 256)
    OUTPUT_TSV: path for results TSV (default: models/benchmark/full_test_eval.tsv)
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from train.datasets import Vocab, PAD_IDX, SOS_IDX, EOS_IDX  # noqa: E402

TEST_PATH = PROJECT / "data" / "splits" / "test.jsonl"
CHAR_VOCAB_PATH = PROJECT / "models" / "vocabs" / "char_vocab.json"
TAG_VOCAB_PATH = PROJECT / "models" / "vocabs" / "tag_vocab.json"
ROOT_VOCAB_PATH = PROJECT / "models" / "vocabs" / "root_vocab.json"

MAX_CHAR_LEN = 64
MAX_TAG_LEN = 15
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "256"))
MAX_TEST_SAMPLES = int(os.environ.get("MAX_TEST_SAMPLES", "0"))  # 0 = all
OUTPUT_TSV = Path(os.environ.get(
    "OUTPUT_TSV",
    str(PROJECT / "models" / "benchmark" / "full_test_eval.tsv"),
))

# ---------------------------------------------------------------------------
# All 25 checkpoint paths (relative to PROJECT)
# ---------------------------------------------------------------------------
CHECKPOINT_PATHS: list[str] = [
    "models/draft_v1/best_model.pt",
    "models/atomizer_v2/best_model.pt",
    "models/atomizer_v3/best_model.pt",
    "models/ensemble/model_seed42/best_model.pt",
    "models/ensemble/model_seed123/best_model.pt",
    "models/ensemble/model_seed456/best_model.pt",
    "models/ensemble/model_seed789/best_model.pt",
    "models/ensemble/model_seed1337/best_model.pt",
    "models/noise_sweep/noise_0.0_fixed/best_model.pt",
    "models/noise_sweep/noise_0.0_taac/best_model.pt",
    "models/noise_sweep/noise_0.05_fixed/best_model.pt",
    "models/noise_sweep/noise_0.05_taac/best_model.pt",
    "models/noise_sweep/noise_0.10_fixed/best_model.pt",
    "models/noise_sweep/noise_0.10_taac/best_model.pt",
    "models/noise_sweep/noise_0.20_fixed/best_model.pt",
    "models/noise_sweep/noise_0.20_taac/best_model.pt",
    "models/ablations/baseline_seq2seq/best_model.pt",
    "models/ablations/dual_head/best_model.pt",
    "models/ablations/dual_head_taac/best_model.pt",
    "models/ablations/dual_head_context/best_model.pt",
    "models/ablations/full_v4/best_model.pt",
    "models/v4/best_model.pt",
    "models/v4_berturk/best_model.pt",
    "models/taac_retest/id3/best_model.pt",
    "models/taac_retest/id4/best_model.pt",
]

# Model name → architecture mapping for cases where auto-detection is
# ambiguous or we want a readable label.
ARCH_MAP: dict[str, str] = {
    "draft_v1": "seq2seq",
    "atomizer_v2": "seq2seq",
    "atomizer_v3": "seq2seq",
    "ensemble/model_seed42": "seq2seq",
    "ensemble/model_seed123": "seq2seq",
    "ensemble/model_seed456": "seq2seq",
    "ensemble/model_seed789": "seq2seq",
    "ensemble/model_seed1337": "seq2seq",
    "ablations/baseline_seq2seq": "seq2seq",
    "ablations/dual_head": "dual_head",
    "ablations/dual_head_taac": "dual_head",
    "noise_sweep/noise_0.0_fixed": "dual_head",
    "noise_sweep/noise_0.0_taac": "dual_head",
    "noise_sweep/noise_0.05_fixed": "dual_head",
    "noise_sweep/noise_0.05_taac": "dual_head",
    "noise_sweep/noise_0.10_fixed": "dual_head",
    "noise_sweep/noise_0.10_taac": "dual_head",
    "noise_sweep/noise_0.20_fixed": "dual_head",
    "noise_sweep/noise_0.20_taac": "dual_head",
    "ablations/dual_head_context": "contextual_dual_head",
    "ablations/full_v4": "contextual_dual_head",
    "v4": "contextual_dual_head",
    "v4_berturk": "contextual_dual_head",
    "taac_retest/id3": "contextual_dual_head",
    "taac_retest/id4": "contextual_dual_head",
}


# ---------------------------------------------------------------------------
# Architecture detection from state dict keys
# ---------------------------------------------------------------------------

def detect_model_class(state_dict: dict) -> str:
    """Detect architecture from checkpoint weight keys."""
    has_context = any(
        "context_encoder" in k or "ctx_proj" in k for k in state_dict
    )
    has_root_head = any("root_head" in k for k in state_dict)
    if has_context:
        return "contextual_dual_head"
    if has_root_head:
        return "dual_head"
    return "seq2seq"


def detect_context_type(state_dict: dict) -> str:
    """Detect which context encoder is inside a contextual checkpoint."""
    if any("context_encoder.bert" in k for k in state_dict):
        return "berturk"
    if any("context_encoder.dropout" in k for k in state_dict):
        return "sentence_bigru"
    if any("context_encoder.word_embed" in k for k in state_dict):
        return "word2vec"
    if any("context_encoder.pos_embed" in k for k in state_dict):
        return "pos_bigram"
    return "word2vec"


def infer_sizes(state_dict: dict, mc: str) -> dict:
    """Infer model hyperparameters from weight tensor shapes."""
    info: dict = {}
    info["char_vocab_size"] = state_dict["encoder.char_embed.weight"].shape[0]
    info["embed_dim"] = state_dict["encoder.char_embed.weight"].shape[1]

    # Handle both bare GRU and WeightDropout-wrapped GRU
    if "encoder.gru.weight_hh_l0" in state_dict:
        info["hidden_dim"] = state_dict["encoder.gru.weight_hh_l0"].shape[1]
        gru_prefix = "encoder.gru.weight_hh_l"
    elif "encoder.gru.module.weight_hh_l0" in state_dict:
        info["hidden_dim"] = state_dict["encoder.gru.module.weight_hh_l0"].shape[1]
        gru_prefix = "encoder.gru.module.weight_hh_l"
    else:
        # Fallback: scan for any weight_hh pattern
        for k in state_dict:
            if "encoder" in k and "weight_hh_l0" in k:
                info["hidden_dim"] = state_dict[k].shape[1]
                gru_prefix = k.replace("weight_hh_l0", "weight_hh_l")
                break
        else:
            info["hidden_dim"] = 128  # safe default
            gru_prefix = None

    # Count GRU layers
    if gru_prefix is not None:
        n = 0
        while f"{gru_prefix}{n}" in state_dict:
            n += 1
        info["num_layers"] = max(n, 1)
    else:
        info["num_layers"] = 2

    if mc == "seq2seq":
        info["tag_vocab_size"] = state_dict["decoder.output_proj.weight"].shape[0]
    else:
        info["tag_vocab_size"] = state_dict["tag_decoder.output_proj.weight"].shape[0]
        info["root_vocab_size"] = state_dict["root_head.fc2.weight"].shape[0]

    if mc == "contextual_dual_head":
        ct = detect_context_type(state_dict)
        info["context_type"] = ct
        if ct in ("word2vec", "sentence_bigru"):
            info["ctx_vocab_size"] = state_dict[
                "context_encoder.word_embed.weight"
            ].shape[0]
            info["ctx_embed_dim"] = state_dict[
                "context_encoder.word_embed.weight"
            ].shape[1]
            if "context_encoder.gru.weight_hh_l0" in state_dict:
                info["ctx_gru_hidden"] = state_dict[
                    "context_encoder.gru.weight_hh_l0"
                ].shape[1]
            else:
                info["ctx_gru_hidden"] = 64
        elif ct == "berturk":
            info["ctx_context_dim"] = state_dict[
                "context_encoder.proj.weight"
            ].shape[0]

    return info


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(mc: str, info: dict) -> torch.nn.Module:
    """Instantiate the correct model class with inferred hyperparameters."""
    if mc == "seq2seq":
        from kokturk.models.char_gru import MorphAtomizer
        return MorphAtomizer(
            char_vocab_size=info["char_vocab_size"],
            tag_vocab_size=info["tag_vocab_size"],
            embed_dim=info["embed_dim"],
            hidden_dim=info["hidden_dim"],
            num_layers=info["num_layers"],
            max_decode_len=MAX_TAG_LEN,
        )

    if mc == "dual_head":
        from kokturk.models.dual_head import DualHeadAtomizer
        return DualHeadAtomizer(
            char_vocab_size=info["char_vocab_size"],
            tag_vocab_size=info["tag_vocab_size"],
            root_vocab_size=info["root_vocab_size"],
            embed_dim=info["embed_dim"],
            hidden_dim=info["hidden_dim"],
            num_layers=info["num_layers"],
            max_decode_len=MAX_TAG_LEN,
        )

    if mc == "contextual_dual_head":
        ct = info.get("context_type", "word2vec")
        if ct == "word2vec":
            from kokturk.models.context_encoder import Word2VecContext
            ctx_enc = Word2VecContext(
                vocab_size=info["ctx_vocab_size"],
                embed_dim=info["ctx_embed_dim"],
                gru_hidden_dim=info["ctx_gru_hidden"],
            )
        elif ct == "sentence_bigru":
            from kokturk.models.context_encoder import SentenceBiGRUContext
            ctx_enc = SentenceBiGRUContext(
                vocab_size=info["ctx_vocab_size"],
                embed_dim=info["ctx_embed_dim"],
                hidden_dim=info["ctx_gru_hidden"],
            )
        elif ct == "berturk":
            # For BERTurk context: create a dummy encoder that outputs zeros.
            # We cannot load full BERTurk on CPU-only nodes, and we're using zero
            # context anyway. So we create a minimal stand-in.
            ctx_enc = _DummyContextEncoder(info["ctx_context_dim"])
        else:
            from kokturk.models.context_encoder import Word2VecContext
            ctx_enc = Word2VecContext(
                vocab_size=5982, embed_dim=128, gru_hidden_dim=64,
            )

        from kokturk.models.contextual_dual_head import ContextualDualHeadAtomizer
        return ContextualDualHeadAtomizer(
            context_encoder=ctx_enc,
            char_vocab_size=info["char_vocab_size"],
            tag_vocab_size=info["tag_vocab_size"],
            root_vocab_size=info["root_vocab_size"],
            embed_dim=info["embed_dim"],
            hidden_dim=info["hidden_dim"],
            num_layers=info["num_layers"],
            max_decode_len=MAX_TAG_LEN,
        )

    raise ValueError(f"Unknown model class: {mc}")


class _DummyContextEncoder(torch.nn.Module):
    """Minimal context encoder that returns zeros (for BERTurk eval without BERTurk weights)."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self._output_dim = output_dim
        # Need a parameter so .to(device) works; never used in output.
        self._dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, *args, **kwargs) -> torch.Tensor:
        # Determine batch size from first arg
        if args:
            first = args[0]
            if isinstance(first, torch.Tensor):
                bs = first.size(0)
            elif isinstance(first, (list, tuple)):
                bs = len(first)
            else:
                bs = 1
        else:
            bs = 1
        return torch.zeros(bs, self._output_dim, device=self._dummy.device)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def extract_ids(t: torch.Tensor) -> list[int]:
    """Strip PAD(0), SOS(1), EOS(2) from a tag-id tensor."""
    return [x for x in t.tolist() if x not in (PAD_IDX, SOS_IDX, EOS_IDX)]


def tag_f1(pred: list[int], gold: list[int]) -> float:
    """Token-level tag F1 using Counter overlap."""
    if not pred and not gold:
        return 1.0
    pc, gc = Counter(pred), Counter(gold)
    tp = sum(min(c, pc.get(t, 0)) for t, c in gc.items())
    p = tp / max(len(pred), 1)
    r = tp / max(len(gold), 1)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ---------------------------------------------------------------------------
# Single-model evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_one(
    ckpt_path: str,
    all_chars: torch.Tensor,
    all_tags: torch.Tensor,
    all_roots: torch.Tensor,
    model_name: str,
) -> dict:
    """Evaluate a single checkpoint on the pre-encoded test data."""
    full = PROJECT / ckpt_path
    if not full.exists():
        return {
            "name": model_name,
            "path": ckpt_path,
            "class": "MISSING",
            "em": -1,
            "root_acc": -1,
            "tag_f1": -1,
            "n_params": 0,
            "epoch": -1,
            "error": f"File not found: {full}",
        }

    ckpt = torch.load(full, map_location="cpu", weights_only=False)

    # Handle both checkpoint formats
    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
        epoch = ckpt.get("epoch", -1)
    elif "model" in ckpt:
        sd = ckpt["model"]
        epoch = ckpt.get("epoch", -1)
    else:
        # The checkpoint IS the state dict
        sd = ckpt
        epoch = -1

    # Detect architecture — prefer our mapping, fall back to auto-detect
    mc_hint = ARCH_MAP.get(model_name)
    mc_auto = detect_model_class(sd)
    mc = mc_hint if mc_hint else mc_auto

    # If hint disagrees with auto-detection, trust auto-detection
    # (it's based on actual weight keys)
    if mc_hint and mc_hint != mc_auto:
        print(
            f"  WARN: arch hint={mc_hint} but auto-detect={mc_auto}, using auto",
            file=sys.stderr,
        )
        mc = mc_auto

    info = infer_sizes(sd, mc)
    model = build_model(mc, info)

    # Load weights (strict=False to handle optional contrastive head, etc.)
    model.load_state_dict(sd, strict=False)
    model.eval()

    n = all_chars.size(0)
    em_count = 0
    root_ok = 0
    f1_sum = 0.0

    for s in range(0, n, BATCH_SIZE):
        e = min(s + BATCH_SIZE, n)
        cb = all_chars[s:e]
        tb = all_tags[s:e]
        rb = all_roots[s:e]
        bs = cb.size(0)

        if mc == "seq2seq":
            # Greedy decode: no target, no teacher forcing
            logits = model(cb, target_tags=None, teacher_forcing_ratio=0.0)
            preds = logits.argmax(dim=-1)  # (B, max_decode_len)
            for i in range(bs):
                pi = extract_ids(preds[i])
                gi = extract_ids(tb[i])
                if pi == gi:
                    em_count += 1
                # For seq2seq: first predicted token is root
                pr = pi[0] if pi else -1
                gr = gi[0] if gi else -2
                if pr == gr:
                    root_ok += 1
                # Tag F1 on suffix tags (everything after root)
                f1_sum += tag_f1(pi[1:], gi[1:])

        elif mc == "dual_head":
            rl, tl = model(
                cb, tag_ids=None, gold_root=None,
                use_gold_root=False, teacher_forcing_ratio=0.0,
            )
            rp = rl.argmax(dim=-1)  # (B,)
            tp = tl.argmax(dim=-1)  # (B, decode_len)
            for i in range(bs):
                rm = rp[i].item() == rb[i].item()
                if rm:
                    root_ok += 1
                pi = extract_ids(tp[i])
                gi_full = extract_ids(tb[i])
                # For dual_head: tag_ids layout is [SOS, root, +TAG1, ..., EOS]
                # The decoder predicts starting at position 2, so gold tags
                # for comparison are gi_full[1:] (skip the root token)
                gi = gi_full[1:] if len(gi_full) > 1 else []
                if rm and pi == gi:
                    em_count += 1
                f1_sum += tag_f1(pi, gi)

        elif mc == "contextual_dual_head":
            # Build zero context input based on context encoder type
            ct = type(model.context_encoder).__name__
            if ct == "Word2VecContext":
                ci = torch.zeros(bs, 4, dtype=torch.long)
            elif ct == "SentenceBiGRUContext":
                ci = (
                    torch.zeros(bs, 64, dtype=torch.long),
                    torch.zeros(bs, dtype=torch.long),
                )
            elif ct == "_DummyContextEncoder":
                # Our BERTurk stand-in — just pass a tensor
                ci = torch.zeros(bs, 1, dtype=torch.long)
            elif ct == "BERTurkContext":
                ci = (["dummy"] * bs, [0] * bs)
            else:
                ci = torch.zeros(bs, 4, dtype=torch.long)

            rl, tl = model(
                cb, ci, tag_ids=None, gold_root=None,
                use_gold_root=False, teacher_forcing_ratio=0.0,
            )
            rp = rl.argmax(dim=-1)
            tp = tl.argmax(dim=-1)
            for i in range(bs):
                rm = rp[i].item() == rb[i].item()
                if rm:
                    root_ok += 1
                pi = extract_ids(tp[i])
                gi_full = extract_ids(tb[i])
                gi = gi_full[1:] if len(gi_full) > 1 else []
                if rm and pi == gi:
                    em_count += 1
                f1_sum += tag_f1(pi, gi)

    n_params = sum(p.numel() for p in model.parameters())
    del model

    return {
        "name": model_name,
        "path": ckpt_path,
        "class": mc,
        "em": em_count / n,
        "root_acc": root_ok / n,
        "tag_f1": f1_sum / n,
        "n_params": n_params,
        "epoch": epoch,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()

    # Load vocabularies
    print("Loading vocabularies ...", file=sys.stderr)
    char_vocab = Vocab.load(CHAR_VOCAB_PATH)
    tag_vocab = Vocab.load(TAG_VOCAB_PATH)

    with open(ROOT_VOCAB_PATH, encoding="utf-8") as f:
        root_vocab_list = json.load(f)
    if isinstance(root_vocab_list, list):
        root_map: dict[str, int] = {r: i for i, r in enumerate(root_vocab_list)}
    else:
        root_map = root_vocab_list
    unk_root = root_map.get("<UNK_ROOT>", 1)

    # Load and encode the full test set
    print(f"Loading test data from {TEST_PATH} ...", file=sys.stderr)
    char_list: list[list[int]] = []
    tag_list: list[list[int]] = []
    root_list: list[int] = []

    with open(TEST_PATH, encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if MAX_TEST_SAMPLES > 0 and line_idx >= MAX_TEST_SAMPLES:
                break
            rec = json.loads(line)
            surface = rec["surface"]
            label = rec["label"]

            cids = [char_vocab.encode(c) for c in surface] + [EOS_IDX]
            cids = (cids[:MAX_CHAR_LEN] + [PAD_IDX] * MAX_CHAR_LEN)[:MAX_CHAR_LEN]

            parts = label.split()
            tids = [SOS_IDX] + [tag_vocab.encode(p) for p in parts] + [EOS_IDX]
            tids = (tids[:MAX_TAG_LEN] + [PAD_IDX] * MAX_TAG_LEN)[:MAX_TAG_LEN]

            char_list.append(cids)
            tag_list.append(tids)
            root_list.append(
                root_map.get(parts[0], unk_root) if parts else unk_root
            )

    all_chars = torch.tensor(char_list, dtype=torch.long)
    all_tags = torch.tensor(tag_list, dtype=torch.long)
    all_roots = torch.tensor(root_list, dtype=torch.long)
    n_test = all_chars.size(0)
    print(
        f"Test set: {n_test} samples (char_vocab={len(char_vocab)}, "
        f"tag_vocab={len(tag_vocab)}, roots={len(root_map)})",
        file=sys.stderr,
    )

    # Evaluate each checkpoint
    results: list[dict] = []
    for i, cp in enumerate(CHECKPOINT_PATHS):
        model_name = cp.replace("models/", "").replace("/best_model.pt", "")
        print(
            f"\n[{i + 1:2d}/{len(CHECKPOINT_PATHS)}] {model_name} ...",
            file=sys.stderr,
            flush=True,
        )
        try:
            r = evaluate_one(cp, all_chars, all_tags, all_roots, model_name)
            results.append(r)
            if r["em"] >= 0:
                print(
                    f"  EM={r['em']:.4f}  Root={r['root_acc']:.4f}  "
                    f"TagF1={r['tag_f1']:.4f}  Params={r['n_params']:,}  "
                    f"Epoch={r['epoch']}",
                    file=sys.stderr,
                )
            else:
                print(f"  SKIP: {r.get('error', 'unknown')}", file=sys.stderr)
        except Exception:
            tb = traceback.format_exc()
            print(f"  FAIL:\n{tb}", file=sys.stderr)
            results.append({
                "name": model_name,
                "path": cp,
                "class": "ERROR",
                "em": -1,
                "root_acc": -1,
                "tag_f1": -1,
                "n_params": 0,
                "epoch": -1,
                "error": tb.strip().split("\n")[-1],
            })

    # Write results TSV
    OUTPUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TSV, "w", encoding="utf-8") as f:
        f.write("model_name\tarchitecture\ttest_EM\troot_acc\ttag_F1\tparams\tepoch\n")
        for r in results:
            em_s = f"{r['em']:.4f}" if r["em"] >= 0 else "ERROR"
            ra_s = f"{r['root_acc']:.4f}" if r["root_acc"] >= 0 else "ERROR"
            tf_s = f"{r['tag_f1']:.4f}" if r["tag_f1"] >= 0 else "ERROR"
            pa_s = f"{r['n_params']:,}" if r["n_params"] > 0 else "N/A"
            ep_s = str(r["epoch"]) if r["epoch"] >= 0 else "N/A"
            f.write(
                f"{r['name']}\t{r['class']}\t{em_s}\t{ra_s}\t{tf_s}\t{pa_s}\t{ep_s}\n"
            )

    elapsed = time.time() - t0
    print(
        f"\n{'=' * 60}\nDone. {len(results)} models evaluated on {n_test} "
        f"test samples in {elapsed:.0f}s.\nResults: {OUTPUT_TSV}\n{'=' * 60}",
        file=sys.stderr,
    )

    # Also print the TSV to stdout for easy viewing in logs
    with open(OUTPUT_TSV, encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()
