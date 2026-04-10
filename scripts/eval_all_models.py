#!/usr/bin/env python3
"""Evaluate all 25 trained model checkpoints on the test set.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/eval_all_models.py [START_IDX] [END_IDX]

If START_IDX and END_IDX are given, only evaluate checkpoints[START_IDX:END_IDX].
Results are appended to eval_results.tsv.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from collections import Counter
from pathlib import Path

import torch

from train.datasets import Vocab, PAD_IDX, SOS_IDX, EOS_IDX

# ---------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parent.parent
TEST_PATH = PROJECT / "data" / "splits" / "test.jsonl"
CHAR_VOCAB_PATH = PROJECT / "models" / "vocabs" / "char_vocab.json"
TAG_VOCAB_PATH = PROJECT / "models" / "vocabs" / "tag_vocab.json"
ROOT_VOCAB_PATH = PROJECT / "models" / "vocabs" / "root_vocab.json"
RESULTS_FILE = PROJECT / "eval_results.tsv"

MAX_CHAR_LEN = 64
MAX_TAG_LEN = 15
BATCH_SIZE = 512
MAX_TEST_SAMPLES = int(os.environ.get("MAX_TEST_SAMPLES", "8140"))

CHECKPOINT_PATHS = [
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


def detect_model_class(state_dict: dict) -> str:
    has_context = any("context_encoder" in k or "ctx_proj" in k for k in state_dict)
    has_root_head = any("root_head" in k for k in state_dict)
    if has_context:
        return "contextual_dual_head"
    elif has_root_head:
        return "dual_head"
    return "seq2seq"


def detect_context_type(state_dict: dict) -> str:
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
    info = {}
    info["char_vocab_size"] = state_dict["encoder.char_embed.weight"].shape[0]
    info["embed_dim"] = state_dict["encoder.char_embed.weight"].shape[1]
    info["hidden_dim"] = state_dict["encoder.gru.weight_hh_l0"].shape[1]
    n = 0
    while f"encoder.gru.weight_hh_l{n}" in state_dict:
        n += 1
    info["num_layers"] = n
    if mc == "seq2seq":
        info["tag_vocab_size"] = state_dict["decoder.output_proj.weight"].shape[0]
    else:
        info["tag_vocab_size"] = state_dict["tag_decoder.output_proj.weight"].shape[0]
        info["root_vocab_size"] = state_dict["root_head.fc2.weight"].shape[0]
    if mc == "contextual_dual_head":
        ct = detect_context_type(state_dict)
        info["context_type"] = ct
        if ct in ("word2vec", "sentence_bigru"):
            info["ctx_vocab_size"] = state_dict["context_encoder.word_embed.weight"].shape[0]
            info["ctx_embed_dim"] = state_dict["context_encoder.word_embed.weight"].shape[1]
            info["ctx_gru_hidden"] = state_dict["context_encoder.gru.weight_hh_l0"].shape[1]
        elif ct == "berturk":
            info["ctx_context_dim"] = state_dict["context_encoder.proj.weight"].shape[0]
    return info


def build_model(mc: str, info: dict):
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
    elif mc == "dual_head":
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
    elif mc == "contextual_dual_head":
        from kokturk.models.contextual_dual_head import ContextualDualHeadAtomizer
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
            try:
                from transformers import AutoModel, AutoTokenizer
                bp = str(PROJECT / "models" / "berturk")
                bm = AutoModel.from_pretrained(bp)
                for p in bm.parameters():
                    p.requires_grad = False
                tok = AutoTokenizer.from_pretrained(bp)
                from kokturk.models.context_encoder import BERTurkContext
                ctx_enc = BERTurkContext(
                    context_dim=info["ctx_context_dim"],
                    bert_model=bm, tokenizer=tok,
                )
            except Exception as e:
                raise RuntimeError(f"Cannot load BERTurk: {e}") from e
        else:
            from kokturk.models.context_encoder import Word2VecContext
            ctx_enc = Word2VecContext(vocab_size=5982, embed_dim=128, gru_hidden_dim=64)
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
    raise ValueError(mc)


def extract_ids(t: torch.Tensor) -> list[int]:
    return [x for x in t.tolist() if x not in (PAD_IDX, SOS_IDX, EOS_IDX)]


def tag_f1(pred: list[int], gold: list[int]) -> float:
    pc, gc = Counter(pred), Counter(gold)
    tp = sum(min(c, pc.get(t, 0)) for t, c in gc.items())
    p = tp / max(len(pred), 1)
    r = tp / max(len(gold), 1)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


@torch.no_grad()
def evaluate_one(ckpt_path: str, all_chars, all_tags, all_roots) -> dict:
    full = PROJECT / ckpt_path
    ckpt = torch.load(full, map_location="cpu", weights_only=False)

    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
        stored_em = ckpt.get("best_em")
    else:
        sd = ckpt["model"]
        stored_em = None

    mc = detect_model_class(sd)
    info = infer_sizes(sd, mc)
    model = build_model(mc, info)
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
            logits = model(cb, target_tags=None, teacher_forcing_ratio=0.0)
            preds = logits.argmax(dim=-1)
            for i in range(bs):
                pi = extract_ids(preds[i])
                gi = extract_ids(tb[i])
                if pi == gi:
                    em_count += 1
                pr = pi[0] if pi else -1
                gr = gi[0] if gi else -2
                if pr == gr:
                    root_ok += 1
                f1_sum += tag_f1(pi[1:], gi[1:])

        elif mc == "dual_head":
            rl, tl = model(cb, tag_ids=None, gold_root=None,
                           use_gold_root=False, teacher_forcing_ratio=0.0)
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

        elif mc == "contextual_dual_head":
            ct = type(model.context_encoder).__name__
            if ct == "Word2VecContext":
                ci = torch.zeros(bs, 4, dtype=torch.long)
            elif ct == "SentenceBiGRUContext":
                ci = (torch.zeros(bs, 64, dtype=torch.long),
                      torch.zeros(bs, dtype=torch.long))
            elif ct == "BERTurkContext":
                ci = (["a"] * bs, [0] * bs)
            else:
                ci = torch.zeros(bs, 4, dtype=torch.long)
            rl, tl = model(cb, ci, tag_ids=None, gold_root=None,
                           use_gold_root=False, teacher_forcing_ratio=0.0)
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

    nparams = sum(p.numel() for p in model.parameters())
    del model
    return {
        "path": ckpt_path,
        "class": mc,
        "em": em_count / n,
        "root_acc": root_ok / n,
        "tag_f1": f1_sum / n,
        "stored_em": stored_em,
        "n_params": nparams,
    }


def main():
    start_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end_idx = int(sys.argv[2]) if len(sys.argv) > 2 else len(CHECKPOINT_PATHS)

    char_vocab = Vocab.load(CHAR_VOCAB_PATH)
    tag_vocab = Vocab.load(TAG_VOCAB_PATH)
    with open(ROOT_VOCAB_PATH) as f:
        root_vocab_list = json.load(f)
    root_map = {r: i for i, r in enumerate(root_vocab_list)}
    unk_root = root_map.get("<UNK_ROOT>", 1)

    # Load + encode test data
    char_list, tag_list, root_list = [], [], []
    with open(TEST_PATH) as f:
        for line_idx, line in enumerate(f):
            if line_idx >= MAX_TEST_SAMPLES:
                break
            rec = json.loads(line)
            surface, label = rec["surface"], rec["label"]
            cids = [char_vocab.encode(c) for c in surface] + [EOS_IDX]
            cids = (cids[:MAX_CHAR_LEN] + [PAD_IDX] * MAX_CHAR_LEN)[:MAX_CHAR_LEN]
            parts = label.split()
            tids = [SOS_IDX] + [tag_vocab.encode(p) for p in parts] + [EOS_IDX]
            tids = (tids[:MAX_TAG_LEN] + [PAD_IDX] * MAX_TAG_LEN)[:MAX_TAG_LEN]
            char_list.append(cids)
            tag_list.append(tids)
            root_list.append(root_map.get(parts[0], unk_root) if parts else unk_root)

    all_chars = torch.tensor(char_list, dtype=torch.long)
    all_tags = torch.tensor(tag_list, dtype=torch.long)
    all_roots = torch.tensor(root_list, dtype=torch.long)
    print(f"Test: {all_chars.size(0)} samples, evaluating [{start_idx}:{end_idx}]", file=sys.stderr)

    results = []
    subset = CHECKPOINT_PATHS[start_idx:end_idx]
    for i, cp in enumerate(subset):
        short = cp.replace("models/", "").replace("/best_model.pt", "")
        print(f"[{start_idx+i+1:2d}/{len(CHECKPOINT_PATHS)}] {short} ... ", end="", file=sys.stderr, flush=True)
        try:
            r = evaluate_one(cp, all_chars, all_tags, all_roots)
            results.append(r)
            print(f"EM={r['em']:.4f}", file=sys.stderr)
        except Exception as e:
            print(f"FAIL: {e}", file=sys.stderr)
            results.append({"path": cp, "class": "ERROR", "em": -1,
                            "root_acc": -1, "tag_f1": -1, "stored_em": None,
                            "n_params": 0, "error": str(e)})

    # Append to results file
    write_header = not RESULTS_FILE.exists() or start_idx == 0
    with open(RESULTS_FILE, "a" if start_idx > 0 else "w") as f:
        if write_header:
            f.write("Model\tClass\tEM\tRoot_Acc\tTag_F1\tStored_EM\tParams\n")
        for r in results:
            short = r["path"].replace("models/", "").replace("/best_model.pt", "")
            se = f"{r['stored_em']:.4f}" if r.get("stored_em") is not None else "N/A"
            em = f"{r['em']:.4f}" if r["em"] >= 0 else "ERROR"
            ra = f"{r['root_acc']:.4f}" if r["root_acc"] >= 0 else "ERROR"
            tf = f"{r['tag_f1']:.4f}" if r["tag_f1"] >= 0 else "ERROR"
            pa = f"{r['n_params']:,}" if r["n_params"] > 0 else "N/A"
            f.write(f"{short}\t{r['class']}\t{em}\t{ra}\t{tf}\t{se}\t{pa}\n")

    print(f"Results written to {RESULTS_FILE}", file=sys.stderr)


if __name__ == "__main__":
    main()
