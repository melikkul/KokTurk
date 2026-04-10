#!/usr/bin/env python3
"""Evaluate ONE model checkpoint on the test set (or a subset).

Usage:
    PYTHONPATH=src .venv/bin/python scripts/eval_one_model.py <checkpoint_path> [max_samples]

Prints one TSV line to stdout:
    Model  Class  EM  Root_Acc  Tag_F1  Stored_EM  Params

For large test sets, consider running on a machine with sufficient CPU time.
For full 8140-token evaluation, a multi-core node is recommended.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch

from train.datasets import Vocab, PAD_IDX, SOS_IDX, EOS_IDX

PROJECT = Path(__file__).resolve().parent.parent
TEST_PATH = PROJECT / "data" / "splits" / "test.jsonl"
CHAR_VOCAB_PATH = PROJECT / "models" / "vocabs" / "char_vocab.json"
TAG_VOCAB_PATH = PROJECT / "models" / "vocabs" / "tag_vocab.json"
ROOT_VOCAB_PATH = PROJECT / "models" / "vocabs" / "root_vocab.json"

MAX_CHAR_LEN = 64
MAX_TAG_LEN = 15


def detect_model_class(sd):
    if any("context_encoder" in k or "ctx_proj" in k for k in sd):
        return "contextual_dual_head"
    # dual_head uses tag_decoder.*, seq2seq uses decoder.*
    # atomizer_v3 has root_head but uses decoder.* (legacy hybrid) -> treat as seq2seq
    if any("tag_decoder." in k for k in sd):
        return "dual_head"
    return "seq2seq"


def detect_context_type(sd):
    if any("context_encoder.bert" in k for k in sd):
        return "berturk"
    if any("context_encoder.dropout" in k for k in sd):
        return "sentence_bigru"
    if any("context_encoder.word_embed" in k for k in sd):
        return "word2vec"
    return "word2vec"


def infer_sizes(sd, mc):
    info = {}
    info["char_vocab_size"] = sd["encoder.char_embed.weight"].shape[0]
    info["embed_dim"] = sd["encoder.char_embed.weight"].shape[1]
    info["hidden_dim"] = sd["encoder.gru.weight_hh_l0"].shape[1]
    n = 0
    while f"encoder.gru.weight_hh_l{n}" in sd:
        n += 1
    info["num_layers"] = n
    if mc == "seq2seq":
        info["tag_vocab_size"] = sd["decoder.output_proj.weight"].shape[0]
    else:
        info["tag_vocab_size"] = sd["tag_decoder.output_proj.weight"].shape[0]
        info["root_vocab_size"] = sd["root_head.fc2.weight"].shape[0]
    if mc == "contextual_dual_head":
        ct = detect_context_type(sd)
        info["context_type"] = ct
        if ct in ("word2vec", "sentence_bigru"):
            info["ctx_vocab_size"] = sd["context_encoder.word_embed.weight"].shape[0]
            info["ctx_embed_dim"] = sd["context_encoder.word_embed.weight"].shape[1]
            info["ctx_gru_hidden"] = sd["context_encoder.gru.weight_hh_l0"].shape[1]
        elif ct == "berturk":
            info["ctx_context_dim"] = sd["context_encoder.proj.weight"].shape[0]
    return info


def build_model(mc, info):
    if mc == "seq2seq":
        from kokturk.models.char_gru import MorphAtomizer
        return MorphAtomizer(
            char_vocab_size=info["char_vocab_size"], tag_vocab_size=info["tag_vocab_size"],
            embed_dim=info["embed_dim"], hidden_dim=info["hidden_dim"],
            num_layers=info["num_layers"], max_decode_len=MAX_TAG_LEN)
    elif mc == "dual_head":
        from kokturk.models.dual_head import DualHeadAtomizer
        return DualHeadAtomizer(
            char_vocab_size=info["char_vocab_size"], tag_vocab_size=info["tag_vocab_size"],
            root_vocab_size=info["root_vocab_size"], embed_dim=info["embed_dim"],
            hidden_dim=info["hidden_dim"], num_layers=info["num_layers"],
            max_decode_len=MAX_TAG_LEN)
    elif mc == "contextual_dual_head":
        from kokturk.models.contextual_dual_head import ContextualDualHeadAtomizer
        ct = info.get("context_type", "word2vec")
        if ct == "word2vec":
            from kokturk.models.context_encoder import Word2VecContext
            ctx_enc = Word2VecContext(vocab_size=info["ctx_vocab_size"],
                embed_dim=info["ctx_embed_dim"], gru_hidden_dim=info["ctx_gru_hidden"])
        elif ct == "sentence_bigru":
            from kokturk.models.context_encoder import SentenceBiGRUContext
            ctx_enc = SentenceBiGRUContext(vocab_size=info["ctx_vocab_size"],
                embed_dim=info["ctx_embed_dim"], hidden_dim=info["ctx_gru_hidden"])
        elif ct == "berturk":
            from transformers import AutoModel, AutoTokenizer
            bp = str(PROJECT / "models" / "berturk")
            bm = AutoModel.from_pretrained(bp)
            for p in bm.parameters():
                p.requires_grad = False
            tok = AutoTokenizer.from_pretrained(bp)
            from kokturk.models.context_encoder import BERTurkContext
            ctx_enc = BERTurkContext(context_dim=info["ctx_context_dim"], bert_model=bm, tokenizer=tok)
        else:
            from kokturk.models.context_encoder import Word2VecContext
            ctx_enc = Word2VecContext(vocab_size=5982, embed_dim=128, gru_hidden_dim=64)
        return ContextualDualHeadAtomizer(
            context_encoder=ctx_enc,
            char_vocab_size=info["char_vocab_size"], tag_vocab_size=info["tag_vocab_size"],
            root_vocab_size=info["root_vocab_size"], embed_dim=info["embed_dim"],
            hidden_dim=info["hidden_dim"], num_layers=info["num_layers"],
            max_decode_len=MAX_TAG_LEN)
    raise ValueError(mc)


def extract_ids(t):
    return [x for x in t.tolist() if x not in (PAD_IDX, SOS_IDX, EOS_IDX)]


def tag_f1(pred, gold):
    pc, gc = Counter(pred), Counter(gold)
    tp = sum(min(c, pc.get(t, 0)) for t, c in gc.items())
    p = tp / max(len(pred), 1)
    r = tp / max(len(gold), 1)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def main():
    ckpt_rel = sys.argv[1]
    max_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 8140

    cv = Vocab.load(CHAR_VOCAB_PATH)
    tv = Vocab.load(TAG_VOCAB_PATH)
    with open(ROOT_VOCAB_PATH) as f:
        rvl = json.load(f)
    rm = {r: i for i, r in enumerate(rvl)}
    ur = rm.get("<UNK_ROOT>", 1)

    cl, tl, rl = [], [], []
    with open(TEST_PATH) as f:
        for idx, line in enumerate(f):
            if idx >= max_samples:
                break
            rec = json.loads(line)
            s = rec["surface"]
            cids = [cv.encode(c) for c in s] + [2]
            cids = (cids[:64] + [0]*64)[:64]
            parts = rec["label"].split()
            tids = [1] + [tv.encode(p) for p in parts] + [2]
            tids = (tids[:15] + [0]*15)[:15]
            cl.append(cids)
            tl.append(tids)
            rl.append(rm.get(parts[0], ur) if parts else ur)

    ac = torch.tensor(cl, dtype=torch.long)
    at = torch.tensor(tl, dtype=torch.long)
    ar = torch.tensor(rl, dtype=torch.long)
    n = ac.size(0)

    fp = PROJECT / ckpt_rel
    ckpt = torch.load(fp, map_location="cpu", weights_only=False)
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
    nparams = sum(p.numel() for p in model.parameters())

    em_count = 0
    root_ok = 0
    f1_sum = 0.0
    BS = 512

    with torch.no_grad():
        for s in range(0, n, BS):
            e = min(s + BS, n)
            cb, tb, rb = ac[s:e], at[s:e], ar[s:e]
            bs = cb.size(0)

            if mc == "seq2seq":
                logits = model(cb, target_tags=None, teacher_forcing_ratio=0.0)
                preds = logits.argmax(dim=-1)
                for i in range(bs):
                    pi = extract_ids(preds[i])
                    gi = extract_ids(tb[i])
                    if pi == gi: em_count += 1
                    pr = pi[0] if pi else -1
                    gr = gi[0] if gi else -2
                    if pr == gr: root_ok += 1
                    f1_sum += tag_f1(pi[1:], gi[1:])
            elif mc in ("dual_head", "contextual_dual_head"):
                if mc == "contextual_dual_head":
                    ctn = type(model.context_encoder).__name__
                    if ctn == "Word2VecContext":
                        ci = torch.zeros(bs, 4, dtype=torch.long)
                    elif ctn == "SentenceBiGRUContext":
                        ci = (torch.zeros(bs, 64, dtype=torch.long), torch.zeros(bs, dtype=torch.long))
                    elif ctn == "BERTurkContext":
                        ci = (["a"] * bs, [0] * bs)
                    else:
                        ci = torch.zeros(bs, 4, dtype=torch.long)
                    rlo, tlo = model(cb, ci, tag_ids=None, gold_root=None,
                                     use_gold_root=False, teacher_forcing_ratio=0.0)
                else:
                    rlo, tlo = model(cb, tag_ids=None, gold_root=None,
                                     use_gold_root=False, teacher_forcing_ratio=0.0)
                rp = rlo.argmax(dim=-1)
                tp = tlo.argmax(dim=-1)
                for i in range(bs):
                    rmatch = rp[i].item() == rb[i].item()
                    if rmatch: root_ok += 1
                    pi = extract_ids(tp[i])
                    gi_full = extract_ids(tb[i])
                    gi = gi_full[1:] if len(gi_full) > 1 else []
                    if rmatch and pi == gi: em_count += 1
                    f1_sum += tag_f1(pi, gi)

    em = em_count / n
    ra = root_ok / n
    tf = f1_sum / n
    short = ckpt_rel.replace("models/", "").replace("/best_model.pt", "")
    se = f"{stored_em:.4f}" if stored_em is not None else "N/A"
    print(f"{short}\t{mc}\t{em:.4f}\t{ra:.4f}\t{tf:.4f}\t{se}\t{nparams}")


if __name__ == "__main__":
    main()
