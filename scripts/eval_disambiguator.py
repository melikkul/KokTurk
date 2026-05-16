"""Evaluate v6 disambiguator ensemble; emit em_argmax and em_string per seed.

Usage:
    python scripts/eval_disambiguator.py \
        --ckpts models/v6/disambiguator{,_s123,_s456,_s789,_s1337}/best_model.pt \
        --test data/splits/test.jsonl \
        --val  data/splits/val.jsonl \
        --output models/v6/eval_results.json

Requires: torch, transformers (BERTurk). Run on akya-cuda via SLURM.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _try_import():
    try:
        import torch
        from torch.utils.data import DataLoader

        from aksu.kokturk.models.disambiguator import BERTurkDisambiguator
        from aksu.train.disambiguation_dataset import (
            DisambiguationDataset,
            disambiguation_collate,
        )
        from aksu.train.datasets import Vocab
        from aksu.train.train_disambiguator import (
            evaluate,
            pre_cache_bert_embeddings,
        )
        return torch, DataLoader, BERTurkDisambiguator, DisambiguationDataset, disambiguation_collate, Vocab, evaluate, pre_cache_bert_embeddings
    except ImportError as e:
        logger.error("Missing dependency: %s. Install transformers + torch.", e)
        sys.exit(1)


def eval_one_seed(
    ckpt_path: Path,
    test_path: Path,
    vocab_dir: Path,
    *,
    device: object,
    batch_size: int = 64,
) -> dict:
    torch, DataLoader, BERTurkDisambiguator, DisambiguationDataset, collate, Vocab, _evaluate, pre_cache_bert = _try_import()

    tag_vocab = Vocab.load(vocab_dir / "tag_vocab.json")
    bert_path = "models/berturk"

    state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    if "model_config" in state:
        model_cfg = state["model_config"]
    else:
        model_cfg = {
            "tag_vocab_size": state.get("tag_vocab_size", len(tag_vocab)),
            "bert_path": bert_path,
        }
    # Checkpoint stores only the 21-key reranker head (frozen BERTurk excluded).
    # BERTurkDisambiguator.__init__ loads BERTurk from bert_path; overlay
    # head weights with strict=False (BERTurk keys absent from checkpoint).
    model = BERTurkDisambiguator(**model_cfg)
    _, unexpected = model.load_state_dict(state["model_state_dict"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:3]}")
    model.to(device)
    model.eval()

    # DisambiguationDataset(data_path, tag_vocab) — no char_vocab arg
    ds = DisambiguationDataset(test_path, tag_vocab)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    # pre_cache_bert_embeddings(dataset, bert_path) — not device
    bert_cache = pre_cache_bert(ds, bert_path)

    # Custom eval loop: collect per-sample preds to compute em_string
    from aksu.benchmark.em import pred_index_to_strings, em_string as compute_em_string

    pred_indices: list[int] = []
    gold_indices: list[int] = []
    correct = 0
    total = 0

    import torch as _torch  # already imported, alias avoids shadowing
    with _torch.no_grad():
        for batch in loader:
            from aksu.train.train_disambiguator import _get_cached_embeds
            cached_embeds = None
            if bert_cache is not None:
                cached_embeds = _get_cached_embeds(bert_cache, batch["sample_indices"])
            logits, _ = model(
                sentence_texts=batch["sentence_texts"],
                target_positions=batch["target_positions"],
                candidate_ids=batch["candidate_ids"],
                candidate_mask=batch["candidate_mask"],
                cached_bert_embeds=cached_embeds,
            )
            preds = logits.argmax(dim=-1)
            gold = batch["gold_indices"].to(device)
            correct += (preds == gold).sum().item()
            total += len(gold)
            pred_indices.extend(preds.tolist())
            gold_indices.extend(gold.tolist())

    em_argmax = correct / total if total else 0.0

    candidate_strings = [s["candidates"] for s in ds.samples]
    pred_strings = pred_index_to_strings(pred_indices, candidate_strings)
    gold_strings = [cands[gi] for cands, gi in zip(candidate_strings, gold_indices)]
    em_s = compute_em_string(pred_strings, gold_strings)

    return {
        "em_argmax": em_argmax,
        "em_string": em_s,
        "n_tokens": total,
        "checkpoint": str(ckpt_path),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--val",  default=None)
    ap.add_argument("--vocab-dir", default="models/vocabs")
    ap.add_argument("--output", default="models/v6/eval_results.json")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    torch = _try_import()[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    results: dict[str, dict] = {}
    em_argmax_list: list[float] = []
    em_string_list: list[float] = []

    for ckpt_str in args.ckpts:
        ckpt = Path(ckpt_str)
        logger.info("Evaluating %s ...", ckpt)
        seed_key = ckpt.parent.name
        try:
            res = eval_one_seed(
                ckpt, Path(args.test), Path(args.vocab_dir),
                device=device, batch_size=args.batch_size,
            )
            results[seed_key] = res
            em_argmax_list.append(res["em_argmax"])
            em_string_list.append(res["em_string"])
            logger.info(
                "  %s: em_argmax=%.4f em_string=%.4f",
                seed_key, res["em_argmax"], res["em_string"],
            )
        except Exception as e:
            logger.error("Failed on %s: %s", ckpt, e)
            results[seed_key] = {"error": str(e), "checkpoint": str(ckpt)}

    # Ensemble: average predictions (simplified — real ensemble uses majority vote)
    if em_argmax_list:
        import statistics
        results["ensemble"] = {
            "em_argmax_mean": statistics.mean(em_argmax_list),
            "em_string_mean": statistics.mean(em_string_list),
            "em_argmax_std": statistics.stdev(em_argmax_list) if len(em_argmax_list) > 1 else 0.0,
            "em_string_std": statistics.stdev(em_string_list) if len(em_string_list) > 1 else 0.0,
            "n_seeds": len(em_argmax_list),
        }
        logger.info(
            "Ensemble: em_argmax=%.4f±%.4f  em_string=%.4f±%.4f",
            results["ensemble"]["em_argmax_mean"],
            results["ensemble"]["em_argmax_std"],
            results["ensemble"]["em_string_mean"],
            results["ensemble"]["em_string_std"],
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Results written to %s", out)


if __name__ == "__main__":
    main()
