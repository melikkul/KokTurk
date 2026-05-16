"""Evaluate a trained DualHead checkpoint on the test split.

Loads the DualHeadAtomizer from a checkpoint, runs greedy decoding over every
token in the test JSONL, and writes exact-match metrics + throughput to JSON.

Always runs on CPU — this is a reproducible evaluation script, not a training
run.  Use ``torch.device("cpu")`` explicitly.

Usage::

    python scripts/eval_dualhead.py \\
        --ckpt   models/dualhead_v2/best_model.pt \\
        --test   data/splits/test.jsonl \\
        --char-vocab models/vocabs/char_vocab.json \\
        --tag-vocab  models/vocabs/tag_vocab.json \\
        --output models/dualhead_v2/eval_results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_vocab(path: Path) -> dict[str, int]:
    """Load vocab JSON → {token: index} dict.

    Handles both list format (token at position = index) and dict format.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return {token: idx for idx, token in enumerate(raw)}
    return raw


def _vocab_inv(vocab: dict[str, int]) -> list[str]:
    """Invert vocab dict to list[str] indexed by integer id."""
    inv: list[str] = [""] * len(vocab)
    for token, idx in vocab.items():
        if 0 <= idx < len(inv):
            inv[idx] = token
    return inv


def _load_test_records(test_path: Path) -> list[dict]:
    """Read JSONL records from test file."""
    records: list[dict] = []
    for line in test_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def evaluate(
    ckpt_path: Path,
    test_path: Path,
    char_vocab_path: Path,
    tag_vocab_path: Path,
    output_path: Path,
) -> dict:
    """Run evaluation and write results.

    Returns the results dict (also written to output_path).

    Raises:
        SystemExit: If required dependencies (torch, aksu) are missing.
    """
    try:
        import torch
    except ImportError as exc:
        logger.error("torch not installed: %s", exc)
        raise SystemExit(1) from exc

    try:
        from aksu.benchmark.em import em_string as compute_em_string
        from aksu.kokturk.models.dual_head import DualHeadAtomizer
    except ImportError as exc:
        logger.error("aksu package not found: %s", exc)
        raise SystemExit(1) from exc

    device = torch.device("cpu")

    # -----------------------------------------------------------------------
    # Load checkpoint
    # -----------------------------------------------------------------------
    logger.info("Loading checkpoint: %s", ckpt_path)
    import pathlib
    torch.serialization.add_safe_globals([pathlib.PosixPath])
    state = torch.load(str(ckpt_path), map_location=device, weights_only=True)

    # -----------------------------------------------------------------------
    # Load vocabularies (needed before model construction for vocab sizes)
    # -----------------------------------------------------------------------
    char_vocab: dict[str, int] = _load_vocab(char_vocab_path)
    tag_vocab: dict[str, int] = _load_vocab(tag_vocab_path)
    char_vocab_inv = _vocab_inv(char_vocab)
    tag_vocab_inv = _vocab_inv(tag_vocab)
    unk_char_idx: int = char_vocab.get("<unk>", 0)

    # -----------------------------------------------------------------------
    # Reconstruct model — support both checkpoint formats:
    #   new: {"model_config": {...}, "model_state_dict": {...}}
    #   old (training script): {"args": {...}, "model": state_dict, ...}
    # -----------------------------------------------------------------------
    if "model_config" in state:
        model_cfg: dict = state["model_config"]
        state_dict = state["model_state_dict"]
    else:
        # Old training-script format: reconstruct config from saved args + vocab sizes
        args = state["args"]
        root_vocab_path_from_args = args.get("root_vocab", str(tag_vocab_path.parent / "root_vocab.json"))
        root_vocab_for_cfg = _load_vocab(Path(str(root_vocab_path_from_args)))
        model_cfg = {
            "char_vocab_size": len(char_vocab),
            "tag_vocab_size": len(tag_vocab),
            "root_vocab_size": len(root_vocab_for_cfg),
            "embed_dim": int(args.get("embed_dim", 64)),
            "hidden_dim": int(args.get("hidden_dim", 128)),
            "num_layers": int(args.get("num_layers", 2)),
            "dropout": float(args.get("dropout", 0.3)),
            "root_head_type": str(args.get("root_head_type", "mlp")),
            "variational_dropout": float(args.get("variational_dropout", 0.0)),
            "weight_dropout": float(args.get("weight_dropout", 0.0)),
        }
        state_dict = state["model"]

    model = DualHeadAtomizer(**model_cfg)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    logger.info("Model loaded — %d parameters", sum(p.numel() for p in model.parameters()))

    # -----------------------------------------------------------------------
    # Load test records
    # -----------------------------------------------------------------------
    records = _load_test_records(test_path)
    logger.info("Test records: %d", len(records))

    # -----------------------------------------------------------------------
    # Inference helpers
    # -----------------------------------------------------------------------

    def _encode_word(word: str) -> torch.Tensor:
        """Encode a single word as a (1, L_char) character-index tensor."""
        indices = [char_vocab.get(ch, unk_char_idx) for ch in word]
        return torch.tensor([indices], dtype=torch.long, device=device)

    def _greedy(word: str) -> str:
        """Return canonical parse string for one word via greedy decoding."""
        chars = _encode_word(word)
        with torch.no_grad():
            predictions = model.greedy_decode(
                chars,
                root_vocab_inv=None,
                tag_vocab_inv=tag_vocab_inv,
            )
        return predictions[0] if predictions else ""

    # -----------------------------------------------------------------------
    # Extract root_vocab_inv from checkpoint or args path
    # -----------------------------------------------------------------------
    root_vocab_inv: list[str] | None = None
    if "root_vocab" in state:
        root_vocab_raw: dict[str, int] = state["root_vocab"]
        root_vocab_inv = _vocab_inv(root_vocab_raw)
    elif "args" in state:
        args_rv_path = state["args"].get("root_vocab")
        if args_rv_path and args_rv_path != "None":
            try:
                root_vocab_inv = _vocab_inv(_load_vocab(Path(str(args_rv_path))))
            except FileNotFoundError:
                pass

    # Re-define with root_vocab_inv if available
    def _greedy_with_roots(word: str) -> str:
        chars = _encode_word(word)
        with torch.no_grad():
            predictions = model.greedy_decode(
                chars,
                root_vocab_inv=root_vocab_inv,
                tag_vocab_inv=tag_vocab_inv,
            )
        return predictions[0] if predictions else ""

    _decode = _greedy_with_roots

    # -----------------------------------------------------------------------
    # Evaluation loop
    # -----------------------------------------------------------------------
    pred_strings: list[str] = []
    gold_strings: list[str] = []
    pred_tag_seqs: list[list[int]] = []
    gold_tag_seqs: list[list[int]] = []

    n_tokens = 0
    t_start = time.perf_counter()

    for rec in records:
        # Records may be sentence-level ({"sentence": ..., "tokens": [...]}) or
        # token-level ({"surface": ..., "tags": ...}).
        if "tokens" in rec:
            token_list = rec["tokens"]
        elif "surface" in rec:
            token_list = [rec]
        else:
            continue

        for tok in token_list:
            surface: str = tok.get("surface", tok.get("word", ""))
            gold_parse: str = tok.get("canonical", tok.get("parse", tok.get("tags", "")))
            if not surface:
                continue

            pred_parse = _decode(surface)
            pred_strings.append(pred_parse)
            gold_strings.append(gold_parse)
            n_tokens += 1

    wall = time.perf_counter() - t_start
    throughput = round(n_tokens / wall, 1) if wall > 0 else 0.0

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------

    # em_string: cross-system canonical string equality
    em_str = compute_em_string(pred_strings, gold_strings)

    # em_argmax: treat each canonical parse as a category label and compare
    # indices (legacy; only meaningful for within-system comparison)
    all_labels = sorted(set(gold_strings) | set(pred_strings))
    label2idx = {lbl: i for i, lbl in enumerate(all_labels)}
    pred_indices = [label2idx[p] for p in pred_strings]
    gold_indices = [label2idx[g] for g in gold_strings]
    em_argmax = (
        sum(p == g for p, g in zip(pred_indices, gold_indices, strict=False)) / len(gold_indices)
        if gold_indices
        else 0.0
    )

    results: dict = {
        "em_argmax": round(em_argmax, 6),
        "em_string": round(em_str, 6),
        "throughput_tok_per_sec": throughput,
        "n_tokens": n_tokens,
        "checkpoint_path": str(ckpt_path),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    logger.info(
        "em_argmax=%.4f  em_string=%.4f  throughput=%.1f tok/s  n=%d",
        em_argmax, em_str, throughput, n_tokens,
    )
    logger.info("Results written to %s", output_path)
    print(json.dumps(results, indent=2))

    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ap = argparse.ArgumentParser(
        description="Evaluate a DualHead checkpoint on the test split (CPU-only)"
    )
    ap.add_argument(
        "--ckpt",
        default="models/dualhead_v2/best_model.pt",
        help="Path to the DualHeadAtomizer checkpoint (default: models/dualhead_v2/best_model.pt)",
    )
    ap.add_argument(
        "--test",
        default="data/splits/test.jsonl",
        help="Path to the test JSONL file (default: data/splits/test.jsonl)",
    )
    ap.add_argument(
        "--char-vocab",
        default="models/vocabs/char_vocab.json",
        dest="char_vocab",
        help="Path to char_vocab.json (default: models/vocabs/char_vocab.json)",
    )
    ap.add_argument(
        "--tag-vocab",
        default="models/vocabs/tag_vocab.json",
        dest="tag_vocab",
        help="Path to tag_vocab.json (default: models/vocabs/tag_vocab.json)",
    )
    ap.add_argument(
        "--output",
        default="models/dualhead_v2/eval_results.json",
        help="Path for the JSON output file (default: models/dualhead_v2/eval_results.json)",
    )
    args = ap.parse_args()

    evaluate(
        ckpt_path=Path(args.ckpt),
        test_path=Path(args.test),
        char_vocab_path=Path(args.char_vocab),
        tag_vocab_path=Path(args.tag_vocab),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
