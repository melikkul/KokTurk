"""Benchmark BERTurk (C2), reranker/disambiguator (C3), and DualHead (C4) inference
throughput on CPU.

Warmup N iterations (cache warm, steady-state JIT), then measure M iterations
with per-sentence/per-token latency recording.  Output JSON is written to
--output and also printed to stdout.

Usage::

    python scripts/benchmark_inference.py \
        --components berturk reranker dualhead \
        --corpus data/splits/val.jsonl \
        --warmup 50 \
        --measure 500 \
        --output audit/benchmark_results/inference_throughput.json

If a component cannot be loaded (ImportError, FileNotFoundError, missing
checkpoint) its metric keys are set to null and a ``_skipped_reason`` key
records why.  The script never crashes due to a missing component.
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_cpu_model() -> str:
    """Return the CPU model name from /proc/cpuinfo, or platform fallback."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except FileNotFoundError:
        pass
    return platform.processor() or platform.machine()


def _load_corpus(corpus_path: Path, n_needed: int) -> list[str]:
    """Load sentences from a JSONL file, cycling if shorter than n_needed."""
    if not corpus_path.exists():
        print(f"[warn] Corpus not found at {corpus_path}; using synthetic sentences.")
        sentences = [
            f"Çocuklar evlerinden çıktı ve {i} numaralı arabaya bindiler."
            for i in range(n_needed)
        ]
        return sentences

    raw_lines = [
        json.loads(line)
        for line in corpus_path.read_text().splitlines()
        if line.strip()
    ]
    field = next(
        (k for k in ("sentence", "text", "surface") if raw_lines and k in raw_lines[0]),
        None,
    )
    if field is None:
        raise ValueError(
            f"Cannot find text field in {corpus_path}; keys: {list(raw_lines[0].keys())}"
        )
    sentences = [row[field] for row in raw_lines]

    if len(sentences) < n_needed:
        factor = n_needed // len(sentences) + 1
        sentences = (sentences * factor)[:n_needed]

    return sentences[:n_needed]


def _measure_latencies(
    fn: Any,
    items: list[Any],
    n_warmup: int,
    n_measure: int,
) -> tuple[list[float], int]:
    """Run fn(item) for warmup then measure iterations.

    Returns:
        latencies: per-call wall-clock times in seconds (length n_measure)
        total_tokens: sum of token counts returned by fn (0 if fn returns None)
    """
    # Warmup
    for item in items[:n_warmup]:
        fn(item)

    gc.collect()
    latencies: list[float] = []
    total_tokens = 0

    for item in items[n_warmup : n_warmup + n_measure]:
        t0 = time.perf_counter()
        result = fn(item)
        latencies.append(time.perf_counter() - t0)
        if isinstance(result, int):
            total_tokens += result
        elif result is not None:
            try:
                total_tokens += len(result)
            except TypeError:
                total_tokens += 1

    return latencies, total_tokens


def _latency_stats(
    latencies: list[float],
) -> tuple[float, float, float]:
    """Return (p50_ms, p95_ms, p99_ms) from a list of second-valued latencies."""
    p50 = statistics.median(latencies) * 1000
    p95 = statistics.quantiles(latencies, n=20)[-1] * 1000
    p99 = statistics.quantiles(latencies, n=100)[-1] * 1000
    return p50, p95, p99


def _peak_rss_mb() -> float:
    """Return current process RSS in MB via psutil."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        return -1.0


# ---------------------------------------------------------------------------
# Component benchmarks
# ---------------------------------------------------------------------------

def benchmark_berturk(
    sentences: list[str],
    n_warmup: int,
    n_measure: int,
    berturk_path: str = "models/berturk",
) -> dict[str, Any]:
    """Tokenize + encode sentences through BERTurk on CPU.

    Measures sentences/second and per-sentence latency.
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        return {
            "berturk_sent_per_sec": None,
            "berturk_p50_ms": None,
            "berturk_p95_ms": None,
            "berturk_peak_rss_mb": None,
            "berturk_skipped_reason": f"ImportError: {exc}",
        }

    model_dir = Path(berturk_path)
    if not model_dir.exists():
        # Fall back to HuggingFace Hub identifier when local dir absent
        hub_id = "dbmdz/bert-base-turkish-cased"
        print(f"[info] {model_dir} not found; loading BERTurk from Hub ({hub_id}).")
        load_path: str | Path = hub_id
    else:
        load_path = model_dir

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(load_path))
        model = AutoModel.from_pretrained(str(load_path))
        model.eval()
        device = torch.device("cpu")
        model.to(device)
    except Exception as exc:  # noqa: BLE001
        return {
            "berturk_sent_per_sec": None,
            "berturk_p50_ms": None,
            "berturk_p95_ms": None,
            "berturk_peak_rss_mb": None,
            "berturk_skipped_reason": f"load failed: {exc}",
        }

    def _run(sentence: str) -> int:
        enc = tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        with torch.no_grad():
            _ = model(**enc)
        return enc["input_ids"].shape[1]

    rss_before = _peak_rss_mb()
    latencies, total_tokens = _measure_latencies(_run, sentences, n_warmup, n_measure)
    rss_after = _peak_rss_mb()

    wall = sum(latencies)
    p50, p95, p99 = _latency_stats(latencies)

    return {
        "berturk_sent_per_sec": round(n_measure / wall, 1),
        "berturk_p50_ms": round(p50, 3),
        "berturk_p95_ms": round(p95, 3),
        "berturk_p99_ms": round(p99, 3),
        "berturk_peak_rss_mb": round(max(rss_before, rss_after), 1),
    }


def benchmark_reranker(
    sentences: list[str],
    n_warmup: int,
    n_measure: int,
    ckpt_path: str = "models/v6/disambiguator/best_model.pt",
) -> dict[str, Any]:
    """Load BERTurkDisambiguator and run it on tokenized sentences.

    Measures tokens/second (each sentence contributes len(tokens) tokens).
    We benchmark the BERTurk forward pass through the disambiguator's frozen
    encoder, which dominates wall-clock time at inference.
    """
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as exc:
        return {
            "reranker_tok_per_sec": None,
            "reranker_p50_ms": None,
            "reranker_p95_ms": None,
            "reranker_peak_rss_mb": None,
            "reranker_skipped_reason": f"ImportError: {exc}",
        }

    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        return {
            "reranker_tok_per_sec": None,
            "reranker_p50_ms": None,
            "reranker_p95_ms": None,
            "reranker_peak_rss_mb": None,
            "reranker_skipped_reason": f"checkpoint not found: {ckpt}",
        }

    try:
        from aksu.kokturk.models.disambiguator import BERTurkDisambiguator
    except ImportError as exc:
        return {
            "reranker_tok_per_sec": None,
            "reranker_p50_ms": None,
            "reranker_p95_ms": None,
            "reranker_peak_rss_mb": None,
            "reranker_skipped_reason": f"ImportError aksu: {exc}",
        }

    try:
        import pathlib as _pathlib
        torch.serialization.add_safe_globals([_pathlib.PosixPath])
        device = torch.device("cpu")
        state = torch.load(str(ckpt), map_location=device, weights_only=True)
        # Checkpoints store only the lightweight reranker head weights, not BERTurk.
        # BERTurk is loaded separately from models/berturk/ (frozen, pre-trained).
        if "model_config" in state:
            model_cfg = state["model_config"]
        else:
            model_cfg = {
                "tag_vocab_size": state.get("tag_vocab_size", 512),
                "bert_path": "models/berturk",
            }
        bert_path = model_cfg.get("bert_path", "models/berturk")
        if not Path(bert_path).exists():
            bert_path = "dbmdz/bert-base-turkish-cased"
        # Initialize model (loads BERTurk from bert_path), then overlay
        # checkpoint weights with strict=False — BERTurk keys will not be in
        # the checkpoint state dict; only the reranker head keys are.
        model = BERTurkDisambiguator(**model_cfg)
        missing, unexpected = model.load_state_dict(state["model_state_dict"], strict=False)
        # Expected: many BERTurk keys missing (loaded from pre-trained). Unexpected keys → error.
        if unexpected:
            raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected[:3]}…")
        model.eval()
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(bert_path)
    except Exception as exc:  # noqa: BLE001
        return {
            "reranker_tok_per_sec": None,
            "reranker_p50_ms": None,
            "reranker_p95_ms": None,
            "reranker_peak_rss_mb": None,
            "reranker_skipped_reason": f"load failed: {exc}",
        }

    def _run(sentence: str) -> int:
        enc = tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        with torch.no_grad():
            if model.bert is not None:
                _ = model.bert(**enc)
        n_tok = enc["input_ids"].shape[1]
        return n_tok

    rss_before = _peak_rss_mb()
    latencies, total_tokens = _measure_latencies(_run, sentences, n_warmup, n_measure)
    rss_after = _peak_rss_mb()

    wall = sum(latencies)
    p50, p95, p99 = _latency_stats(latencies)

    return {
        "reranker_tok_per_sec": round(total_tokens / wall, 1),
        "reranker_p50_ms": round(p50, 3),
        "reranker_p95_ms": round(p95, 3),
        "reranker_p99_ms": round(p99, 3),
        "reranker_peak_rss_mb": round(max(rss_before, rss_after), 1),
    }


def benchmark_dualhead(
    sentences: list[str],
    n_warmup: int,
    n_measure: int,
    ckpt_candidates: list[str] | None = None,
) -> dict[str, Any]:
    """Load DualHeadAtomizer and run a greedy-decode forward pass per sentence.

    Prefers dualhead_v2 over dualhead_v1_cpu.  Measures tokens/second.
    """
    if ckpt_candidates is None:
        ckpt_candidates = [
            "models/dualhead_v2/best_model.pt",
            "models/dualhead_v1_cpu/best_model.pt",
        ]

    ckpt: Path | None = None
    for candidate in ckpt_candidates:
        p = Path(candidate)
        if p.exists():
            ckpt = p
            break

    if ckpt is None:
        return {
            "dualhead_tok_per_sec": None,
            "dualhead_p50_ms": None,
            "dualhead_p95_ms": None,
            "dualhead_peak_rss_mb": None,
            "dualhead_skipped_reason": "no checkpoint found",
        }

    try:
        import torch
    except ImportError as exc:
        return {
            "dualhead_tok_per_sec": None,
            "dualhead_p50_ms": None,
            "dualhead_p95_ms": None,
            "dualhead_peak_rss_mb": None,
            "dualhead_skipped_reason": f"ImportError: {exc}",
        }

    try:
        from aksu.kokturk.models.dual_head import DualHeadAtomizer
        from aksu.train.datasets import Vocab
    except ImportError as exc:
        return {
            "dualhead_tok_per_sec": None,
            "dualhead_p50_ms": None,
            "dualhead_p95_ms": None,
            "dualhead_peak_rss_mb": None,
            "dualhead_skipped_reason": f"ImportError aksu: {exc}",
        }

    try:
        import pathlib as _pathlib
        torch.serialization.add_safe_globals([_pathlib.PosixPath])
        device = torch.device("cpu")
        state = torch.load(str(ckpt), map_location=device, weights_only=True)

        # Load vocabs from models/vocabs/ (relative to project root)
        vocab_dir = ckpt.parent.parent / "vocabs"
        char_vocab = Vocab.load(vocab_dir / "char_vocab.json")
        tag_vocab = Vocab.load(vocab_dir / "tag_vocab.json")
        root_vocab = Vocab.load(vocab_dir / "root_vocab.json")

        # Support both checkpoint formats:
        #   new: {"model_config": {...}, "model_state_dict": {...}}
        #   old (training script): {"args": {...}, "model": state_dict, ...}
        if "model_config" in state:
            model_cfg = state["model_config"]
            state_dict = state["model_state_dict"]
        else:
            args = state["args"]
            model_cfg = {
                "char_vocab_size": len(char_vocab),
                "tag_vocab_size": len(tag_vocab),
                "root_vocab_size": len(root_vocab),
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
    except Exception as exc:  # noqa: BLE001
        return {
            "dualhead_tok_per_sec": None,
            "dualhead_p50_ms": None,
            "dualhead_p95_ms": None,
            "dualhead_peak_rss_mb": None,
            "dualhead_skipped_reason": f"load failed: {exc}",
        }

    def _encode_sentence(sentence: str) -> torch.Tensor:
        """Encode all tokens in a sentence as char-index tensors, padded."""
        words = sentence.split()
        max_len = max((len(w) for w in words), default=1)
        batch = torch.zeros(len(words), max_len + 2, dtype=torch.long)
        for i, word in enumerate(words):
            indices = [char_vocab.encode(ch) for ch in word]
            batch[i, : len(indices)] = torch.tensor(indices, dtype=torch.long)
        return batch

    def _run(sentence: str) -> int:
        chars = _encode_sentence(sentence)
        with torch.no_grad():
            _ = model.forward(chars)
        return chars.size(0)  # number of tokens (words)

    rss_before = _peak_rss_mb()
    latencies, total_tokens = _measure_latencies(_run, sentences, n_warmup, n_measure)
    rss_after = _peak_rss_mb()

    wall = sum(latencies)
    p50, p95, p99 = _latency_stats(latencies)

    return {
        "dualhead_tok_per_sec": round(total_tokens / wall, 1),
        "dualhead_p50_ms": round(p50, 3),
        "dualhead_p95_ms": round(p95, 3),
        "dualhead_p99_ms": round(p99, 3),
        "dualhead_peak_rss_mb": round(max(rss_before, rss_after), 1),
        "dualhead_checkpoint": str(ckpt),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Benchmark BERTurk / reranker / DualHead inference throughput on CPU"
    )
    ap.add_argument(
        "--components",
        nargs="+",
        choices=["berturk", "reranker", "dualhead"],
        default=["berturk", "reranker", "dualhead"],
        help="Which components to benchmark (default: all three)",
    )
    ap.add_argument(
        "--corpus",
        default="data/splits/val.jsonl",
        help="JSONL file with a 'sentence' field per line",
    )
    ap.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Number of warmup iterations before measurement (default: 50)",
    )
    ap.add_argument(
        "--measure",
        type=int,
        default=500,
        help="Number of iterations to measure (default: 500)",
    )
    ap.add_argument(
        "--output",
        default="audit/benchmark_results/inference_throughput.json",
        help="Path for the JSON output file",
    )
    ap.add_argument(
        "--include-dualhead",
        action="store_true",
        help="Enable DualHead benchmarking (skipped if no checkpoint found)",
    )
    ap.add_argument(
        "--berturk-path",
        default="models/berturk",
        help="Local path or HuggingFace Hub ID for BERTurk (default: models/berturk)",
    )
    ap.add_argument(
        "--reranker-ckpt",
        default="models/v6/disambiguator/best_model.pt",
        help="Path to the BERTurkDisambiguator checkpoint",
    )
    args = ap.parse_args()

    # If --include-dualhead was passed without 'dualhead' in --components, add it
    components: list[str] = list(args.components)
    if args.include_dualhead and "dualhead" not in components:
        components.append("dualhead")

    n_needed = args.warmup + args.measure
    sentences = _load_corpus(Path(args.corpus), n_needed)

    out: dict[str, Any] = {
        "host_cpu": _read_cpu_model(),
        "host_kernel": platform.release(),
        "n_warmup": args.warmup,
        "n_measure": args.measure,
    }

    if "berturk" in components:
        print("[bench] BERTurk …")
        result = benchmark_berturk(
            sentences, args.warmup, args.measure, berturk_path=args.berturk_path
        )
        out.update(result)
    else:
        out["berturk_sent_per_sec"] = None
        out["berturk_skipped_reason"] = "not selected"

    if "reranker" in components:
        print("[bench] Reranker/disambiguator …")
        result = benchmark_reranker(
            sentences, args.warmup, args.measure, ckpt_path=args.reranker_ckpt
        )
        out.update(result)
    else:
        out["reranker_tok_per_sec"] = None
        out["reranker_skipped_reason"] = "not selected"

    if "dualhead" in components:
        print("[bench] DualHead …")
        result = benchmark_dualhead(sentences, args.warmup, args.measure)
        out.update(result)
    else:
        out["dualhead_tok_per_sec"] = None
        out["dualhead_skipped_reason"] = "not selected (pass --include-dualhead to enable)"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
