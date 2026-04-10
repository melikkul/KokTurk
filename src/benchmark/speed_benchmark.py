"""Inference speed and memory benchmarking.

Reports tokens-per-second, p50/p95/p99 latencies, peak memory, and
startup time. Also benchmarks optional competitor systems (Zeyrek,
Stanza, spaCy) on the same token list so that the final report lands
the project numbers on a comparable axis.
"""

from __future__ import annotations

import importlib.util
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SpeedReport:
    name: str
    device: str
    batch_size: int
    tps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    peak_mem_mb: float


@dataclass
class BenchmarkBundle:
    model_results: list[SpeedReport] = field(default_factory=list)
    competitor_results: dict[str, float] = field(default_factory=dict)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * pct / 100.0
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def benchmark_inference(
    predict_fn,
    test_tokens: list[str],
    device: str = "cpu",
    num_runs: int = 100,
    warmup_runs: int = 10,
    batch_sizes: tuple[int, ...] = (1, 32, 128),
    name: str = "model",
) -> list[SpeedReport]:
    """Time ``predict_fn`` across a batch-size sweep.

    ``predict_fn`` takes ``list[str]`` and returns ``list[str]``.
    """
    reports: list[SpeedReport] = []
    for bs in batch_sizes:
        batches = [test_tokens[i : i + bs] for i in range(0, len(test_tokens), bs)] or [[]]
        # Warmup.
        for _ in range(warmup_runs):
            predict_fn(batches[0])
        latencies_ms: list[float] = []
        total_tokens = 0
        for _ in range(num_runs):
            for batch in batches:
                if not batch:
                    continue
                t0 = time.perf_counter()
                predict_fn(batch)
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)
                total_tokens += len(batch)
        if device == "cuda":
            try:
                import torch
                torch.cuda.synchronize()
                peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
            except Exception:
                peak_mem = 0.0
        else:
            try:
                import resource
                peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            except Exception:
                peak_mem = 0.0
        total_time_s = sum(latencies_ms) / 1000.0
        tps = total_tokens / total_time_s if total_time_s > 0 else 0.0
        reports.append(
            SpeedReport(
                name=name,
                device=device,
                batch_size=bs,
                tps=tps,
                p50_ms=_percentile(latencies_ms, 50),
                p95_ms=_percentile(latencies_ms, 95),
                p99_ms=_percentile(latencies_ms, 99),
                peak_mem_mb=peak_mem,
            )
        )
    return reports


def benchmark_competitors(test_tokens: list[str]) -> dict[str, float]:
    """Run available competitors on the same tokens. Missing ones are skipped."""
    results: dict[str, float] = {}
    if importlib.util.find_spec("zeyrek") is not None:
        try:
            import zeyrek
            analyzer = zeyrek.MorphAnalyzer()
            t0 = time.perf_counter()
            for tok in test_tokens:
                analyzer.analyze(tok)
            dt = time.perf_counter() - t0
            results["zeyrek"] = len(test_tokens) / dt if dt > 0 else 0.0
        except Exception:
            pass
    if importlib.util.find_spec("stanza") is not None:  # pragma: no cover - optional
        try:
            import stanza
            nlp = stanza.Pipeline(lang="tr", processors="tokenize,mwt,pos,lemma", verbose=False)
            t0 = time.perf_counter()
            nlp(" ".join(test_tokens))
            dt = time.perf_counter() - t0
            results["stanza"] = len(test_tokens) / dt if dt > 0 else 0.0
        except Exception:
            pass
    if importlib.util.find_spec("spacy") is not None:  # pragma: no cover - optional
        try:
            import spacy
            nlp = spacy.blank("xx")
            t0 = time.perf_counter()
            list(nlp.pipe(test_tokens))
            dt = time.perf_counter() - t0
            results["spacy_blank"] = len(test_tokens) / dt if dt > 0 else 0.0
        except Exception:
            pass
    return results


def generate_speed_report(
    bundle: BenchmarkBundle, output_path: str | Path
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Speed Benchmark\n",
        "## Model",
        "| Name | Device | Batch | TPS | p50 ms | p95 ms | p99 ms | Peak MB |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in bundle.model_results:
        lines.append(
            f"| {r.name} | {r.device} | {r.batch_size} | {r.tps:.1f} | "
            f"{r.p50_ms:.3f} | {r.p95_ms:.3f} | {r.p99_ms:.3f} | {r.peak_mem_mb:.1f} |"
        )
    lines.append("\n## Competitors (TPS, CPU)")
    for name, tps in bundle.competitor_results.items():
        lines.append(f"- {name}: {tps:.1f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
