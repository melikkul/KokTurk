"""Tests for benchmark.speed_benchmark."""

from __future__ import annotations

from benchmark.speed_benchmark import (
    BenchmarkBundle,
    _percentile,
    benchmark_competitors,
    benchmark_inference,
    generate_speed_report,
)


def test_percentile_basic():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _percentile(vals, 50) == 3.0
    assert _percentile(vals, 100) == 5.0
    assert _percentile(vals, 0) == 1.0


def test_percentile_empty_returns_zero():
    assert _percentile([], 50) == 0.0


def test_benchmark_inference_runs_stub_model():
    def stub(batch):
        return ["x"] * len(batch)

    tokens = ["a"] * 20
    reports = benchmark_inference(
        stub, tokens, num_runs=2, warmup_runs=1, batch_sizes=(1, 4), name="stub"
    )
    assert len(reports) == 2
    for r in reports:
        assert r.tps >= 0
        assert r.p50_ms >= 0
        assert r.batch_size in (1, 4)


def test_benchmark_competitors_skips_missing_gracefully():
    # Whether or not zeyrek is installed, this must not raise.
    out = benchmark_competitors(["ev", "el"])
    assert isinstance(out, dict)


def test_generate_speed_report_writes_markdown(tmp_path):
    bundle = BenchmarkBundle()
    bundle.competitor_results = {"zeyrek": 123.4}

    def stub(batch):
        return list(batch)

    bundle.model_results = benchmark_inference(
        stub, ["a"] * 5, num_runs=2, warmup_runs=1, batch_sizes=(1,), name="stub"
    )
    out = tmp_path / "speed.md"
    generate_speed_report(bundle, out)
    text = out.read_text(encoding="utf-8")
    assert "Speed Benchmark" in text
    assert "zeyrek" in text
