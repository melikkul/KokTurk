"""Tests for benchmark.llm_baseline."""

from __future__ import annotations

import dataclasses
import json

import pytest

from benchmark.llm_baseline import (
    LLMBaselineResult,
    build_prompt,
    evaluate_via_anthropic,
    evaluate_via_openai,
    generate_prompt_templates,
    parse_llm_response,
)


def test_build_prompt_interpolation():
    p = build_prompt("evlerinden")
    assert "evlerinden" in p
    assert "root" in p
    assert "tags" in p


def test_build_prompt_with_context():
    p = build_prompt("evlerinden", context_sentence="Onlar evlerinden cikti.")
    assert "evlerinden" in p
    assert "Onlar evlerinden cikti." in p


def test_parse_valid_json():
    raw = '{"root": "ev", "tags": ["+Noun", "+PLU", "+ABL"]}'
    result = parse_llm_response(raw)
    assert result is not None
    assert result["root"] == "ev"
    assert "+PLU" in result["tags"]


def test_parse_fenced_json():
    raw = '```json\n{"root": "ev", "tags": ["+PLU"]}\n```'
    result = parse_llm_response(raw)
    assert result is not None
    assert result["root"] == "ev"


def test_parse_fenced_no_lang():
    raw = '```\n{"root": "ev", "tags": ["+PLU"]}\n```'
    result = parse_llm_response(raw)
    assert result is not None
    assert result["root"] == "ev"


def test_parse_json_with_surrounding_text():
    raw = 'Here is the analysis:\n{"root": "gel", "tags": ["+Verb", "+PAST"]}\nDone.'
    result = parse_llm_response(raw)
    assert result is not None
    assert result["root"] == "gel"


def test_parse_garbage_returns_none():
    assert parse_llm_response("I don't know how to analyze this") is None
    assert parse_llm_response("") is None
    assert parse_llm_response("   ") is None


def test_parse_json_missing_keys_returns_none():
    assert parse_llm_response('{"lemma": "ev"}') is None
    assert parse_llm_response('{"root": "ev"}') is None


def test_evaluate_anthropic_returns_none_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert evaluate_via_anthropic([]) is None


def test_evaluate_openai_returns_none_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert evaluate_via_openai([]) is None


def test_generate_prompt_templates_writes_json(tmp_path):
    test_data = [
        {"surface": "evlerinden", "expected_root": "ev", "expected_tags": ["+Noun", "+PLU"]},
        {"surface": "gidiyorum", "expected_root": "gitmek", "expected_tags": ["+Verb", "+PROG"]},
    ]
    out = tmp_path / "prompts.json"
    result_path = generate_prompt_templates(test_data, out, num_samples=2)
    assert result_path == out
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data) == 2
    assert "prompt" in data[0]
    assert "gold_root" in data[0]
    assert "evlerinden" in data[0]["prompt"]


def test_llm_baseline_result_dataclass():
    r = LLMBaselineResult(
        model_name="test-model",
        em=0.37,
        lemma_accuracy=0.45,
        tag_accuracy=0.30,
        n=100,
        avg_latency_ms=1500.0,
        cost_usd=0.05,
    )
    assert r.model_name == "test-model"
    assert r.em == 0.37
    assert r.cost_usd == 0.05
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.em = 0.5  # type: ignore[misc]


def test_llm_baseline_result_none_cost():
    r = LLMBaselineResult(
        model_name="x",
        em=0.0,
        lemma_accuracy=0.0,
        tag_accuracy=0.0,
        n=0,
        avg_latency_ms=0.0,
        cost_usd=None,
    )
    assert r.cost_usd is None
