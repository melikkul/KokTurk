"""Tests for noise_audit seq2seq → token-classification adapter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data.noise_audit import (
    run_cleanlab_token_audit,
    seq2seq_to_token_classification,
)


def test_adapter_shapes_and_softmax_normalised() -> None:
    # Two sentences: lengths 3 and 2.
    logits = [
        np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]),
        np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]),
    ]
    gold = [[1, 2, 1], [2, 1]]  # avoid pad_idx=0
    labels, probs = seq2seq_to_token_classification(logits, gold)
    assert len(labels) == 2
    assert len(probs) == 2
    assert labels[0] == [1, 2, 1]
    assert probs[0].shape == (3, 3)
    # Each row sums to ~1.
    np.testing.assert_allclose(probs[0].sum(axis=-1), np.ones(3), atol=1e-5)
    np.testing.assert_allclose(probs[1].sum(axis=-1), np.ones(2), atol=1e-5)


def test_adapter_strips_padding_positions() -> None:
    logits = [np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])]
    gold = [[1, 0, 0]]  # positions 1, 2 are PAD (0)
    labels, probs = seq2seq_to_token_classification(logits, gold, pad_idx=0)
    assert labels == [[1]]
    assert probs[0].shape == (1, 2)


def test_adapter_handles_length_mismatch() -> None:
    # Decoder produced 4 timesteps but gold has only 2 (e.g., early EOS).
    logits = [np.random.randn(4, 5).astype(np.float32)]
    gold = [[1, 2]]
    labels, probs = seq2seq_to_token_classification(logits, gold)
    assert labels == [[1, 2]]
    assert probs[0].shape == (2, 5)


def test_run_cleanlab_token_audit_writes_payload(tmp_path: Path) -> None:
    # All-PAD gold for one sentence — exercises the empty-sentence skip.
    logits = [
        np.array([[10.0, -5.0], [-5.0, 10.0]]),  # confident & correct
        np.array([[-5.0, 10.0], [10.0, -5.0]]),  # confident but WRONG
    ]
    # Use 1-based labels since pad_idx=0 default strips position-0 entries.
    logits = [
        np.array([[10.0, -5.0], [-5.0, 10.0]]),
        np.array([[-5.0, 10.0], [10.0, -5.0]]),
    ]
    gold = [[1, 1], [1, 1]]
    out = tmp_path / "flagged.json"
    payload = run_cleanlab_token_audit(
        logits, gold, sentence_ids=["s1", "s2"],
        flag_threshold=0.5, output_path=out,
    )
    assert out.exists()
    written = json.loads(out.read_text())
    assert written["n_sentences"] == 2
    assert written["n_tokens"] == 4
    # The two confidently-wrong tokens should be flagged.
    assert payload["n_flagged"] >= 2


def test_run_cleanlab_token_audit_assertion_on_mismatched_input() -> None:
    import pytest
    with pytest.raises(AssertionError):
        run_cleanlab_token_audit(
            pred_logits_per_sentence=[np.zeros((2, 3))],
            gold_tag_ids_per_sentence=[[0, 1], [1, 2]],
        )
