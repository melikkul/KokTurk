"""Tests for AtomizerEnsemble majority vote logic."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import torch

from kokturk.models.ensemble import AtomizerEnsemble

# ---------------------------------------------------------------------------
# Majority vote logic (no model loading required)
# ---------------------------------------------------------------------------


class TestMajorityVoteLogic:
    """Test the vote-counting logic used by AtomizerEnsemble.analyze."""

    def test_clear_majority(self) -> None:
        predictions = ["ev +PLU", "ev +PLU", "ev +ACC"]
        counts = Counter(predictions)
        best = counts.most_common(1)[0][0]
        assert best == "ev +PLU"

    def test_unanimous(self) -> None:
        predictions = ["git +PAST", "git +PAST", "git +PAST"]
        counts = Counter(predictions)
        best = counts.most_common(1)[0][0]
        assert best == "git +PAST"

    def test_all_different_picks_first_in_counter(self) -> None:
        """When all predictions differ, Counter.most_common returns one of them."""
        predictions = ["ev +PLU", "ev +ACC", "ev +DAT"]
        counts = Counter(predictions)
        best = counts.most_common(1)[0][0]
        assert best in predictions

    def test_tie_breaking_by_em(self) -> None:
        """Simulate tie-breaking: prefer prediction from model with highest EM."""
        predictions = ["ev +PLU", "ev +ACC", "ev +DAT"]
        model_ems = [0.80, 0.85, 0.82]

        counts = Counter(predictions)
        best_pred = counts.most_common(1)[0][0]
        max_count = counts[best_pred]
        tied = [p for p, c in counts.items() if c == max_count]

        if len(tied) > 1:
            for _em, pred in sorted(zip(model_ems, predictions, strict=False), reverse=True):
                if pred in tied:
                    best_pred = pred
                    break

        # Model with EM=0.85 predicted "ev +ACC"
        assert best_pred == "ev +ACC"


# ---------------------------------------------------------------------------
# Decode prediction helper (requires Vocab — test with mock)
# ---------------------------------------------------------------------------


class TestDecodePrediction:
    """Test _decode_prediction by mocking the Vocab dependency."""

    def test_decode_stops_at_eos(self) -> None:
        """Decoding should stop at EOS_IDX (=2)."""
        from unittest.mock import MagicMock

        ensemble = object.__new__(AtomizerEnsemble)

        # Build a minimal mock tag_vocab
        tag_vocab = MagicMock()
        tag_vocab.decode.side_effect = lambda idx: {
            4: "ev",
            5: "+PLU",
            6: "+ACC",
        }.get(idx, "<unk>")
        ensemble.tag_vocab = tag_vocab

        # Prediction: [4, 5, 2, 6] — should stop at EOS (2), ignore 6
        pred = torch.tensor([[4, 5, 2, 6]], dtype=torch.long)
        result = ensemble._decode_prediction(pred)
        assert result == "ev +PLU"

    def test_decode_skips_special_tokens(self) -> None:
        """Indices 0-3 (PAD/SOS/EOS/UNK) should be skipped."""
        from unittest.mock import MagicMock

        ensemble = object.__new__(AtomizerEnsemble)
        tag_vocab = MagicMock()
        tag_vocab.decode.side_effect = lambda idx: {5: "git", 6: "+PAST"}.get(idx, "?")
        ensemble.tag_vocab = tag_vocab

        # Indices: 0 (PAD), 1 (SOS), 3 (UNK), 5, 6 — only 5 and 6 decoded
        pred = torch.tensor([[0, 1, 3, 5, 6]], dtype=torch.long)
        result = ensemble._decode_prediction(pred)
        assert result == "git +PAST"


# ---------------------------------------------------------------------------
# Full ensemble (skip if checkpoints don't exist)
# ---------------------------------------------------------------------------

CKPT_DIR = Path("$PROJECT_DIR/models/ensemble")
VOCAB_DIR = Path("$PROJECT_DIR/models/vocab")


@pytest.mark.skipif(
    not CKPT_DIR.exists() or not any(CKPT_DIR.glob("*.pt")),
    reason="No ensemble checkpoints found",
)
class TestEnsembleIntegration:
    """Integration tests — only run if checkpoints are available."""

    def test_instantiation(self) -> None:
        paths = sorted(CKPT_DIR.glob("*.pt"))
        ensemble = AtomizerEnsemble(
            model_paths=paths,
            vocab_dir=VOCAB_DIR,
            device="cpu",
        )
        assert len(ensemble.models) == len(paths)

    def test_analyze_returns_string(self) -> None:
        paths = sorted(CKPT_DIR.glob("*.pt"))
        ensemble = AtomizerEnsemble(
            model_paths=paths,
            vocab_dir=VOCAB_DIR,
            device="cpu",
        )
        result = ensemble.analyze("evlerinden")
        assert isinstance(result, str)
        assert len(result) > 0
