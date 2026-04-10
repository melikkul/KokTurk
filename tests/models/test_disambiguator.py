"""Tests for the BERTurk morphological disambiguator.

Covers: CandidateEncoder shape, BERTurkDisambiguator forward (with mock BERT),
score masking, single-candidate trivial case, loss decrease, gold matching,
and OOV handling.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import torch
import pytest

from kokturk.models.disambiguator import BERTurkDisambiguator, CandidateEncoder

B = 4  # batch size for tests
K = 10  # max candidates
L = 15  # max parse length
SEQ_LEN = 20  # mock BERT sequence length
TAG_VOCAB_SIZE = 100  # small vocab for tests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_bert_and_tokenizer(batch_size: int = B, seq_len: int = SEQ_LEN):
    """Build mock BERT model and tokenizer with predictable shapes."""
    mock_output = MagicMock()
    mock_output.last_hidden_state = torch.randn(batch_size, seq_len, 768)

    mock_bert = MagicMock()
    mock_bert.return_value = mock_output
    mock_bert.parameters = lambda: iter([])

    mock_encoding = MagicMock()
    mock_encoding.__getitem__ = MagicMock(side_effect=lambda k: {
        "input_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
    }[k])
    mock_encoding.word_ids = MagicMock(
        side_effect=lambda b: [None] + list(range(seq_len - 2)) + [None],
    )

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = mock_encoding

    return mock_bert, mock_tokenizer


def _make_model(batch_size: int = B) -> BERTurkDisambiguator:
    """Create a disambiguator with mocked BERT."""
    mock_bert, mock_tokenizer = _make_mock_bert_and_tokenizer(batch_size)
    return BERTurkDisambiguator(
        tag_vocab_size=TAG_VOCAB_SIZE,
        bert_model=mock_bert,
        tokenizer=mock_tokenizer,
    )


# ---------------------------------------------------------------------------
# 1. CandidateEncoder shape
# ---------------------------------------------------------------------------

class TestCandidateEncoder:
    def test_output_shape(self):
        """CandidateEncoder returns (B, K, output_dim)."""
        enc = CandidateEncoder(
            tag_vocab_size=TAG_VOCAB_SIZE, embed_dim=32, hidden_dim=64,
        )
        x = torch.randint(0, TAG_VOCAB_SIZE, (B, K, L))
        out = enc(x)
        assert out.shape == (B, K, 128), f"Expected (4, 10, 128), got {out.shape}"

    def test_output_dim_property(self):
        enc = CandidateEncoder(
            tag_vocab_size=TAG_VOCAB_SIZE, embed_dim=32, hidden_dim=64,
        )
        assert enc.output_dim == 128  # 64 * 2 (bidirectional)

    def test_padding_idx_zero(self):
        """Embedding at index 0 should be zero (padding)."""
        enc = CandidateEncoder(tag_vocab_size=TAG_VOCAB_SIZE)
        zeros = enc.embedding(torch.zeros(1, dtype=torch.long))
        assert torch.allclose(zeros, torch.zeros_like(zeros))


# ---------------------------------------------------------------------------
# 2. BERTurkDisambiguator forward with mock BERT
# ---------------------------------------------------------------------------

class TestBERTurkDisambiguator:
    def test_forward_shape(self):
        """Forward returns logits (B, K) and loss scalar."""
        model = _make_model()
        sentences = ["ev geldi bugün", "kitap okuyor"] * (B // 2)
        positions = torch.tensor([0, 1, 2, 0])
        cand_ids = torch.randint(0, TAG_VOCAB_SIZE, (B, K, L))
        cand_mask = torch.ones(B, K, dtype=torch.bool)
        gold = torch.zeros(B, dtype=torch.long)

        logits, loss = model(
            sentence_texts=sentences,
            target_positions=positions,
            candidate_ids=cand_ids,
            candidate_mask=cand_mask,
            gold_indices=gold,
        )

        assert logits.shape == (B, K)
        assert loss is not None
        assert loss.ndim == 0  # scalar

    def test_forward_no_loss_without_gold(self):
        """Forward without gold_indices returns None loss."""
        model = _make_model()
        sentences = ["test sentence"] * B
        positions = torch.zeros(B, dtype=torch.long)
        cand_ids = torch.randint(0, TAG_VOCAB_SIZE, (B, K, L))
        cand_mask = torch.ones(B, K, dtype=torch.bool)

        logits, loss = model(
            sentence_texts=sentences,
            target_positions=positions,
            candidate_ids=cand_ids,
            candidate_mask=cand_mask,
        )

        assert logits.shape == (B, K)
        assert loss is None

    def test_cached_bert_embeds(self):
        """Forward with cached_bert_embeds skips BERT forward pass."""
        model = _make_model()
        sentences = ["test"] * B
        positions = torch.zeros(B, dtype=torch.long)
        cand_ids = torch.randint(0, TAG_VOCAB_SIZE, (B, K, L))
        cand_mask = torch.ones(B, K, dtype=torch.bool)
        cached = torch.randn(B, 768)

        logits, _ = model(
            sentence_texts=sentences,
            target_positions=positions,
            candidate_ids=cand_ids,
            candidate_mask=cand_mask,
            cached_bert_embeds=cached,
        )

        assert logits.shape == (B, K)
        # BERT should not have been called since we provided cached embeds
        model.bert.assert_not_called()


# ---------------------------------------------------------------------------
# 3. Score masking
# ---------------------------------------------------------------------------

class TestScoreMasking:
    def test_invalid_candidates_masked(self):
        """Invalid candidates (mask=False) get -inf score."""
        model = _make_model()
        sentences = ["test"] * B
        positions = torch.zeros(B, dtype=torch.long)
        cand_ids = torch.randint(0, TAG_VOCAB_SIZE, (B, K, L))

        # Only first 2 candidates are valid
        cand_mask = torch.zeros(B, K, dtype=torch.bool)
        cand_mask[:, :2] = True

        logits, _ = model(
            sentence_texts=sentences,
            target_positions=positions,
            candidate_ids=cand_ids,
            candidate_mask=cand_mask,
        )

        # Invalid positions should be -inf
        assert (logits[:, 2:] == float("-inf")).all()
        # Valid positions should be finite
        assert torch.isfinite(logits[:, :2]).all()

    def test_argmax_ignores_masked(self):
        """Argmax should only select from valid candidates."""
        model = _make_model()
        sentences = ["test"] * B
        positions = torch.zeros(B, dtype=torch.long)
        cand_ids = torch.randint(0, TAG_VOCAB_SIZE, (B, K, L))

        cand_mask = torch.zeros(B, K, dtype=torch.bool)
        cand_mask[:, :3] = True  # only 3 valid candidates

        logits, _ = model(
            sentence_texts=sentences,
            target_positions=positions,
            candidate_ids=cand_ids,
            candidate_mask=cand_mask,
        )

        preds = logits.argmax(dim=-1)
        assert (preds < 3).all(), "Predictions should be within valid candidates"


# ---------------------------------------------------------------------------
# 4. Single candidate trivial case
# ---------------------------------------------------------------------------

class TestSingleCandidate:
    def test_single_candidate_always_selected(self):
        """When only 1 candidate exists, it should always be selected."""
        model = _make_model()
        sentences = ["test"] * B
        positions = torch.zeros(B, dtype=torch.long)
        cand_ids = torch.randint(0, TAG_VOCAB_SIZE, (B, K, L))

        # Only candidate 0 is valid
        cand_mask = torch.zeros(B, K, dtype=torch.bool)
        cand_mask[:, 0] = True

        logits, _ = model(
            sentence_texts=sentences,
            target_positions=positions,
            candidate_ids=cand_ids,
            candidate_mask=cand_mask,
        )

        preds = logits.argmax(dim=-1)
        assert (preds == 0).all(), "Single candidate must always be selected"


# ---------------------------------------------------------------------------
# 5. Loss decreases over gradient steps
# ---------------------------------------------------------------------------

class TestTraining:
    def test_loss_decreases(self):
        """Loss should decrease over 5 gradient steps."""
        model = _make_model()
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=1e-2)

        sentences = ["ev geldi bugün kitap"] * B
        positions = torch.tensor([0, 1, 2, 3])
        cand_ids = torch.randint(1, TAG_VOCAB_SIZE, (B, K, L))
        cand_mask = torch.ones(B, K, dtype=torch.bool)
        gold = torch.zeros(B, dtype=torch.long)

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            _, loss = model(
                sentence_texts=sentences,
                target_positions=positions,
                candidate_ids=cand_ids,
                candidate_mask=cand_mask,
                gold_indices=gold,
            )
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss should decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )


# ---------------------------------------------------------------------------
# 6. Gold matching (Zeyrek .to_str() matches gold label format)
# ---------------------------------------------------------------------------

class TestGoldMatching:
    def test_to_str_format_matches_gold(self):
        """MorphologicalAnalysis.to_str() produces the same format as gold labels."""
        from kokturk.core.datatypes import MorphologicalAnalysis

        # Simulate what Zeyrek produces after ZEYREK_TO_CANONICAL mapping
        analysis = MorphologicalAnalysis(
            surface="evlerinden",
            root="ev",
            tags=("+Noun", "+PLU", "+POSS.3SG", "+ABL"),
            morphemes=(),
            source="zeyrek",
            score=1.0,
        )

        expected_gold = "ev +Noun +PLU +POSS.3SG +ABL"
        assert analysis.to_str() == expected_gold

    def test_root_only_format(self):
        """Root with single POS tag matches gold format."""
        from kokturk.core.datatypes import MorphologicalAnalysis

        analysis = MorphologicalAnalysis(
            surface="bir",
            root="bir",
            tags=("+Det",),
            morphemes=(),
            source="zeyrek",
            score=1.0,
        )

        assert analysis.to_str() == "bir +Det"


# ---------------------------------------------------------------------------
# 7. OOV handling
# ---------------------------------------------------------------------------

class TestOOVHandling:
    def test_oov_gold_added_as_candidate(self):
        """When gold not in candidates, it should be added as extra candidate."""
        # Simulate the logic from DisambiguationDataset
        candidates = ["ev +Noun +PLU", "ev +Noun +ACC"]
        gold_label = "ev +Noun +POSS.3SG"

        gold_idx = -1
        for i, c in enumerate(candidates):
            if c == gold_label:
                gold_idx = i
                break

        if gold_idx == -1:
            candidates.append(gold_label)
            gold_idx = len(candidates) - 1

        assert gold_idx == 2
        assert candidates[gold_idx] == gold_label
        assert len(candidates) == 3
