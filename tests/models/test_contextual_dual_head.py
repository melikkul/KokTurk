"""Tests for context encoders and ContextualDualHeadAtomizer.

Covers:
1. POSBigramContext output shape
2. Word2VecContext output shape
3. SentenceBiGRUContext output shape
4. BERTurkContext output shape (BERT mocked — no download required)
5. ContextualDualHeadAtomizer forward shapes
6. Context actually changes output (non-zero effect)
7. Loss decreases over 5 gradient steps
8. greedy_decode returns valid strings
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F

from kokturk.models.context_encoder import (
    BERTurkContext,
    POSBigramContext,
    SentenceBiGRUContext,
    Word2VecContext,
)
from kokturk.models.contextual_dual_head import ContextualDualHeadAtomizer

# ---------------------------------------------------------------------------
# Shared constants for tests
# ---------------------------------------------------------------------------
B = 4        # batch size
CHAR_V = 50  # char vocab size
TAG_V = 100  # tag vocab size
ROOT_V = 30  # root vocab size
L_CHAR = 10  # character sequence length
MAX_TAG = 8  # tag sequence length (including SOS, root, tags, EOS, PAD)
SENT_LEN = 16  # sentence length for context encoders


def _make_chars() -> torch.Tensor:
    """Random character index tensor (B, L_CHAR)."""
    chars = torch.randint(1, CHAR_V, (B, L_CHAR))
    # Pad last 2 positions to test masking
    chars[:, -2:] = 0
    return chars


def _make_tag_ids() -> torch.Tensor:
    """Tag ids: [SOS=1, root_token, +TAG1, ..., EOS=2, PAD=0]."""
    tag_ids = torch.zeros(B, MAX_TAG, dtype=torch.long)
    tag_ids[:, 0] = 1   # SOS
    tag_ids[:, 1] = 5   # root token index
    tag_ids[:, 2] = 10  # first tag
    tag_ids[:, 3] = 11  # second tag
    tag_ids[:, 4] = 2   # EOS
    return tag_ids


def _make_gold_root() -> torch.Tensor:
    return torch.randint(0, ROOT_V, (B,))


def _make_model(ctx_enc) -> ContextualDualHeadAtomizer:
    return ContextualDualHeadAtomizer(
        context_encoder=ctx_enc,
        char_vocab_size=CHAR_V,
        tag_vocab_size=TAG_V,
        root_vocab_size=ROOT_V,
        embed_dim=16,
        hidden_dim=32,
        num_layers=2,
        dropout=0.0,
        context_dropout=0.0,
        max_decode_len=MAX_TAG,
    )


# ---------------------------------------------------------------------------
# 1. POSBigramContext shape
# ---------------------------------------------------------------------------

class TestPOSBigramContext:
    def test_shape(self):
        enc = POSBigramContext(num_pos_tags=20, pos_embed_dim=8)
        pos_ids = torch.randint(0, 20, (B, 2))
        out = enc(pos_ids)
        assert out.shape == (B, 2 * 8), f"Expected (4, 16), got {out.shape}"

    def test_padding_zeros(self):
        """Index 0 (boundary) should produce zero-like embeddings (padding_idx)."""
        enc = POSBigramContext(num_pos_tags=20, pos_embed_dim=8)
        # All-zero input → should embed as zeros (padding_idx=0)
        pos_ids = torch.zeros(B, 2, dtype=torch.long)
        out = enc(pos_ids)
        assert out.abs().sum().item() == pytest.approx(0.0, abs=1e-6)

    def test_output_dim_property(self):
        enc = POSBigramContext(num_pos_tags=20, pos_embed_dim=8)
        assert enc.output_dim == 16


# ---------------------------------------------------------------------------
# 2. Word2VecContext shape
# ---------------------------------------------------------------------------

class TestWord2VecContext:
    def test_shape(self):
        enc = Word2VecContext(vocab_size=200, embed_dim=32, gru_hidden_dim=24)
        neighbor_ids = torch.randint(0, 200, (B, 4))
        out = enc(neighbor_ids)
        assert out.shape == (B, 2 * 24), f"Expected (4, 48), got {out.shape}"

    def test_pretrained_weights_frozen(self):
        weights = torch.randn(200, 32)
        enc = Word2VecContext(
            vocab_size=200, embed_dim=32, gru_hidden_dim=24,
            pretrained_weights=weights,
        )
        assert not enc.word_embed.weight.requires_grad

    def test_output_dim_property(self):
        enc = Word2VecContext(vocab_size=200, embed_dim=32, gru_hidden_dim=24)
        assert enc.output_dim == 48


# ---------------------------------------------------------------------------
# 3. SentenceBiGRUContext shape
# ---------------------------------------------------------------------------

class TestSentenceBiGRUContext:
    def test_shape(self):
        enc = SentenceBiGRUContext(vocab_size=200, embed_dim=32, hidden_dim=48)
        word_ids = torch.randint(0, 200, (B, SENT_LEN))
        target_pos = torch.randint(0, SENT_LEN, (B,))
        out = enc(word_ids, target_pos)
        assert out.shape == (B, 2 * 48), f"Expected (4, 96), got {out.shape}"

    def test_different_positions_differ(self):
        """Different target positions should give different context vectors."""
        enc = SentenceBiGRUContext(vocab_size=200, embed_dim=32, hidden_dim=48)
        word_ids = torch.randint(1, 200, (B, SENT_LEN))
        pos_a = torch.zeros(B, dtype=torch.long)
        pos_b = torch.full((B,), SENT_LEN // 2, dtype=torch.long)
        out_a = enc(word_ids, pos_a)
        out_b = enc(word_ids, pos_b)
        assert not torch.allclose(out_a, out_b)

    def test_output_dim_property(self):
        enc = SentenceBiGRUContext(vocab_size=200, embed_dim=32, hidden_dim=48)
        assert enc.output_dim == 96


# ---------------------------------------------------------------------------
# 4. BERTurkContext shape (BERT mocked)
# ---------------------------------------------------------------------------

class TestBERTurkContext:
    def _make_mock_bert_and_tokenizer(self, seq_len: int = 20):
        """Build mock BERT model and tokenizer that return predictable shapes."""
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(B, seq_len, 768)

        mock_bert = MagicMock()
        mock_bert.return_value = mock_output
        # Make parameters() iterable so next(self.proj.parameters()).device works
        mock_bert.parameters = lambda: iter([])

        # Tokenizer: returns dict-like with input_ids + attention_mask
        mock_encoding = MagicMock()
        mock_encoding.__getitem__ = MagicMock(side_effect=lambda k: {
            "input_ids": torch.zeros(B, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(B, seq_len, dtype=torch.long),
        }[k])
        # word_ids(b) maps subword idx → word idx (simulate 1:1 for simplicity)
        mock_encoding.word_ids = MagicMock(
            side_effect=lambda b: [None] + list(range(seq_len - 2)) + [None]
        )

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = mock_encoding

        return mock_bert, mock_tokenizer

    def test_shape(self):
        """BERTurkContext returns (B, context_dim) with mocked BERT."""
        mock_bert, mock_tokenizer = self._make_mock_bert_and_tokenizer()
        enc = BERTurkContext(
            bert_model=mock_bert,
            tokenizer=mock_tokenizer,
            context_dim=64,
        )
        # Verify sentence_text is accepted (no pre-tokenisation needed by caller)
        sentences = ["ev geldi bugün", "çocuk kitap okuyor"] * (B // 2)
        positions = [0, 1, 2, 1]

        out = enc(sentences, positions)
        assert out.shape == (B, 64), f"Expected (4, 64), got {out.shape}"

    def test_output_dim_property(self):
        mock_bert, mock_tokenizer = self._make_mock_bert_and_tokenizer()
        enc = BERTurkContext(
            bert_model=mock_bert, tokenizer=mock_tokenizer, context_dim=64
        )
        assert enc.output_dim == 64

    def test_bert_not_called_before_forward(self):
        """BERT is not called at init — only during forward()."""
        mock_bert, mock_tokenizer = self._make_mock_bert_and_tokenizer()
        _enc = BERTurkContext(
            bert_model=mock_bert, tokenizer=mock_tokenizer, context_dim=64
        )
        mock_bert.assert_not_called()


# ---------------------------------------------------------------------------
# 5. ContextualDualHeadAtomizer forward shapes
# ---------------------------------------------------------------------------

class TestContextualDualHeadAtomizerForward:
    def test_shapes_with_word2vec_context(self):
        """Tag outputs have shape (B, MAX_TAG-2, TAG_V); root logits (B, ROOT_V)."""
        ctx_enc = Word2VecContext(vocab_size=200, embed_dim=16, gru_hidden_dim=16)
        model = _make_model(ctx_enc)

        chars = _make_chars()
        tag_ids = _make_tag_ids()
        gold_root = _make_gold_root()
        context = torch.randint(0, 200, (B, 4))  # word indices for ±2 neighbours

        root_logits, tag_outputs = model(
            chars, context, tag_ids=tag_ids, gold_root=gold_root
        )

        assert root_logits.shape == (B, ROOT_V), f"root_logits: {root_logits.shape}"
        assert tag_outputs.shape == (B, MAX_TAG - 2, TAG_V), (
            f"tag_outputs: {tag_outputs.shape}"
        )

    def test_shapes_with_tuple_context(self):
        """Tuple context_inputs correctly unpacked for SentenceBiGRUContext."""
        ctx_enc = SentenceBiGRUContext(vocab_size=200, embed_dim=16, hidden_dim=16)
        model = _make_model(ctx_enc)

        chars = _make_chars()
        tag_ids = _make_tag_ids()
        gold_root = _make_gold_root()
        word_ids = torch.randint(0, 200, (B, SENT_LEN))
        target_pos = torch.randint(0, SENT_LEN, (B,))

        # context_inputs as tuple → unpacked as (*args)
        root_logits, tag_outputs = model(
            chars, (word_ids, target_pos), tag_ids=tag_ids, gold_root=gold_root
        )
        assert root_logits.shape == (B, ROOT_V)
        assert tag_outputs.shape == (B, MAX_TAG - 2, TAG_V)


# ---------------------------------------------------------------------------
# 6. Context actually changes output
# ---------------------------------------------------------------------------

class TestContextAffectsOutput:
    def test_different_context_different_root_logits(self):
        """Identical chars + different context → different root logits."""
        ctx_enc = Word2VecContext(vocab_size=200, embed_dim=16, gru_hidden_dim=16)
        model = _make_model(ctx_enc)
        model.eval()

        chars = _make_chars()
        tag_ids = _make_tag_ids()
        gold_root = _make_gold_root()

        ctx_a = torch.zeros(B, 4, dtype=torch.long)
        ctx_b = torch.ones(B, 4, dtype=torch.long) * 100  # different words

        with torch.no_grad():
            root_a, _ = model(
                chars, ctx_a, tag_ids=tag_ids, gold_root=gold_root, use_gold_root=False
            )
            root_b, _ = model(
                chars, ctx_b, tag_ids=tag_ids, gold_root=gold_root, use_gold_root=False
            )

        assert not torch.allclose(root_a, root_b), (
            "Root logits must differ when context differs"
        )


# ---------------------------------------------------------------------------
# 7. Loss decreases over gradient steps
# ---------------------------------------------------------------------------

class TestContextualLossDecreases:
    def test_loss_decreases(self):
        ctx_enc = Word2VecContext(vocab_size=200, embed_dim=16, gru_hidden_dim=16)
        model = _make_model(ctx_enc)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        chars = _make_chars()
        tag_ids = _make_tag_ids()
        gold_root = _make_gold_root()
        context = torch.randint(0, 200, (B, 4))

        losses: list[float] = []
        for _ in range(5):
            optimizer.zero_grad()
            root_logits, tag_outputs = model(
                chars, context, tag_ids=tag_ids,
                gold_root=gold_root, teacher_forcing_ratio=1.0
            )
            root_loss = F.cross_entropy(root_logits, gold_root)
            tag_targets = tag_ids[:, 2:]
            tag_loss = F.cross_entropy(
                tag_outputs.reshape(-1, TAG_V),
                tag_targets.reshape(-1),
                ignore_index=0,
            )
            loss = tag_loss + 0.3 * root_loss
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        )


# ---------------------------------------------------------------------------
# 8. Greedy decode produces valid strings
# ---------------------------------------------------------------------------

class TestContextualGreedyDecode:
    def test_decode_produces_strings(self):
        ctx_enc = Word2VecContext(vocab_size=200, embed_dim=16, gru_hidden_dim=16)
        model = _make_model(ctx_enc)
        model.eval()

        chars = _make_chars()
        context = torch.randint(0, 200, (B, 4))

        # Build minimal vocab inverses
        root_vocab_inv = [f"root_{i}" for i in range(ROOT_V)]
        tag_vocab_inv = (
            ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
            + [f"+TAG{i}" for i in range(TAG_V - 4)]
        )

        results = model.greedy_decode(
            chars, context,
            root_vocab_inv=root_vocab_inv,
            tag_vocab_inv=tag_vocab_inv,
        )

        assert len(results) == B
        for label in results:
            assert isinstance(label, str)
            assert len(label) > 0, "Label must be non-empty"
            # Root should not start with "+" (that would be a tag, not a root)
            root_token = label.split()[0]
            assert not root_token.startswith("+"), (
                f"Root token '{root_token}' starts with '+' — should be a root word"
            )

    def test_decode_without_vocab_inv(self):
        """greedy_decode works without vocab inverses (falls back to index strings)."""
        ctx_enc = Word2VecContext(vocab_size=200, embed_dim=16, gru_hidden_dim=16)
        model = _make_model(ctx_enc)
        model.eval()

        chars = _make_chars()
        context = torch.randint(0, 200, (B, 4))
        results = model.greedy_decode(chars, context)
        assert len(results) == B
        assert all(isinstance(r, str) and r for r in results)
