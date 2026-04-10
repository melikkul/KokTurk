"""Tests for GRU Seq2Seq with copy mechanism (char_gru_copy)."""

from __future__ import annotations

import pytest
import torch

from kokturk.models.char_gru import MorphAtomizer
from kokturk.models.char_gru_copy import CopyTagDecoder, MorphAtomizerCopy

CHAR_VOCAB = 20
TAG_VOCAB = 30
EMBED_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 2
BATCH = 4
SEQ_LEN = 10
TAG_LEN = 8


@pytest.fixture()
def model() -> MorphAtomizerCopy:
    return MorphAtomizerCopy(
        char_vocab_size=CHAR_VOCAB,
        tag_vocab_size=TAG_VOCAB,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
    )


@pytest.fixture()
def chars() -> torch.Tensor:
    return torch.randint(1, CHAR_VOCAB, (BATCH, SEQ_LEN))


@pytest.fixture()
def target_tags() -> torch.Tensor:
    return torch.randint(1, TAG_VOCAB, (BATCH, TAG_LEN))


class TestForwardPass:
    """Forward pass shape and type tests."""

    def test_forward_returns_tuple(
        self, model: MorphAtomizerCopy, chars: torch.Tensor, target_tags: torch.Tensor,
    ) -> None:
        logits, root_logits = model(chars, target_tags, teacher_forcing_ratio=0.5)
        assert isinstance(logits, torch.Tensor)
        assert isinstance(root_logits, torch.Tensor)

    def test_logits_shape(
        self, model: MorphAtomizerCopy, chars: torch.Tensor, target_tags: torch.Tensor,
    ) -> None:
        logits, _ = model(chars, target_tags, teacher_forcing_ratio=0.5)
        assert logits.shape == (BATCH, TAG_LEN, TAG_VOCAB)

    def test_root_logits_shape(
        self, model: MorphAtomizerCopy, chars: torch.Tensor, target_tags: torch.Tensor,
    ) -> None:
        _, root_logits = model(chars, target_tags, teacher_forcing_ratio=0.5)
        assert root_logits.shape == (BATCH, TAG_VOCAB)

    def test_forward_no_target(
        self, model: MorphAtomizerCopy, chars: torch.Tensor,
    ) -> None:
        """Without target_tags, decode length defaults to max_decode_len."""
        logits, root_logits = model(chars, target_tags=None, teacher_forcing_ratio=0.0)
        assert logits.shape == (BATCH, model.max_decode_len, TAG_VOCAB)
        assert root_logits.shape == (BATCH, TAG_VOCAB)


class TestGreedyDecode:
    """Greedy decoding tests."""

    def test_decode_shape(
        self, model: MorphAtomizerCopy, chars: torch.Tensor,
    ) -> None:
        predicted = model.greedy_decode(chars)
        assert predicted.shape == (BATCH, model.max_decode_len)

    def test_decode_dtype(
        self, model: MorphAtomizerCopy, chars: torch.Tensor,
    ) -> None:
        predicted = model.greedy_decode(chars)
        assert predicted.dtype == torch.long


class TestCopyTagDecoder:
    """CopyTagDecoder-specific tests."""

    def test_p_gen_range(self) -> None:
        """p_gen values should be in [0, 1] (sigmoid output)."""
        encoder_dim = 2 * HIDDEN_DIM
        decoder = CopyTagDecoder(
            tag_vocab_size=TAG_VOCAB,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            encoder_dim=encoder_dim,
            num_layers=NUM_LAYERS,
        )
        prev_tag = torch.randint(0, TAG_VOCAB, (BATCH, 1))
        hidden = torch.randn(NUM_LAYERS, BATCH, HIDDEN_DIM)
        encoder_outputs = torch.randn(BATCH, SEQ_LEN, encoder_dim)

        _, _, _, p_gen = decoder.forward_step(prev_tag, hidden, encoder_outputs)
        assert p_gen.shape == (BATCH, 1)
        assert (p_gen >= 0.0).all()
        assert (p_gen <= 1.0).all()


class TestRootHead:
    """Root auxiliary head tests."""

    def test_root_head_output_shape(self, model: MorphAtomizerCopy) -> None:
        encoder_dim = 2 * HIDDEN_DIM
        pooled = torch.randn(BATCH, encoder_dim)
        out = model.root_head(pooled)
        assert out.shape == (BATCH, TAG_VOCAB)


class TestCountParameters:
    """Parameter counting tests."""

    def test_count_positive(self, model: MorphAtomizerCopy) -> None:
        assert model.count_parameters() > 0

    def test_copy_model_larger_than_base(self) -> None:
        """Copy model should have more parameters than base MorphAtomizer."""
        base = MorphAtomizer(
            char_vocab_size=CHAR_VOCAB,
            tag_vocab_size=TAG_VOCAB,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
        )
        copy = MorphAtomizerCopy(
            char_vocab_size=CHAR_VOCAB,
            tag_vocab_size=TAG_VOCAB,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
        )
        base_params = sum(p.numel() for p in base.parameters() if p.requires_grad)
        copy_params = copy.count_parameters()
        assert copy_params > base_params
