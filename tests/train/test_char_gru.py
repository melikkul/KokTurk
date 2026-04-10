"""Tests for the GRU Seq2Seq morphological atomizer model."""

from __future__ import annotations

import torch

from kokturk.models.char_gru import BahdanauAttention, MorphAtomizer


class TestBahdanauAttention:
    def test_output_shapes(self) -> None:
        attn = BahdanauAttention(decoder_dim=128, encoder_dim=256, attn_dim=64)
        decoder_hidden = torch.randn(4, 128)
        encoder_outputs = torch.randn(4, 20, 256)
        context, weights = attn(decoder_hidden, encoder_outputs)
        assert context.shape == (4, 256)
        assert weights.shape == (4, 20)

    def test_weights_sum_to_one(self) -> None:
        attn = BahdanauAttention(decoder_dim=128, encoder_dim=256)
        context, weights = attn(torch.randn(2, 128), torch.randn(2, 10, 256))
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5)


class TestMorphAtomizer:
    def test_forward_with_teacher_forcing(self) -> None:
        model = MorphAtomizer(
            char_vocab_size=50, tag_vocab_size=100,
            embed_dim=32, hidden_dim=64, num_layers=1,
        )
        chars = torch.randint(1, 50, (4, 20))
        tags = torch.randint(1, 100, (4, 10))
        logits = model(chars, tags, teacher_forcing_ratio=1.0)
        assert logits.shape == (4, 10, 100)

    def test_forward_without_target(self) -> None:
        model = MorphAtomizer(
            char_vocab_size=50, tag_vocab_size=100,
            embed_dim=32, hidden_dim=64, num_layers=1,
            max_decode_len=8,
        )
        chars = torch.randint(1, 50, (2, 15))
        logits = model(chars, target_tags=None, teacher_forcing_ratio=0.0)
        assert logits.shape == (2, 8, 100)

    def test_greedy_decode(self) -> None:
        model = MorphAtomizer(
            char_vocab_size=50, tag_vocab_size=100,
            embed_dim=32, hidden_dim=64, num_layers=1,
            max_decode_len=8,
        )
        chars = torch.randint(1, 50, (3, 15))
        preds = model.greedy_decode(chars)
        assert preds.shape == (3, 8)

    def test_loss_decreases(self) -> None:
        """Verify the model can overfit on a tiny batch (loss goes down)."""
        torch.manual_seed(42)
        model = MorphAtomizer(
            char_vocab_size=50, tag_vocab_size=100,
            embed_dim=32, hidden_dim=64, num_layers=1,
        )
        chars = torch.randint(4, 50, (8, 12))
        tags = torch.randint(4, 100, (8, 6))

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        losses = []
        for _ in range(10):
            model.train()
            optimizer.zero_grad()
            logits = model(chars, tags, teacher_forcing_ratio=1.0)
            loss = criterion(
                logits.reshape(-1, 100), tags.reshape(-1),
            )
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease over 10 steps of overfitting
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.3f} → {losses[-1]:.3f}"
        )

    def test_count_parameters(self) -> None:
        model = MorphAtomizer(
            char_vocab_size=50, tag_vocab_size=100,
            embed_dim=32, hidden_dim=64, num_layers=1,
        )
        assert model.count_parameters() > 0
