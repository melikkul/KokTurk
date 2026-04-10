"""Minimal training pipeline test (tiny model, no real data needed)."""

from __future__ import annotations

import torch
import torch.nn as nn

from kokturk.models.char_gru import MorphAtomizer
from train.datasets import EOS_IDX, PAD_IDX, SOS_IDX, Vocab


class TestTrainingPipeline:
    """Verify the entire train→eval pipeline works with synthetic data."""

    def test_end_to_end_tiny(self) -> None:
        """Build model, train 3 steps, decode, verify loss decreases."""
        torch.manual_seed(42)

        # Tiny vocabs
        char_vocab = Vocab(list("abcdefghij"))
        tag_vocab = Vocab(["root_a", "root_b", "+TAG1", "+TAG2", "+TAG3"])

        model = MorphAtomizer(
            char_vocab_size=len(char_vocab),
            tag_vocab_size=len(tag_vocab),
            embed_dim=16, hidden_dim=32, num_layers=1, dropout=0.0,
        )

        # Synthetic batch: "abcde" → [SOS, root_a, +TAG1, EOS, PAD...]
        chars = torch.tensor([
            [char_vocab.encode(c) for c in "abcde"] + [EOS_IDX] + [PAD_IDX] * 4,
            [char_vocab.encode(c) for c in "fghij"] + [EOS_IDX] + [PAD_IDX] * 4,
        ])
        tags = torch.tensor([
            [SOS_IDX, tag_vocab.encode("root_a"), tag_vocab.encode("+TAG1"),
             EOS_IDX, PAD_IDX, PAD_IDX],
            [SOS_IDX, tag_vocab.encode("root_b"), tag_vocab.encode("+TAG2"),
             EOS_IDX, PAD_IDX, PAD_IDX],
        ])

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

        losses = []
        for _ in range(5):
            model.train()
            optimizer.zero_grad()
            logits = model(chars, tags, teacher_forcing_ratio=1.0)
            loss = criterion(logits.reshape(-1, len(tag_vocab)), tags.reshape(-1))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], "Loss should decrease"

        # Greedy decode should produce valid indices
        model.eval()
        preds = model.greedy_decode(chars)
        assert preds.shape == (2, model.max_decode_len)
        assert all(0 <= idx < len(tag_vocab) for idx in preds[0].tolist())
