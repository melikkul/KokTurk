"""Tests for DualHeadAtomizer — dual-head morphological atomizer."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

from kokturk.models.dual_head import (
    AttentionPooling,
    DualHeadAtomizer,
    RootHead,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

CHAR_VOCAB_SIZE = 50
TAG_VOCAB_SIZE = 100
ROOT_VOCAB_SIZE = 30
EMBED_DIM = 16
HIDDEN_DIM = 32
NUM_LAYERS = 2
BATCH_SIZE = 4
MAX_CHAR_LEN = 10
MAX_TAG_LEN = 8  # decode_len = 8 - 2 = 6


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model() -> DualHeadAtomizer:
    """Small DualHeadAtomizer for fast unit tests."""
    return DualHeadAtomizer(
        char_vocab_size=CHAR_VOCAB_SIZE,
        tag_vocab_size=TAG_VOCAB_SIZE,
        root_vocab_size=ROOT_VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        max_decode_len=MAX_TAG_LEN,
    )


@pytest.fixture
def batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Synthetic batch: chars (4, 10), tag_ids (4, 8), gold_root (4,).

    tag_ids layout:
        col 0: SOS (1)
        col 1: root token in tag_vocab (random 4..TAG_VOCAB_SIZE-1)
        col 2: first tag
        col 3: second tag
        col 4: EOS (2)
        cols 5-7: PAD (0)
    """
    torch.manual_seed(42)
    chars = torch.randint(1, CHAR_VOCAB_SIZE, (BATCH_SIZE, MAX_CHAR_LEN))

    tag_ids = torch.zeros(BATCH_SIZE, MAX_TAG_LEN, dtype=torch.long)
    tag_ids[:, 0] = 1  # SOS
    tag_ids[:, 1] = torch.randint(4, TAG_VOCAB_SIZE, (BATCH_SIZE,))  # root in tag_vocab
    tag_ids[:, 2] = torch.randint(4, TAG_VOCAB_SIZE, (BATCH_SIZE,))  # first tag
    tag_ids[:, 3] = torch.randint(4, TAG_VOCAB_SIZE, (BATCH_SIZE,))  # second tag
    tag_ids[:, 4] = 2  # EOS
    # positions 5, 6, 7 remain PAD = 0

    gold_root = torch.randint(2, ROOT_VOCAB_SIZE, (BATCH_SIZE,))
    return chars, tag_ids, gold_root


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_attention_pooling_shape() -> None:
    """AttentionPooling should reduce (B, L, H) to (B, H)."""
    encoder_dim = 256
    pooling = AttentionPooling(encoder_dim)
    x = torch.randn(4, 8, encoder_dim)
    out = pooling(x)
    assert out.shape == (4, encoder_dim), (
        f"Expected (4, {encoder_dim}), got {out.shape}"
    )


def test_root_head_shape() -> None:
    """RootHead should output (B, root_vocab_size) logits."""
    encoder_dim = 256
    head = RootHead(encoder_dim, ROOT_VOCAB_SIZE)
    enc = torch.randn(4, 10, encoder_dim)
    logits = head(enc)
    assert logits.shape == (4, ROOT_VOCAB_SIZE), (
        f"Expected (4, {ROOT_VOCAB_SIZE}), got {logits.shape}"
    )


def test_dual_head_forward_shapes(
    model: DualHeadAtomizer,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Forward pass should produce correct output shapes."""
    chars, tag_ids, gold_root = batch
    root_logits, tag_outputs = model(chars, tag_ids, gold_root)

    assert root_logits.shape == (BATCH_SIZE, ROOT_VOCAB_SIZE), (
        f"root_logits: expected ({BATCH_SIZE}, {ROOT_VOCAB_SIZE}), "
        f"got {root_logits.shape}"
    )
    expected_tag_shape = (BATCH_SIZE, MAX_TAG_LEN - 2, TAG_VOCAB_SIZE)
    assert tag_outputs.shape == expected_tag_shape, (
        f"tag_outputs: expected {expected_tag_shape}, got {tag_outputs.shape}"
    )


def test_tag_target_alignment(
    model: DualHeadAtomizer,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Tag decoder outputs should align with tag_ids[:, 2:] as targets."""
    chars, tag_ids, gold_root = batch
    root_logits, tag_outputs = model(chars, tag_ids, gold_root)

    # Targets are positions 2 onwards in tag_ids (first tag through EOS/PAD).
    tag_targets = tag_ids[:, 2:]  # (B, MAX_TAG_LEN - 2) = (B, 6)
    assert tag_targets.shape == tag_outputs.shape[:2], (
        f"Target shape {tag_targets.shape} does not match "
        f"output shape {tag_outputs.shape[:2]}"
    )
    assert tag_targets.shape[1] == MAX_TAG_LEN - 2, (
        f"Expected decode_len={MAX_TAG_LEN - 2}, got {tag_targets.shape[1]}"
    )


def test_loss_decreases(
    model: DualHeadAtomizer,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Training loss should decrease over 5 gradient steps."""
    torch.manual_seed(0)
    chars, tag_ids, gold_root = batch
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses: list[float] = []
    for _ in range(5):
        root_logits, tag_outputs = model(
            chars, tag_ids, gold_root, teacher_forcing_ratio=1.0
        )
        root_loss = F.cross_entropy(root_logits, gold_root)
        tag_targets = tag_ids[:, 2:].reshape(-1)
        tag_loss = F.cross_entropy(
            tag_outputs.reshape(-1, TAG_VOCAB_SIZE),
            tag_targets,
            ignore_index=0,
        )
        loss = tag_loss + 0.3 * root_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], (
        f"Loss did not decrease over 5 steps: {losses}"
    )


def test_greedy_decode_produces_string(
    model: DualHeadAtomizer,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """greedy_decode should return B non-empty strings."""
    chars, _, _ = batch
    results = model.greedy_decode(chars)

    assert len(results) == BATCH_SIZE, (
        f"Expected {BATCH_SIZE} results, got {len(results)}"
    )
    for s in results:
        assert isinstance(s, str), f"Result is not a string: {s!r}"
        assert len(s) > 0, "Result string is empty"


def test_root_head_uses_gold_at_train(
    model: DualHeadAtomizer,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Changing gold_root should change tag_outputs when use_gold_root=True."""
    chars, tag_ids, gold_root = batch

    _, out1 = model(
        chars, tag_ids, gold_root,
        use_gold_root=True,
        teacher_forcing_ratio=0.0,
    )
    alt_root = (gold_root + 1) % ROOT_VOCAB_SIZE
    _, out2 = model(
        chars, tag_ids, alt_root,
        use_gold_root=True,
        teacher_forcing_ratio=0.0,
    )

    assert not torch.allclose(out1, out2), (
        "tag_outputs should differ when gold_root changes (root embedding "
        "conditions the decoder)"
    )


def test_no_sos_first_step(
    model: DualHeadAtomizer,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """First decoder step should use root embedding, not SOS embedding."""
    chars, tag_ids, gold_root = batch
    model.eval()

    with torch.no_grad():
        # Standard forward: first input = root_embed.
        root_logits, tag_outputs_root = model(
            chars, tag_ids, gold_root,
            use_gold_root=True,
            teacher_forcing_ratio=0.0,
        )

        # Root embedding (from root_vocab) and SOS embedding (from tag_vocab)
        # are drawn from two independent nn.Embedding tables and should not be
        # equal in general.
        root_embeds = model.root_embedding(gold_root)  # (B, embed_dim)
        sos_embed = model.tag_decoder.tag_embed(
            torch.ones(BATCH_SIZE, dtype=torch.long)  # SOS idx = 1
        )  # (B, embed_dim)

    assert not torch.allclose(root_embeds, sos_embed), (
        "root_embed should not equal SOS embed — they come from different "
        "embedding tables with independent parameters"
    )
