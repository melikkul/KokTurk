"""Tests for MorphotacticMask (Cat B Task 3)."""

from __future__ import annotations

import torch

from kokturk.core.constants import ZEYREK_TO_CANONICAL
from kokturk.models.morphotactic_mask import (
    TAG_TO_CATEGORY,
    MorphotacticMask,
    MorphState,
    assert_full_coverage,
)


def _build_vocab(tags: list[str]) -> dict[str, int]:
    return {t: i for i, t in enumerate(tags)}


def test_full_canonical_coverage():
    """Every canonical tag in constants.py must have a category."""
    assert_full_coverage()
    canonical = {v for v in ZEYREK_TO_CANONICAL.values() if v}
    for tag in canonical:
        assert tag in TAG_TO_CATEGORY, f"missing: {tag}"


def test_legal_chain_passes():
    vocab_tags = ["<PAD>", "<SOS>", "<EOS>", "+PLU", "+POSS.3SG", "+ABL"]
    mask = MorphotacticMask(_build_vocab(vocab_tags))
    mask.reset(batch_size=1)

    # +PLU (NUMBER) — allowed in START
    allowed = mask.get_mask()[0]
    assert allowed[3].item() is True
    mask.update(torch.tensor([3]))

    # +POSS.3SG (POSSESSIVE) — allowed after NUMBER
    allowed = mask.get_mask()[0]
    assert allowed[4].item() is True
    mask.update(torch.tensor([4]))

    # +ABL (CASE) — allowed after POSSESSIVE
    allowed = mask.get_mask()[0]
    assert allowed[5].item() is True
    mask.update(torch.tensor([5]))

    # EOS always allowed
    allowed = mask.get_mask()[0]
    assert allowed[2].item() is True


def test_case_after_case_blocked():
    vocab_tags = ["<PAD>", "<SOS>", "<EOS>", "+ABL", "+DAT"]
    mask = MorphotacticMask(_build_vocab(vocab_tags))
    mask.reset(batch_size=1)

    mask.update(torch.tensor([3]))  # +ABL (CASE)
    allowed = mask.get_mask()[0]
    # +DAT is also CASE — must be blocked after we've consumed +ABL.
    assert allowed[4].item() is False
    # EOS still allowed.
    assert allowed[2].item() is True


def test_inflectional_before_derivational_allowed_via_state():
    """+NESS (DERIV) after +PLU (NUMBER) — DERIV reset is allowed in our FSA."""
    vocab_tags = ["<PAD>", "<SOS>", "<EOS>", "+PLU", "+NESS"]
    mask = MorphotacticMask(_build_vocab(vocab_tags))
    mask.reset(batch_size=1)

    mask.update(torch.tensor([3]))  # +PLU
    allowed = mask.get_mask()[0]
    # +NESS is DERIV which is allowed from NUMBER (we permit DERIV resets).
    assert allowed[4].item() is True


def test_pad_never_allowed():
    vocab_tags = ["<PAD>", "<SOS>", "<EOS>", "+PLU"]
    mask = MorphotacticMask(_build_vocab(vocab_tags))
    mask.reset(batch_size=1)
    allowed = mask.get_mask()[0]
    assert allowed[0].item() is False  # PAD


def test_sos_only_allowed_in_start():
    vocab_tags = ["<PAD>", "<SOS>", "<EOS>", "+PLU"]
    mask = MorphotacticMask(_build_vocab(vocab_tags))
    mask.reset(batch_size=1)
    # SOS in START
    allowed = mask.get_mask()[0]
    assert allowed[1].item() is True
    # advance — SOS now blocked
    mask.update(torch.tensor([3]))
    allowed = mask.get_mask()[0]
    assert allowed[1].item() is False


def test_batch_independent_state():
    vocab_tags = ["<PAD>", "<SOS>", "<EOS>", "+PLU", "+ABL", "+DAT"]
    mask = MorphotacticMask(_build_vocab(vocab_tags))
    mask.reset(batch_size=2)

    # Sample 0 takes +ABL, sample 1 takes +PLU.
    mask.update(torch.tensor([4, 3]))
    allowed = mask.get_mask()
    # Sample 0 (now CASE) cannot take +DAT.
    assert allowed[0, 5].item() is False
    # Sample 1 (now NUMBER) cannot take +DAT either (DAT is CASE; allowed
    # from NUMBER actually — POSSESSIVE/CASE/DERIV/EOS).
    assert allowed[1, 5].item() is True


def test_reset_between_words():
    vocab_tags = ["<PAD>", "<SOS>", "<EOS>", "+ABL", "+DAT"]
    mask = MorphotacticMask(_build_vocab(vocab_tags))
    mask.reset(batch_size=1)
    mask.update(torch.tensor([3]))  # +ABL → CASE
    assert mask.get_mask()[0, 4].item() is False
    mask.reset(batch_size=1)
    # Fresh START → +DAT allowed again.
    assert mask.get_mask()[0, 4].item() is True


def test_voice_to_deriv_allowed():
    """gel+CAUS+INF — derivational after voice must be permitted."""
    vocab_tags = ["<PAD>", "<SOS>", "<EOS>", "+CAUS", "+INF"]
    mask = MorphotacticMask(_build_vocab(vocab_tags))
    mask.reset(batch_size=1)
    mask.update(torch.tensor([3]))  # +CAUS → VOICE
    allowed = mask.get_mask()[0]
    assert allowed[4].item() is True  # +INF (DERIV) allowed from VOICE
