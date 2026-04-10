"""Morphotactic Finite-State Constraint Mask for the GRU decoder.

Turkish suffix ordering follows strict rules. The standard schema is::

    ROOT → [DERIVATIONAL]* → [INFLECTIONAL]

with nominal inflection ordered ``NUMBER → POSSESSIVE → CASE`` and verbal
inflection ordered ``VOICE → POLARITY → TENSE → MODALITY → AGREEMENT``.

This module encodes those rules as a small finite-state automaton and
exposes :class:`MorphotacticMask` which produces a per-step boolean mask
over the tag vocabulary. The mask multiplies into the decoder logits
(``logits[~allowed] = -inf``) so that illegal next-tag predictions cannot
be selected.

The mask is **opt-in** — pass an instance of :class:`MorphotacticMask`
into ``MorphAtomizer.greedy_decode(...)`` or
``DualHeadAtomizer.greedy_decode(...)`` via the ``morphotactic_mask``
keyword. Default behaviour (no mask) is bit-for-bit unchanged.
"""

from __future__ import annotations

from enum import IntEnum

import torch

from kokturk.core.constants import ZEYREK_TO_CANONICAL


class MorphState(IntEnum):
    """States in the morphotactic automaton."""

    START = 0
    DERIVATIONAL = 1
    NUMBER = 2
    POSSESSIVE = 3
    CASE = 4  # terminal except for EOS
    VOICE = 5
    POLARITY = 6
    TENSE = 7
    MODALITY = 8
    AGREEMENT = 9  # terminal except for EOS
    TERMINAL = 10  # only EOS allowed


# State → set of allowed *category* labels.  ``EOS`` is always allowed
# (terminate any time); ``PAD`` is never allowed.
TRANSITIONS: dict[MorphState, set[str]] = {
    MorphState.START: {
        "DERIV", "NUMBER", "POSSESSIVE", "CASE", "VOICE",
        "POLARITY", "TENSE", "MODALITY", "AGREEMENT", "POS", "EOS",
    },
    MorphState.DERIVATIONAL: {
        "DERIV", "NUMBER", "POSSESSIVE", "CASE", "VOICE",
        "POLARITY", "TENSE", "MODALITY", "AGREEMENT", "EOS",
    },
    MorphState.NUMBER: {"POSSESSIVE", "CASE", "DERIV", "EOS"},
    MorphState.POSSESSIVE: {"CASE", "DERIV", "EOS"},
    MorphState.CASE: {"DERIV", "EOS"},
    # VOICE includes DERIV so verbal-noun chains like gel+CAUS+INF work,
    # and includes VOICE itself so causative+passive / double-causative stack.
    MorphState.VOICE: {"VOICE", "DERIV", "POLARITY", "TENSE", "MODALITY", "AGREEMENT", "EOS"},
    MorphState.POLARITY: {"TENSE", "MODALITY", "AGREEMENT", "EOS"},
    MorphState.TENSE: {"MODALITY", "AGREEMENT", "DERIV", "EOS"},
    MorphState.MODALITY: {"AGREEMENT", "EOS"},
    MorphState.AGREEMENT: {"EOS"},
    MorphState.TERMINAL: {"EOS"},
}

# After predicting a tag of a given *category*, advance the FSA to which
# new state?  ``DERIV`` resets the verbal/nominal subchain.
CATEGORY_TO_NEXT_STATE: dict[str, MorphState] = {
    "DERIV": MorphState.DERIVATIONAL,
    "NUMBER": MorphState.NUMBER,
    "POSSESSIVE": MorphState.POSSESSIVE,
    "CASE": MorphState.CASE,
    "VOICE": MorphState.VOICE,
    "POLARITY": MorphState.POLARITY,
    "TENSE": MorphState.TENSE,
    "MODALITY": MorphState.MODALITY,
    "AGREEMENT": MorphState.AGREEMENT,
    "POS": MorphState.START,  # POS tag (e.g. +Noun) doesn't advance
    "EOS": MorphState.TERMINAL,
}


# Canonical-tag → morphotactic-category map.  Built by walking the
# canonical inventory in :mod:`kokturk.core.constants`.
TAG_TO_CATEGORY: dict[str, str] = {
    # POS tags
    "+Noun": "POS", "+Verb": "POS", "+Adj": "POS", "+Adv": "POS",
    "+Det": "POS", "+Pron": "POS", "+Postp": "POS", "+Conj": "POS",
    "+Interj": "POS", "+Num": "POS", "+Ques": "POS", "+Punc": "POS",
    "+Dup": "POS", "+Prop": "POS",
    # Case
    "+NOM": "CASE", "+ACC": "CASE", "+DAT": "CASE", "+LOC": "CASE",
    "+ABL": "CASE", "+GEN": "CASE", "+INS": "CASE", "+EQU": "CASE",
    # Number
    "+PLU": "NUMBER", "+1PL": "AGREEMENT", "+2PL": "AGREEMENT",
    # Possession
    "+POSS.1SG": "POSSESSIVE", "+POSS.2SG": "POSSESSIVE",
    "+POSS.3SG": "POSSESSIVE", "+POSS.1PL": "POSSESSIVE",
    "+POSS.2PL": "POSSESSIVE", "+POSS.3PL": "POSSESSIVE",
    # TAM
    "+PAST": "TENSE", "+EVID": "TENSE", "+AOR": "TENSE",
    "+PROG": "TENSE", "+FUT": "TENSE", "+PRES": "TENSE",
    "+IMP": "MODALITY", "+OPT": "MODALITY", "+DESR": "MODALITY",
    "+NECES": "MODALITY", "+COND": "MODALITY",
    "+ABIL": "VOICE",  # ability is morphologically pre-tense
    "+NEG": "POLARITY",
    # Voice
    "+CAUS": "VOICE", "+PASS": "VOICE", "+RECIP": "VOICE", "+REFLEX": "VOICE",
    # Copula
    "+COP": "TENSE",
    # Derivational
    "+BECOME": "DERIV", "+ACQUIRE": "DERIV", "+DIM": "DERIV",
    "+AGT": "DERIV", "+NESS": "DERIV", "+WITH": "DERIV",
    "+WITHOUT": "DERIV", "+REL": "DERIV", "+FITFOR": "DERIV",
    "+LY": "DERIV", "+INF": "DERIV", "+PASTPART": "DERIV",
    "+FUTPART": "DERIV", "+PRESPART": "DERIV", "+NARRPART": "DERIV",
    "+AORPART": "DERIV", "+NOTSTATE": "DERIV", "+FEELLIKE": "DERIV",
    "+JUSTLIKE": "DERIV", "+ASIF": "DERIV", "+WHILE": "DERIV",
    "+WHEN": "DERIV", "+SINCEDOINGSO": "DERIV", "+BYDOINGSO": "DERIV",
    "+ADAMANTLY": "DERIV", "+AFTERDOINGSO": "DERIV",
    "+WITHOUTDOINGSO": "DERIV", "+ASLONGAS": "DERIV", "+INSTEAD": "DERIV",
    # LVC tags from Cat B Task 1
    "+LVC.ET": "DERIV", "+LVC.OL": "DERIV",
    # Agreement (1sg/2sg/3sg/3pl emitted by some pipelines as bare tags)
    "+1SG": "AGREEMENT", "+2SG": "AGREEMENT", "+3SG": "AGREEMENT",
    "+3PL": "AGREEMENT",
}

# Special tokens that the decoder vocabulary may include.
SPECIAL_SOS = ("<SOS>", "<sos>", "[SOS]", "SOS")
SPECIAL_EOS = ("<EOS>", "<eos>", "[EOS]", "EOS")
SPECIAL_PAD = ("<PAD>", "<pad>", "[PAD]", "PAD")
SPECIAL_UNK = ("<UNK>", "<unk>", "[UNK]", "UNK")


def _category_for(tag: str) -> str:
    """Return the morphotactic category for a vocabulary token.

    Special tokens are routed to synthetic categories:

    * ``SOS`` — only ever fired at decoder t=0; treated as ``POS`` so it
      keeps the FSA in ``START``.
    * ``EOS`` — always allowed; mapped to category ``EOS``.
    * ``PAD`` / ``UNK`` — category ``BLOCK`` (never permitted).
    * Unmapped + tags — default to ``DERIV`` (most permissive) to avoid
      blocking legitimate decoder outputs while we extend the table.
    """
    if tag in SPECIAL_EOS:
        return "EOS"
    if tag in SPECIAL_SOS:
        return "POS"
    if tag in SPECIAL_PAD or tag in SPECIAL_UNK:
        return "BLOCK"
    if tag in TAG_TO_CATEGORY:
        return TAG_TO_CATEGORY[tag]
    return "DERIV"


def assert_full_coverage() -> None:
    """Sanity-check that every canonical tag in constants.py has a category.

    Used by tests; safe to call at import time but kept as a function so
    failures surface as test failures, not import-time crashes.
    """
    canonical_tags = {v for v in ZEYREK_TO_CANONICAL.values() if v}
    missing = canonical_tags - set(TAG_TO_CATEGORY)
    assert not missing, f"Tags missing morphotactic category: {sorted(missing)}"


class MorphotacticMask:
    """Per-step boolean mask over the tag vocabulary.

    Usage::

        mask = MorphotacticMask(tag_vocab)
        mask.reset(batch_size)

        for step in range(max_len):
            logits = decoder(...)                       # (B, V)
            allowed = mask.get_mask()                   # (B, V) bool
            logits = logits.masked_fill(~allowed, float("-inf"))
            pred = logits.argmax(-1)
            mask.update(pred)

    The mask precomputes ``(num_states, vocab_size)`` once at construction
    so per-step lookup is a single index op.
    """

    def __init__(
        self,
        tag_vocab: dict[str, int] | list[str],
        device: str | torch.device = "cpu",
    ) -> None:
        if isinstance(tag_vocab, dict):
            self._idx_to_tag: list[str] = ["" for _ in range(len(tag_vocab))]
            for tag, idx in tag_vocab.items():
                if 0 <= idx < len(self._idx_to_tag):
                    self._idx_to_tag[idx] = tag
        else:
            self._idx_to_tag = list(tag_vocab)

        self._vocab_size = len(self._idx_to_tag)
        self._device = torch.device(device)

        # Precompute per-state allowed mask: (num_states, vocab_size).
        num_states = len(MorphState)
        allowed = torch.zeros(
            (num_states, self._vocab_size), dtype=torch.bool, device=self._device
        )
        # Per-token next state.
        next_state = torch.full(
            (self._vocab_size,), MorphState.START.value, dtype=torch.long,
            device=self._device,
        )

        for v_idx, tag in enumerate(self._idx_to_tag):
            cat = _category_for(tag)
            if cat == "BLOCK":
                continue
            if cat == "EOS":
                # EOS is allowed in every state.
                allowed[:, v_idx] = True
                next_state[v_idx] = MorphState.TERMINAL.value
                continue
            for state in MorphState:
                if cat in TRANSITIONS[state]:
                    allowed[state.value, v_idx] = True
            next_state[v_idx] = CATEGORY_TO_NEXT_STATE.get(
                cat, MorphState.DERIVATIONAL
            ).value

        # SOS only allowed in START.
        for v_idx, tag in enumerate(self._idx_to_tag):
            if tag in SPECIAL_SOS:
                allowed[:, v_idx] = False
                allowed[MorphState.START.value, v_idx] = True
                next_state[v_idx] = MorphState.START.value

        self._allowed = allowed  # (S, V)
        self._next_state = next_state  # (V,)
        self._state: torch.Tensor | None = None  # (B,)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def reset(self, batch_size: int = 1) -> None:
        """Reset the FSA to ``START`` for every item in the batch."""
        self._state = torch.full(
            (batch_size,),
            MorphState.START.value,
            dtype=torch.long,
            device=self._device,
        )

    def get_mask(self) -> torch.Tensor:
        """Return the current ``(batch_size, vocab_size)`` boolean mask."""
        if self._state is None:
            raise RuntimeError("MorphotacticMask.reset() must be called first")
        return self._allowed[self._state]

    def update(self, predicted_tags: torch.Tensor) -> None:
        """Advance the FSA after the given tag predictions.

        Args:
            predicted_tags: ``(batch_size,)`` or ``(batch_size, 1)`` long
                tensor of vocabulary indices.
        """
        if self._state is None:
            raise RuntimeError("MorphotacticMask.reset() must be called first")
        idx = predicted_tags.reshape(-1).to(self._device)
        self._state = self._next_state[idx]
