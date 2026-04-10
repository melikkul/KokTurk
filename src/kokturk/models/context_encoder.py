"""Sentence-level context encoders for contextual morphological disambiguation.

Four encoder types are provided, all inheriting from ``ContextEncoderBase``:

- ``POSBigramContext``: Embed POS tags of ±1 neighboring tokens (~+2-3 EM).
- ``Word2VecContext``: Pre-trained Word2Vec BiGRU over ±2-word window (~+4-6 EM).
- ``BERTurkContext``: Frozen BERTurk feature extractor, 768→context_dim (~+6-8 EM).
- ``SentenceBiGRUContext``: Full sentence BiGRU, extract hidden at target pos (~+5-7 EM).

All encoders expose:
- ``output_dim: int`` property
- ``forward(*args) -> Tensor[B, output_dim]``

Usage in ``ContextualDualHeadAtomizer``::

    enc = Word2VecContext(vocab_size=50000, embed_dim=128, gru_hidden_dim=64)
    context_vec = enc(neighbor_ids)   # (B, 4) → (B, 128)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ContextEncoderBase(nn.Module):
    """Abstract base class for sentence-context encoders.

    All subclasses must implement ``output_dim`` and ``forward``.
    """

    @property
    def output_dim(self) -> int:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# POSBigramContext
# ---------------------------------------------------------------------------

class POSBigramContext(ContextEncoderBase):
    """Embed POS tags of the ±1 neighboring tokens and concatenate.

    The simplest context encoder — embeds two POS tag indices and concatenates
    them.  A PAD index of 0 is used for sentence boundaries.

    Input:  ``pos_ids`` of shape ``(B, 2)`` — ``[left_pos, right_pos]``.
            Index 0 = PAD (no neighbor at boundary).
    Output: ``(B, 2 * pos_embed_dim)``.

    Args:
        num_pos_tags: Number of POS tags in the vocabulary (not counting PAD).
        pos_embed_dim: Embedding dimension per POS tag.
    """

    def __init__(self, num_pos_tags: int, pos_embed_dim: int = 32) -> None:
        super().__init__()
        # +1 for PAD at index 0
        self.pos_embed = nn.Embedding(num_pos_tags + 1, pos_embed_dim, padding_idx=0)
        self._output_dim = 2 * pos_embed_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, pos_ids: torch.Tensor) -> torch.Tensor:
        """Embed and concatenate left/right POS tags.

        Args:
            pos_ids: ``(B, 2)`` tensor of POS indices (left, right).

        Returns:
            ``(B, 2 * pos_embed_dim)`` context vector.
        """
        # pos_ids: (B, 2)
        left = self.pos_embed(pos_ids[:, 0])   # (B, pos_embed_dim)
        right = self.pos_embed(pos_ids[:, 1])  # (B, pos_embed_dim)
        return torch.cat([left, right], dim=-1)  # (B, 2*pos_embed_dim)


# ---------------------------------------------------------------------------
# Word2VecContext
# ---------------------------------------------------------------------------

class Word2VecContext(ContextEncoderBase):
    """Look up pre-trained Word2Vec for ±2 neighbours, then pool with a BiGRU.

    The embedding matrix defaults to random init but can be initialised from a
    pre-trained Word2Vec binary via ``pretrained_weights``.  The matrix is
    frozen (``requires_grad=False``) when pre-trained weights are provided.

    Input:  ``neighbor_ids`` of shape ``(B, 4)`` — word indices for the four
            neighbours ``[w_{t-2}, w_{t-1}, w_{t+1}, w_{t+2}]``.
            Index 0 = PAD (boundary / unknown).
    Output: ``(B, 2 * gru_hidden_dim)``.

    Args:
        vocab_size: Number of words in the vocabulary (index 0 = PAD).
        embed_dim: Word embedding dimension (should match pre-trained if given).
        gru_hidden_dim: GRU hidden state dimension.
        pretrained_weights: Optional ``(vocab_size, embed_dim)`` tensor.
            When provided, the embedding is initialised and frozen.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        gru_hidden_dim: int = 64,
        pretrained_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.word_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_weights is not None:
            self.word_embed.weight.data.copy_(pretrained_weights)
            self.word_embed.weight.requires_grad = False

        self.gru = nn.GRU(
            embed_dim,
            gru_hidden_dim,
            bidirectional=True,
            batch_first=True,
        )
        self._gru_hidden_dim = gru_hidden_dim
        self._output_dim = 2 * gru_hidden_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, neighbor_ids: torch.Tensor) -> torch.Tensor:
        """Encode 4-word window with BiGRU and return final hidden state.

        Args:
            neighbor_ids: ``(B, 4)`` word indices for the four neighbours.

        Returns:
            ``(B, 2 * gru_hidden_dim)`` context vector.
        """
        emb = self.word_embed(neighbor_ids)  # (B, 4, embed_dim)
        _, hidden = self.gru(emb)            # hidden: (2, B, gru_hidden_dim)
        # Concat fwd and bwd final hidden states
        return torch.cat([hidden[0], hidden[1]], dim=-1)  # (B, 2*gru_hidden_dim)


# ---------------------------------------------------------------------------
# BERTurkContext
# ---------------------------------------------------------------------------

class BERTurkContext(ContextEncoderBase):
    """Frozen BERTurk feature extractor — projects target-token embedding.

    The BERT model is kept frozen (all parameters ``requires_grad=False``).
    For each sentence in the batch the target word's first subword token
    embedding is extracted and projected to ``context_dim``.

    To avoid download failures in CI, pass ``bert_model`` and ``tokenizer``
    directly (useful for testing with mocks).

    Input:
        - ``sentence_texts``: list of B raw sentence strings.
        - ``target_word_positions``: list/tensor of B integer word positions
          (0-based) indicating which word in the sentence is the target.

    Output: ``(B, context_dim)``.

    Args:
        bert_path: Path to a local BERTurk save (created by
            ``scripts/download_berturk.py``).  Ignored when ``bert_model``
            is provided directly.
        context_dim: Output dimension after the linear projection from 768.
        bert_model: Optional pre-constructed BERT model (inject for testing).
        tokenizer: Optional pre-constructed tokenizer (inject for testing).
    """

    def __init__(
        self,
        bert_path: str = "models/berturk",
        context_dim: int = 128,
        bert_model: object | None = None,
        tokenizer: object | None = None,
    ) -> None:
        super().__init__()
        self._context_dim = context_dim

        if bert_model is not None:
            self.bert = bert_model  # type: ignore[assignment]
        else:
            from transformers import AutoModel  # type: ignore[import]
            self.bert = AutoModel.from_pretrained(bert_path)
            for param in self.bert.parameters():
                param.requires_grad = False

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            from transformers import AutoTokenizer  # type: ignore[import]
            self.tokenizer = AutoTokenizer.from_pretrained(bert_path)

        self.proj = nn.Linear(768, context_dim)

    @property
    def output_dim(self) -> int:
        return self._context_dim

    def forward(
        self,
        sentence_texts: list[str],
        target_word_positions: list[int] | torch.Tensor,
    ) -> torch.Tensor:
        """Encode sentences with BERTurk and extract per-token embeddings.

        Args:
            sentence_texts: List of B raw sentence strings.
            target_word_positions: B integer word positions (0-based, word-level).

        Returns:
            ``(B, context_dim)`` context tensor.
        """
        device = next(self.proj.parameters()).device
        B = len(sentence_texts)

        # Tokenise batch
        encoding = self.tokenizer(
            sentence_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # Forward through frozen BERT
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        hidden = outputs.last_hidden_state  # (B, L, 768)

        if isinstance(target_word_positions, torch.Tensor):
            positions = target_word_positions.tolist()
        else:
            positions = list(target_word_positions)

        # Extract embedding at the first subword token for each target word
        target_embeds: list[torch.Tensor] = []
        for b in range(B):
            try:
                word_ids_b = encoding.word_ids(b)
                subword_idx = next(
                    (i for i, wid in enumerate(word_ids_b) if wid == int(positions[b])),
                    1,  # fallback: first non-special token
                )
            except Exception:
                # Fallback if fast-tokenizer word_ids() not available
                subword_idx = min(int(positions[b]) + 1, hidden.size(1) - 1)
            target_embeds.append(hidden[b, subword_idx])

        target_hidden = torch.stack(target_embeds)  # (B, 768)
        return self.proj(target_hidden)             # (B, context_dim)


# ---------------------------------------------------------------------------
# SentenceBiGRUContext
# ---------------------------------------------------------------------------

class SentenceBiGRUContext(ContextEncoderBase):
    """Word-embedding BiGRU over the full sentence; extract hidden at target position.

    Unlike ``Word2VecContext`` which uses a fixed-size window, this encoder runs
    a bidirectional GRU over the entire sentence and plucks out the hidden state
    at the target word's position, giving full left/right context.

    Input:
        - ``word_ids``: ``(B, S)`` sentence word indices (0 = PAD).
        - ``target_positions``: ``(B,)`` 0-based positions within the sentence.
    Output: ``(B, 2 * hidden_dim)``.

    Args:
        vocab_size: Number of words in vocabulary (index 0 = PAD).
        embed_dim: Word embedding dimension.
        hidden_dim: GRU hidden state dimension (per direction).
        dropout: Dropout probability on embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.word_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim, hidden_dim,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self._hidden_dim = hidden_dim
        self._output_dim = 2 * hidden_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        word_ids: torch.Tensor,
        target_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Encode sentence and extract hidden state at target position.

        Args:
            word_ids: ``(B, S)`` sentence word indices.
            target_positions: ``(B,)`` target word positions (clamped to [0, S-1]).

        Returns:
            ``(B, 2 * hidden_dim)`` context vector.
        """
        emb = self.dropout(self.word_embed(word_ids))   # (B, S, embed_dim)
        outputs, _ = self.gru(emb)                      # (B, S, 2*hidden_dim)

        B, S, D = outputs.shape
        pos = target_positions.long().clamp(0, S - 1)   # (B,)
        # Gather hidden state at target position: outputs[b, pos[b], :]
        pos_exp = pos.view(B, 1, 1).expand(B, 1, D)
        target_hidden = outputs.gather(1, pos_exp).squeeze(1)  # (B, 2*hidden_dim)
        return target_hidden
