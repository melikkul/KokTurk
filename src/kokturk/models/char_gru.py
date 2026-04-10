"""Character-level GRU Seq2Seq model for morphological atomization.

Encodes surface character sequence, decodes root + ordered tag sequence.
Uses Bahdanau (additive) attention and teacher forcing with scheduled decay.

This is the neural draft model for Phase 2 active learning. The final
production model (Phase 3) will use a Char Transformer with cross-attention
to distilled BERTurk context embeddings.

Example:
    "evlerinden" → encoder → decoder → "ev +PLU +POSS.3SG +ABL"
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.nn import functional

if TYPE_CHECKING:
    from kokturk.models.morphotactic_mask import MorphotacticMask


class BahdanauAttention(nn.Module):
    """Additive attention (Bahdanau et al., 2015)."""

    def __init__(self, decoder_dim: int, encoder_dim: int, attn_dim: int = 64) -> None:
        super().__init__()
        self.W_dec = nn.Linear(decoder_dim, attn_dim, bias=False)
        self.W_enc = nn.Linear(encoder_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attention context and weights.

        Args:
            decoder_hidden: (B, decoder_dim)
            encoder_outputs: (B, L, encoder_dim)

        Returns:
            context: (B, encoder_dim)
            weights: (B, L)
        """
        scores = self.v(torch.tanh(
            self.W_dec(decoder_hidden).unsqueeze(1) + self.W_enc(encoder_outputs)
        ))  # (B, L, 1)
        weights = functional.softmax(scores, dim=1)  # (B, L, 1)
        context = (weights * encoder_outputs).sum(dim=1)  # (B, encoder_dim)
        return context, weights.squeeze(-1)


class CharGRUEncoder(nn.Module):
    """Bidirectional GRU over character embeddings."""

    def __init__(
        self,
        char_vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        variational_dropout: float = 0.0,
        weight_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.char_embed = nn.Embedding(
            char_vocab_size, embed_dim, padding_idx=0,
        )
        gru = nn.GRU(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        if weight_dropout > 0.0:
            from kokturk.models.variational_dropout import WeightDropout
            w_names = [f"weight_hh_l{i}" for i in range(num_layers)]
            w_names += [f"weight_hh_l{i}_reverse" for i in range(num_layers)]
            self.gru = WeightDropout(gru, w_names, dropout=weight_dropout)
        else:
            self.gru = gru
        self.dropout = nn.Dropout(dropout)
        if variational_dropout > 0.0:
            from kokturk.models.variational_dropout import VariationalDropout
            self.vdrop = VariationalDropout(p=variational_dropout)
        else:
            self.vdrop = None  # type: ignore[assignment]

    def forward(
        self, chars: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode character sequence.

        Args:
            chars: (B, L) character indices.

        Returns:
            outputs: (B, L, 2*hidden_dim) encoder outputs.
            hidden: (num_layers, B, hidden_dim) merged hidden state.
        """
        embedded = self.dropout(self.char_embed(chars))
        if self.vdrop is not None:
            embedded = self.vdrop(embedded)
        outputs, hidden = self.gru(embedded)
        if self.vdrop is not None:
            outputs = self.vdrop(outputs)
        # Merge bidirectional: (2*num_layers, B, H) → (num_layers, B, H)
        # Reshape from [fwd_0, bwd_0, fwd_1, bwd_1, ...] to sum pairs
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)
        hidden = hidden.sum(dim=1)  # (num_layers, B, hidden_dim)
        return outputs, hidden


class TagDecoder(nn.Module):
    """Autoregressive GRU decoder with Bahdanau attention."""

    def __init__(
        self,
        tag_vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        encoder_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        weight_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.tag_embed = nn.Embedding(
            tag_vocab_size, embed_dim, padding_idx=0,
        )
        self.attention = BahdanauAttention(hidden_dim, encoder_dim)
        gru = nn.GRU(
            embed_dim + encoder_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        if weight_dropout > 0.0:
            from kokturk.models.variational_dropout import WeightDropout
            w_names = [f"weight_hh_l{i}" for i in range(num_layers)]
            self.gru = WeightDropout(gru, w_names, dropout=weight_dropout)
        else:
            self.gru = gru
        self.output_proj = nn.Linear(hidden_dim, tag_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward_step(
        self,
        prev_tag: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single decoding step.

        Args:
            prev_tag: (B, 1) previous tag index.
            hidden: (num_layers, B, hidden_dim) decoder hidden state.
            encoder_outputs: (B, L, encoder_dim) from encoder.

        Returns:
            logits: (B, 1, tag_vocab_size)
            hidden: updated hidden state
            attn_weights: (B, L) attention weights
        """
        embedded = self.dropout(self.tag_embed(prev_tag))  # (B, 1, E)
        context, attn_weights = self.attention(
            hidden[-1], encoder_outputs
        )  # (B, encoder_dim), (B, L)
        gru_input = torch.cat(
            [embedded, context.unsqueeze(1)], dim=-1
        )  # (B, 1, E + encoder_dim)
        output, hidden = self.gru(gru_input, hidden)
        logits = self.output_proj(output)  # (B, 1, V)
        return logits, hidden, attn_weights


class MorphAtomizer(nn.Module):
    """Character-level GRU Seq2Seq for morphological atomization.

    Encodes surface form character-by-character, decodes root + tag sequence
    with attention over the character representations.

    Args:
        char_vocab_size: Number of characters in vocabulary.
        tag_vocab_size: Number of output tags (roots + suffixes).
        embed_dim: Embedding dimension for both chars and tags.
        hidden_dim: GRU hidden state dimension.
        num_layers: Number of GRU layers.
        dropout: Dropout probability.
        max_decode_len: Maximum decoding length.
        sos_idx: Start-of-sequence token index.
        eos_idx: End-of-sequence token index.
    """

    def __init__(
        self,
        char_vocab_size: int,
        tag_vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        max_decode_len: int = 15,
        sos_idx: int = 1,
        eos_idx: int = 2,
        variational_dropout: float = 0.0,
        weight_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = CharGRUEncoder(
            char_vocab_size, embed_dim, hidden_dim, num_layers, dropout,
            variational_dropout=variational_dropout,
            weight_dropout=weight_dropout,
        )
        self.decoder = TagDecoder(
            tag_vocab_size, embed_dim, hidden_dim,
            encoder_dim=2 * hidden_dim,
            num_layers=num_layers, dropout=dropout,
            weight_dropout=weight_dropout,
        )
        self.tag_vocab_size = tag_vocab_size
        self.max_decode_len = max_decode_len
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def forward(
        self,
        chars: torch.Tensor,
        target_tags: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """Forward pass with optional teacher forcing.

        Args:
            chars: (B, L_char) character indices.
            target_tags: (B, L_tag) target tag indices (for training).
            teacher_forcing_ratio: Probability of using ground truth as
                next decoder input. 0.0 = greedy decode.

        Returns:
            logits: (B, decode_len, tag_vocab_size)
        """
        batch_size = chars.size(0)
        encoder_outputs, hidden = self.encoder(chars)

        decode_len = (
            target_tags.size(1) if target_tags is not None
            else self.max_decode_len
        )

        all_logits: list[torch.Tensor] = []
        prev_tag = torch.full(
            (batch_size, 1), self.sos_idx,
            dtype=torch.long, device=chars.device,
        )

        for t in range(decode_len):
            logits, hidden, _ = self.decoder.forward_step(
                prev_tag, hidden, encoder_outputs,
            )
            all_logits.append(logits)

            if target_tags is not None and random.random() < teacher_forcing_ratio:
                prev_tag = target_tags[:, t : t + 1]
            else:
                prev_tag = logits.argmax(dim=-1)

        return torch.cat(all_logits, dim=1)  # (B, decode_len, V)

    @torch.no_grad()
    def greedy_decode(
        self,
        chars: torch.Tensor,
        morphotactic_mask: "MorphotacticMask | None" = None,
    ) -> torch.Tensor:
        """Greedy decoding without teacher forcing.

        Args:
            chars: (B, L_char) character indices.
            morphotactic_mask: Optional :class:`MorphotacticMask` instance.
                When supplied, illegal next-tag transitions are masked to
                ``-inf`` before each argmax. Default ``None`` preserves
                existing behaviour bit-for-bit.

        Returns:
            predicted: (B, max_decode_len) predicted tag indices.
        """
        self.eval()
        batch_size = chars.size(0)
        encoder_outputs, hidden = self.encoder(chars)

        prev_tag = torch.full(
            (batch_size, 1), self.sos_idx,
            dtype=torch.long, device=chars.device,
        )
        predictions: list[torch.Tensor] = []

        if morphotactic_mask is not None:
            morphotactic_mask.reset(batch_size)

        for _ in range(self.max_decode_len):
            logits, hidden, _ = self.decoder.forward_step(
                prev_tag, hidden, encoder_outputs,
            )
            if morphotactic_mask is not None:
                # logits is (B, 1, V); squeeze step dim, mask, restore.
                step_logits = logits.squeeze(1)
                allowed = morphotactic_mask.get_mask()
                step_logits = step_logits.masked_fill(~allowed, float("-inf"))
                pred = step_logits.argmax(dim=-1, keepdim=True)  # (B, 1)
                morphotactic_mask.update(pred)
            else:
                pred = logits.argmax(dim=-1)  # (B, 1)
            predictions.append(pred)
            prev_tag = pred

        return torch.cat(predictions, dim=1)  # (B, max_decode_len)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
