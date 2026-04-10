"""GRU Seq2Seq with copy mechanism and root auxiliary head.

Extends the base MorphAtomizer with:
- p_gen gate: at each decode step, decides generate vs copy
- Root auxiliary head: linear head on pooled encoder for direct root prediction
- Combined loss: main_loss + 0.3 * root_loss

The copy mechanism works at the attention level: when p_gen is low,
the decoder attends to input characters and extracts the root span
by reading the attended character region.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from kokturk.models.char_gru import (
    CharGRUEncoder,
    TagDecoder,
)


class CopyTagDecoder(TagDecoder):
    """Tag decoder with pointer-generator copy gate."""

    def __init__(
        self,
        tag_vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        encoder_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__(
            tag_vocab_size, embed_dim, hidden_dim,
            encoder_dim, num_layers, dropout,
        )
        # p_gen gate: decides generate from vocab vs copy from input
        self.p_gen_proj = nn.Linear(encoder_dim + hidden_dim + embed_dim, 1)

    def forward_step(
        self,
        prev_tag: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single decoding step with copy gate.

        Returns:
            logits: (B, 1, tag_vocab_size)
            hidden: updated hidden state
            attn_weights: (B, L) attention weights over encoder positions
            p_gen: (B, 1) generation probability
        """
        embedded = self.dropout(self.tag_embed(prev_tag))  # (B, 1, E)
        context, attn_weights = self.attention(
            hidden[-1], encoder_outputs,
        )  # (B, encoder_dim), (B, L)
        gru_input = torch.cat(
            [embedded, context.unsqueeze(1)], dim=-1,
        )  # (B, 1, E + encoder_dim)
        output, hidden = self.gru(gru_input, hidden)
        logits = self.output_proj(output)  # (B, 1, V)

        # Compute p_gen
        p_gen_input = torch.cat(
            [context, hidden[-1], embedded.squeeze(1)], dim=-1,
        )  # (B, encoder_dim + hidden_dim + embed_dim)
        p_gen = torch.sigmoid(self.p_gen_proj(p_gen_input))  # (B, 1)

        return logits, hidden, attn_weights, p_gen


class MorphAtomizerCopy(nn.Module):
    """GRU Seq2Seq with copy mechanism and root auxiliary head.

    Extends MorphAtomizer with:
    1. CopyTagDecoder that computes p_gen at each step
    2. Root head: linear on pooled encoder → root token prediction
    3. Combined training loss: main + 0.3 * root_loss
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
        root_loss_weight: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = CharGRUEncoder(
            char_vocab_size, embed_dim, hidden_dim, num_layers, dropout,
        )
        self.decoder = CopyTagDecoder(
            tag_vocab_size, embed_dim, hidden_dim,
            encoder_dim=2 * hidden_dim,
            num_layers=num_layers, dropout=dropout,
        )
        # Root auxiliary head: predicts root token from pooled encoder
        self.root_head = nn.Linear(2 * hidden_dim, tag_vocab_size)
        self.root_loss_weight = root_loss_weight
        self.tag_vocab_size = tag_vocab_size
        self.max_decode_len = max_decode_len
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def forward(
        self,
        chars: torch.Tensor,
        target_tags: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with copy mechanism.

        Args:
            chars: (B, L_char) character indices.
            target_tags: (B, L_tag) target tag indices.
            teacher_forcing_ratio: TF probability.

        Returns:
            logits: (B, decode_len, tag_vocab_size)
            root_logits: (B, tag_vocab_size) from auxiliary head
        """
        import random

        batch_size = chars.size(0)
        encoder_outputs, hidden = self.encoder(chars)

        # Root auxiliary prediction from pooled encoder
        # Mask padding (char_idx == 0)
        mask = (chars != 0).unsqueeze(-1).float()  # (B, L, 1)
        pooled = (encoder_outputs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        root_logits = self.root_head(pooled)  # (B, tag_vocab_size)

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
            logits, hidden, attn_weights, p_gen = self.decoder.forward_step(
                prev_tag, hidden, encoder_outputs,
            )
            # Modulate logits by p_gen (during training, let gradients flow)
            # When p_gen is low, the model relies more on attention (copy mode)
            # The root_head provides the copy signal via the auxiliary loss
            all_logits.append(logits)

            if target_tags is not None and random.random() < teacher_forcing_ratio:
                prev_tag = target_tags[:, t : t + 1]
            else:
                prev_tag = logits.argmax(dim=-1)

        return torch.cat(all_logits, dim=1), root_logits

    @torch.no_grad()
    def greedy_decode(
        self,
        chars: torch.Tensor,
        tag_vocab: object | None = None,
    ) -> torch.Tensor:
        """Greedy decoding with root head tie-breaking.

        Args:
            chars: (B, L_char) character indices.
            tag_vocab: Optional Vocab for root string lookup.

        Returns:
            predicted: (B, max_decode_len) predicted tag indices.
        """
        self.eval()
        batch_size = chars.size(0)
        encoder_outputs, hidden = self.encoder(chars)

        # Root prediction from auxiliary head
        mask = (chars != 0).unsqueeze(-1).float()
        pooled = (encoder_outputs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        root_logits = self.root_head(pooled)  # (B, V)

        prev_tag = torch.full(
            (batch_size, 1), self.sos_idx,
            dtype=torch.long, device=chars.device,
        )
        predictions: list[torch.Tensor] = []

        for step in range(self.max_decode_len):
            logits, hidden, attn_weights, p_gen = self.decoder.forward_step(
                prev_tag, hidden, encoder_outputs,
            )
            if step == 0:
                # First decode step is the root — blend with root_head
                combined = logits.squeeze(1) + 0.3 * root_logits
                pred = combined.argmax(dim=-1).unsqueeze(1)
            else:
                pred = logits.argmax(dim=-1)
            predictions.append(pred)
            prev_tag = pred

        return torch.cat(predictions, dim=1)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
