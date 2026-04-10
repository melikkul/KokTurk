"""Dual-head morphological atomizer for Turkish.

Separates root classification (single-pass, no autoregression) from tag
sequence decoding (autoregressive GRU conditioned on the root embedding).
This eliminates root-error propagation that plagues purely seq2seq models.

Example::

    model = DualHeadAtomizer(
        char_vocab_size=100,
        tag_vocab_size=7807,
        root_vocab_size=7732,
    )
    root_logits, tag_outputs = model(chars, tag_ids, gold_root)
    # root_logits: (B, root_vocab_size)
    # tag_outputs: (B, decode_len, tag_vocab_size)
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from kokturk.models.morphotactic_mask import MorphotacticMask

from kokturk.models.char_gru import BahdanauAttention, CharGRUEncoder


class AttentionPooling(nn.Module):
    """Attention-weighted pooling over the sequence dimension.

    Args:
        hidden_dim: Dimension of the encoder outputs
            (``2 * gru_hidden_dim`` for a bidirectional encoder).
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute attention-weighted sum over the sequence.

        Args:
            encoder_outputs: ``(B, L, hidden_dim)`` encoder hidden states.
            mask: ``(B, L)`` boolean mask; ``True`` marks real (non-padding)
                positions.  If ``None`` all positions are treated as real.

        Returns:
            pooled: ``(B, hidden_dim)`` weighted sum of encoder outputs.
        """
        # scores: (B, L, 1)
        scores = self.score(encoder_outputs)

        if mask is not None:
            # Set padding positions to a large negative value before softmax.
            scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))

        weights = torch.softmax(scores, dim=1)  # (B, L, 1)
        pooled = (weights * encoder_outputs).sum(dim=1)  # (B, hidden_dim)
        return pooled


class RootHead(nn.Module):
    """Classifies the morphological root in a single forward pass.

    No autoregressive generation — eliminates error propagation for roots.
    Architecture: ``AttentionPooling → Dropout → Linear → ReLU → Dropout →
    Linear``.

    Args:
        encoder_dim: Encoder output dimension (``2 * gru_hidden_dim``).
        root_vocab_size: Number of unique roots in the root vocabulary.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        encoder_dim: int,
        root_vocab_size: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.pooling = AttentionPooling(encoder_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(encoder_dim, encoder_dim // 2)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(encoder_dim // 2, root_vocab_size)

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Classify root from encoder outputs.

        Args:
            encoder_outputs: ``(B, L, encoder_dim)`` from the char encoder.
            mask: ``(B, L)`` boolean mask; ``True`` for real tokens.

        Returns:
            root_logits: ``(B, root_vocab_size)`` unnormalised log-probabilities.
        """
        pooled = self.pooling(encoder_outputs, mask)  # (B, encoder_dim)
        x = self.dropout1(pooled)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)  # (B, root_vocab_size)


class ConditionalTagDecoder(nn.Module):
    """Autoregressive GRU tag decoder conditioned on a provided input embedding.

    Unlike ``TagDecoder`` in ``char_gru.py``, ``forward_step`` accepts a
    pre-computed embedding tensor (rather than a token index), enabling the
    root embedding to serve as the first decoder input without requiring the
    root to be present in the tag vocabulary.

    Args:
        tag_vocab_size: Number of output tag tokens.
        embed_dim: Tag embedding dimension.
        hidden_dim: GRU hidden state dimension.
        encoder_dim: Encoder output dimension (``2 * gru_hidden_dim``).
        num_layers: Number of stacked GRU layers.
        dropout: Dropout probability.
    """

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
        self.tag_embed = nn.Embedding(tag_vocab_size, embed_dim, padding_idx=0)
        self.attention = BahdanauAttention(hidden_dim, encoder_dim)
        gru = nn.GRU(
            embed_dim + encoder_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        if weight_dropout > 0.0:
            from kokturk.models.variational_dropout import WeightDropout
            self.gru = WeightDropout(
                gru, [f"weight_hh_l{i}" for i in range(num_layers)],
                dropout=weight_dropout,
            )
        else:
            self.gru = gru
        self.output_proj = nn.Linear(hidden_dim, tag_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward_step(
        self,
        input_embed: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single decoding step with a pre-computed input embedding.

        Args:
            input_embed: ``(B, embed_dim)`` pre-computed embedding for this
                step (root embedding on the first step; tag embedding on
                subsequent steps).
            hidden: ``(num_layers, B, hidden_dim)`` decoder hidden state.
            encoder_outputs: ``(B, L, encoder_dim)`` from the char encoder.

        Returns:
            logits: ``(B, tag_vocab_size)`` unnormalised log-probabilities.
            hidden: Updated hidden state ``(num_layers, B, hidden_dim)``.
            attn_weights: ``(B, L)`` attention weights.
        """
        # Compute attention context using the top GRU layer hidden state.
        context, attn_weights = self.attention(
            hidden[-1], encoder_outputs
        )  # (B, encoder_dim), (B, L)

        # Concatenate input embedding and context, then run one GRU step.
        gru_input = torch.cat(
            [self.dropout(input_embed), context], dim=-1
        ).unsqueeze(1)  # (B, 1, embed_dim + encoder_dim)

        output, hidden = self.gru(gru_input, hidden)  # output: (B, 1, hidden_dim)
        logits = self.output_proj(output.squeeze(1))  # (B, tag_vocab_size)
        return logits, hidden, attn_weights


class DualHeadAtomizer(nn.Module):
    """Dual-head morphological atomizer.

    Root is classified in a single pass (no autoregression), while the tag
    sequence decoder is conditioned on the predicted or gold root embedding.
    This design removes root classification errors from polluting tag decoding.

    The ``tag_ids`` tensor encodes ``"ev +PLU +POSS.3SG +ABL"`` as::

        index:    0     1      2      3           4     5   6...
        content: SOS  "ev"  +PLU  +POSS.3SG   +ABL  EOS  PAD

    The tag decoder therefore:

    * Uses ``root_embed`` (not SOS) as the first step input.
    * Predicts tokens starting at ``tag_ids[:, 2]`` (first morphological tag).
    * Runs for ``tag_ids.size(1) - 2`` steps total.

    Args:
        char_vocab_size: Character vocabulary size.
        tag_vocab_size: Full tag vocabulary size (``7807`` in this project).
        root_vocab_size: Root-only vocabulary size (``~7732`` roots).
        embed_dim: Embedding dimension for chars, tags, and roots.
        hidden_dim: GRU hidden state dimension.
        num_layers: Number of GRU layers.
        dropout: Dropout probability.
        max_decode_len: Maximum tag sequence length to decode at inference.
        sos_idx: Start-of-sequence index (``1``).
        eos_idx: End-of-sequence index (``2``).
        root_loss_weight: Weight for the root auxiliary loss (default ``0.3``).
    """

    def __init__(
        self,
        char_vocab_size: int,
        tag_vocab_size: int,
        root_vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        max_decode_len: int = 15,
        sos_idx: int = 1,
        eos_idx: int = 2,
        root_loss_weight: float = 0.3,
        root_head_type: str = "mlp",
        contrastive_margin: float = 1.0,
        root_vocab_path: str | None = None,
        variational_dropout: float = 0.0,
        weight_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = CharGRUEncoder(
            char_vocab_size, embed_dim, hidden_dim, num_layers, dropout,
            variational_dropout=variational_dropout,
            weight_dropout=weight_dropout,
        )
        self.root_head = RootHead(
            encoder_dim=2 * hidden_dim,
            root_vocab_size=root_vocab_size,
            dropout=dropout,
        )
        # Optional contrastive root head — only created when explicitly
        # requested. Default "mlp" preserves backward compat: existing v2
        # checkpoints load with strict=True because no new parameters appear
        # under the default constructor.
        self.root_head_type = root_head_type
        if root_head_type == "contrastive":
            from kokturk.models.contrastive_root import (  # noqa: PLC0415
                ContrastiveRootHead,
                load_prefix_negatives_from_file,
            )
            negative_index = None
            if root_vocab_path is not None:
                try:
                    negative_index = load_prefix_negatives_from_file(root_vocab_path)
                except Exception:  # noqa: BLE001
                    negative_index = None
            self.contrastive_root_head = ContrastiveRootHead(
                encoder_dim=2 * hidden_dim,
                root_vocab_size=root_vocab_size,
                embed_dim=128,
                margin=contrastive_margin,
                negative_index=negative_index,
                dropout=dropout,
            )
        elif root_head_type != "mlp":
            raise ValueError(
                f"root_head_type must be 'mlp' or 'contrastive', got {root_head_type!r}"
            )
        self.root_embedding = nn.Embedding(root_vocab_size, embed_dim)
        self.tag_decoder = ConditionalTagDecoder(
            tag_vocab_size,
            embed_dim,
            hidden_dim,
            encoder_dim=2 * hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            weight_dropout=weight_dropout,
        )
        # Learned projection from encoder hidden to decoder initial hidden.
        # Both have shape (num_layers, B, hidden_dim) so this is applied
        # per-layer with a shared Linear.
        self.bridge = nn.Linear(hidden_dim, hidden_dim)

        self.max_decode_len = max_decode_len
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.root_loss_weight = root_loss_weight

    def forward(
        self,
        chars: torch.Tensor,
        tag_ids: torch.Tensor | None = None,
        gold_root: torch.Tensor | None = None,
        use_gold_root: bool = True,
        teacher_forcing_ratio: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional teacher forcing.

        Args:
            chars: ``(B, L_char)`` character indices.
            tag_ids: ``(B, max_tag_len)`` full tag sequence including SOS and
                root token.  Required for teacher forcing; if ``None`` decodes
                greedily for ``max_decode_len - 2`` steps.
            gold_root: ``(B,)`` root indices in ``root_vocab``.  Used as the
                decoder conditioning signal when ``use_gold_root`` is ``True``.
            use_gold_root: If ``True`` and ``gold_root`` is provided, condition
                the tag decoder on the gold root embedding (teacher forcing for
                the root).  If ``False``, use the predicted root.
            teacher_forcing_ratio: Probability of feeding the gold tag as the
                next decoder input (0.0 = fully greedy, 1.0 = full teacher
                forcing).

        Returns:
            root_logits: ``(B, root_vocab_size)`` root classification logits.
            tag_outputs: ``(B, decode_len, tag_vocab_size)`` tag prediction
                logits where ``decode_len = tag_ids.size(1) - 2``.
        """
        encoder_outputs, hidden = self.encoder(chars)
        # encoder_outputs: (B, L, 2*hidden_dim)
        # hidden:          (num_layers, B, hidden_dim)

        # Build padding mask: True for real characters.
        mask = (chars != 0)  # (B, L)

        # --- Root classification ---
        if self.root_head_type == "contrastive":
            # Contrastive head is used for the auxiliary loss; we still emit
            # logits-shaped output (negative distances) so callers see the
            # same return signature.
            cr_out = self.contrastive_root_head(
                encoder_outputs,
                mask=mask,
                gold_root=gold_root,
            )
            root_logits = cr_out["logits"]  # (B, root_vocab_size)
            # Cache loss for retrieval by the training loop. Forward returns
            # (root_logits, tag_outputs) — unchanged signature.
            self._cached_contrastive_loss = cr_out.get("loss")
        else:
            root_logits = self.root_head(encoder_outputs, mask)  # (B, root_vocab_size)
            self._cached_contrastive_loss = None

        # Choose root embedding for conditioning the tag decoder.
        if use_gold_root and gold_root is not None:
            root_idx = gold_root
        else:
            root_idx = root_logits.argmax(dim=-1)  # (B,)

        root_embed = self.root_embedding(root_idx)  # (B, embed_dim)

        # Apply learned bridge to initialise decoder hidden state.
        hidden = torch.stack(
            [self.bridge(hidden[i]) for i in range(hidden.size(0))]
        )  # (num_layers, B, hidden_dim)

        # --- Tag decoding ---
        decode_len = (
            tag_ids.size(1) - 2 if tag_ids is not None else self.max_decode_len - 2
        )

        all_logits: list[torch.Tensor] = []
        current_embed = root_embed  # (B, embed_dim) — first step input

        for t in range(decode_len):
            logits, hidden, _ = self.tag_decoder.forward_step(
                current_embed, hidden, encoder_outputs
            )
            all_logits.append(logits.unsqueeze(1))  # (B, 1, V)

            # Determine next-step input embedding.
            if tag_ids is not None and random.random() < teacher_forcing_ratio:
                # Teacher forcing: feed gold tag at position t+2.
                # When t = decode_len-1, t+2 = tag_ids.size(1)-1 (last col: EOS/PAD).
                next_tag_idx = tag_ids[:, t + 2]  # (B,)
            else:
                next_tag_idx = logits.argmax(dim=-1)  # (B,)

            current_embed = self.tag_decoder.tag_embed(next_tag_idx)  # (B, embed_dim)

        tag_outputs = torch.cat(all_logits, dim=1)  # (B, decode_len, V)
        return root_logits, tag_outputs

    @torch.no_grad()
    def greedy_decode(
        self,
        chars: torch.Tensor,
        root_vocab_inv: list[str] | None = None,
        tag_vocab_inv: list[str] | None = None,
        morphotactic_mask: "MorphotacticMask | None" = None,
    ) -> list[str]:
        """Greedy decode; returns canonical label strings like ``"ev +PLU +ABL"``.

        Args:
            chars: ``(B, L_char)`` character indices.
            root_vocab_inv: List of root strings indexed by root vocabulary
                index.  If ``None``, roots are rendered as ``"root_<idx>"``.
            tag_vocab_inv: List of tag strings indexed by tag vocabulary index.
                If ``None``, tags are rendered as ``"tag_<idx>"``.

        Returns:
            List of B label strings, one per sample in the batch.
        """
        self.eval()
        encoder_outputs, hidden = self.encoder(chars)
        mask = (chars != 0)

        root_logits = self.root_head(encoder_outputs, mask)
        root_idx = root_logits.argmax(dim=-1)  # (B,)
        root_embed = self.root_embedding(root_idx)  # (B, embed_dim)

        hidden = torch.stack(
            [self.bridge(hidden[i]) for i in range(hidden.size(0))]
        )

        results: list[str] = []
        batch_size = chars.size(0)

        for b in range(batch_size):
            if root_vocab_inv is not None:
                root_str = root_vocab_inv[root_idx[b].item()]
            else:
                root_str = f"root_{root_idx[b].item()}"

            # dual_head decodes one sample at a time, so reset the FSA
            # at the top of each iteration with batch_size=1.  Resetting
            # once outside the loop would corrupt state across samples.
            if morphotactic_mask is not None:
                morphotactic_mask.reset(batch_size=1)

            tags: list[str] = []
            curr_embed = root_embed[b : b + 1]  # (1, embed_dim)
            h = hidden[:, b : b + 1, :].contiguous()  # (num_layers, 1, hidden_dim)
            enc_out = encoder_outputs[b : b + 1]  # (1, L, 2H)

            for _ in range(self.max_decode_len - 2):
                logit, h, _ = self.tag_decoder.forward_step(curr_embed, h, enc_out)
                if morphotactic_mask is not None:
                    step_logit = logit.squeeze(1)  # (1, V)
                    allowed = morphotactic_mask.get_mask()
                    step_logit = step_logit.masked_fill(~allowed, float("-inf"))
                    pred_tensor = step_logit.argmax(dim=-1)  # (1,)
                    morphotactic_mask.update(pred_tensor)
                    pred = int(pred_tensor.item())
                else:
                    pred = int(logit.argmax(dim=-1).item())
                if pred == self.eos_idx:
                    break
                if tag_vocab_inv is not None and pred < len(tag_vocab_inv):
                    token = tag_vocab_inv[pred]
                    if token not in ("<PAD>", "<SOS>", "<EOS>", "<UNK>"):
                        tags.append(token)
                else:
                    tags.append(f"tag_{pred}")
                curr_embed = self.tag_decoder.tag_embed(
                    torch.tensor([[pred]], device=chars.device)
                ).squeeze(1)  # (1, embed_dim)

            label = root_str + (" " + " ".join(tags) if tags else "")
            results.append(label)

        return results

    def count_parameters(self) -> int:
        """Count trainable parameters.

        Returns:
            Total number of parameters with ``requires_grad=True``.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
