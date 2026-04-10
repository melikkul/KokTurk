"""Context-augmented dual-head morphological atomizer.

Wraps :class:`DualHeadAtomizer` with a pluggable sentence-level context encoder
and injects the context vector at two fusion points:

1. **Root head**: context is added to the attention-pooled encoder output
   *before* the root classification layers.
2. **Tag decoder init**: context is added to the bridge-projected encoder hidden
   state *before* it initialises the GRU decoder.

Both fusion points use **additive projection** followed by **LayerNorm** (critical
to prevent BERTurk magnitude dominance) and **context dropout** (p=0.3) to keep
char-level predictions strong when context is absent or noisy.

Usage::

    from kokturk.models.context_encoder import Word2VecContext
    from kokturk.models.contextual_dual_head import ContextualDualHeadAtomizer

    ctx_enc = Word2VecContext(vocab_size=50000, embed_dim=128, gru_hidden_dim=64)
    model = ContextualDualHeadAtomizer(
        context_encoder=ctx_enc,
        char_vocab_size=100, tag_vocab_size=7807, root_vocab_size=7732,
    )
    root_logits, tag_outputs = model(chars, neighbor_ids, tag_ids=tag_ids)
"""
from __future__ import annotations

import random

import torch
import torch.nn as nn

from kokturk.models.char_gru import BahdanauAttention, CharGRUEncoder
from kokturk.models.context_encoder import ContextEncoderBase
from kokturk.models.dual_head import (
    AttentionPooling,
    ConditionalTagDecoder,
    RootHead,
)


class ContextualDualHeadAtomizer(nn.Module):
    """DualHeadAtomizer augmented with a sentence-level context encoder.

    The char encoder, root head, root embedding, tag decoder, and bridge are
    re-instantiated here (composition, not inheritance) so their weights are
    independent from any existing :class:`DualHeadAtomizer` checkpoint.

    Fusion strategy — **additive projection with post-addition LayerNorm**:

    .. code-block:: text

        context_vec      = context_encoder(context_inputs)   # (B, C)
        ctx_root         = context_dropout(ctx_proj_root(context_vec))  # (B, enc_dim)
        fused_pool       = root_ln(attn_pool + ctx_root)     # (B, enc_dim)
        → passes fused_pool through root-head FC layers → root_logits

        ctx_dec          = context_dropout(ctx_proj_dec(context_vec))  # (B, H)
        fused_hidden[i]  = dec_ln(bridge(hidden[i]) + ctx_dec)        # per layer
        → feeds fused_hidden as initial decoder hidden state

    Context dropout at p=0.3 forces the model to maintain good char-only
    predictions even when context is absent or uninformative.

    Args:
        context_encoder: Any :class:`ContextEncoderBase` subclass.
        char_vocab_size: Character vocabulary size.
        tag_vocab_size: Full tag vocabulary size.
        root_vocab_size: Root-only vocabulary size (~7 732 roots).
        embed_dim: Embedding dimension for chars, tags, and roots.
        hidden_dim: GRU hidden state dimension.
        num_layers: Number of GRU layers.
        dropout: Dropout probability for the sequence model.
        context_dropout: Dropout probability applied to context projections.
        max_decode_len: Maximum tag sequence length for greedy inference.
        sos_idx: Start-of-sequence index (1).
        eos_idx: End-of-sequence index (2).
        root_loss_weight: Weight for the root auxiliary loss (default 0.3).
    """

    def __init__(
        self,
        context_encoder: ContextEncoderBase,
        char_vocab_size: int,
        tag_vocab_size: int,
        root_vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        context_dropout: float = 0.3,
        max_decode_len: int = 15,
        sos_idx: int = 1,
        eos_idx: int = 2,
        root_loss_weight: float = 0.3,
        variational_dropout: float = 0.0,
        weight_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        enc_dim = 2 * hidden_dim  # bidirectional encoder output dimension

        # ---- Shared with DualHeadAtomizer (same architecture, fresh weights) ----
        self.encoder = CharGRUEncoder(
            char_vocab_size, embed_dim, hidden_dim, num_layers, dropout,
            variational_dropout=variational_dropout,
            weight_dropout=weight_dropout,
        )
        self.root_head = RootHead(
            encoder_dim=enc_dim,
            root_vocab_size=root_vocab_size,
            dropout=dropout,
        )
        self.root_embedding = nn.Embedding(root_vocab_size, embed_dim)
        self.tag_decoder = ConditionalTagDecoder(
            tag_vocab_size,
            embed_dim,
            hidden_dim,
            encoder_dim=enc_dim,
            num_layers=num_layers,
            dropout=dropout,
            weight_dropout=weight_dropout,
        )
        # Bridge: per-layer linear from encoder hidden to decoder init hidden
        self.bridge = nn.Linear(hidden_dim, hidden_dim)

        # ---- Context encoder & fusion layers ----
        self.context_encoder = context_encoder
        ctx_dim = context_encoder.output_dim

        # Project context to encoder_dim (for root-head fusion)
        self.ctx_proj_root = nn.Linear(ctx_dim, enc_dim, bias=False)
        self.root_ln = nn.LayerNorm(enc_dim)

        # Project context to hidden_dim (for decoder-init fusion)
        self.ctx_proj_dec = nn.Linear(ctx_dim, hidden_dim, bias=False)
        self.dec_ln = nn.LayerNorm(hidden_dim)

        # Context dropout — shared across both fusion points
        self.ctx_dropout = nn.Dropout(context_dropout)

        # ---- Inference params ----
        self.max_decode_len = max_decode_len
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.root_loss_weight = root_loss_weight

    # ---------------------------------------------------------------------- #
    # Forward pass
    # ---------------------------------------------------------------------- #

    def _encode_context(self, context_inputs: object) -> torch.Tensor:
        """Dispatch context_inputs to the context encoder.

        Args:
            context_inputs: A single tensor (for POSBigramContext /
                Word2VecContext) or a tuple/list of args (for
                SentenceBiGRUContext / BERTurkContext).

        Returns:
            ``(B, context_encoder.output_dim)`` context vector.
        """
        if isinstance(context_inputs, (tuple, list)):
            return self.context_encoder(*context_inputs)
        return self.context_encoder(context_inputs)

    def forward(
        self,
        chars: torch.Tensor,
        context_inputs: object,
        tag_ids: torch.Tensor | None = None,
        gold_root: torch.Tensor | None = None,
        use_gold_root: bool = True,
        teacher_forcing_ratio: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with context fusion.

        Args:
            chars: ``(B, L_char)`` character indices.
            context_inputs: Context features forwarded to the context encoder.
                - Single tensor → ``context_encoder(context_inputs)``
                - Tuple/list   → ``context_encoder(*context_inputs)``
            tag_ids: ``(B, max_tag_len)`` tag sequence including SOS and root.
                Required for teacher forcing; if ``None``, decodes greedily for
                ``max_decode_len - 2`` steps.
            gold_root: ``(B,)`` root indices in root_vocab.  Used as decoder
                conditioning signal when ``use_gold_root`` is ``True``.
            use_gold_root: If ``True`` and ``gold_root`` is provided, condition
                the tag decoder on the gold root embedding.
            teacher_forcing_ratio: Probability of feeding the gold tag as next
                decoder input (0.0 = fully greedy, 1.0 = full teacher forcing).

        Returns:
            root_logits: ``(B, root_vocab_size)`` root classification logits.
            tag_outputs: ``(B, decode_len, tag_vocab_size)`` tag prediction
                logits where ``decode_len = tag_ids.size(1) - 2``.
        """
        # ---- Encode characters ----
        encoder_outputs, hidden = self.encoder(chars)
        # encoder_outputs: (B, L, enc_dim)
        # hidden:          (num_layers, B, hidden_dim)

        mask = (chars != 0)  # (B, L) — True for real characters

        # ---- Encode context ----
        context_vec = self._encode_context(context_inputs)  # (B, ctx_dim)

        # ---- Root classification with context fusion ----
        # Attention-pool encoder outputs (reuse RootHead's pooling layer)
        pooled = self.root_head.pooling(encoder_outputs, mask)  # (B, enc_dim)

        # Add projected context and normalise
        ctx_root = self.ctx_dropout(self.ctx_proj_root(context_vec))  # (B, enc_dim)
        fused_pool = self.root_ln(pooled + ctx_root)                  # (B, enc_dim)

        # Apply RootHead classification layers on the fused representation
        x = self.root_head.dropout1(fused_pool)
        x = self.root_head.relu(self.root_head.fc1(x))
        x = self.root_head.dropout2(x)
        root_logits = self.root_head.fc2(x)  # (B, root_vocab_size)

        # ---- Root embedding for decoder conditioning ----
        if use_gold_root and gold_root is not None:
            root_idx = gold_root
        else:
            root_idx = root_logits.argmax(dim=-1)  # (B,)

        root_embed = self.root_embedding(root_idx)  # (B, embed_dim)

        # ---- Bridge encoder hidden → decoder init, with context fusion ----
        ctx_dec = self.ctx_dropout(self.ctx_proj_dec(context_vec))  # (B, hidden_dim)
        # Bridge each layer and add context; broadcast ctx_dec over layers
        hidden = torch.stack(
            [self.dec_ln(self.bridge(hidden[i]) + ctx_dec)
             for i in range(hidden.size(0))]
        )  # (num_layers, B, hidden_dim)

        # ---- Tag decoding ----
        decode_len = (
            tag_ids.size(1) - 2 if tag_ids is not None else self.max_decode_len - 2
        )

        all_logits: list[torch.Tensor] = []
        current_embed = root_embed  # first step input is the root embedding

        for t in range(decode_len):
            logits, hidden, _ = self.tag_decoder.forward_step(
                current_embed, hidden, encoder_outputs
            )
            all_logits.append(logits.unsqueeze(1))  # (B, 1, V)

            if tag_ids is not None and random.random() < teacher_forcing_ratio:
                next_tag_idx = tag_ids[:, t + 2]
            else:
                next_tag_idx = logits.argmax(dim=-1)

            current_embed = self.tag_decoder.tag_embed(next_tag_idx)  # (B, embed_dim)

        tag_outputs = torch.cat(all_logits, dim=1)  # (B, decode_len, V)
        return root_logits, tag_outputs

    # ---------------------------------------------------------------------- #
    # Greedy inference
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def greedy_decode(
        self,
        chars: torch.Tensor,
        context_inputs: object,
        root_vocab_inv: list[str] | None = None,
        tag_vocab_inv: list[str] | None = None,
    ) -> list[str]:
        """Greedy decode; returns canonical label strings like ``"ev +PLU +ABL"``.

        Args:
            chars: ``(B, L_char)`` character indices.
            context_inputs: Context features (same format as ``forward``).
            root_vocab_inv: Root strings indexed by root vocabulary index.
            tag_vocab_inv: Tag strings indexed by tag vocabulary index.

        Returns:
            List of B label strings.
        """
        self.eval()
        encoder_outputs, hidden = self.encoder(chars)
        mask = (chars != 0)

        context_vec = self._encode_context(context_inputs)

        # Root classification with context
        pooled = self.root_head.pooling(encoder_outputs, mask)
        ctx_root = self.ctx_proj_root(context_vec)
        fused_pool = self.root_ln(pooled + ctx_root)
        x = self.root_head.relu(self.root_head.fc1(fused_pool))
        root_logits = self.root_head.fc2(x)
        root_idx = root_logits.argmax(dim=-1)
        root_embed = self.root_embedding(root_idx)

        # Bridge with context
        ctx_dec = self.ctx_proj_dec(context_vec)
        hidden = torch.stack(
            [self.dec_ln(self.bridge(hidden[i]) + ctx_dec)
             for i in range(hidden.size(0))]
        )

        results: list[str] = []
        batch_size = chars.size(0)

        for b in range(batch_size):
            root_str = (
                root_vocab_inv[root_idx[b].item()]
                if root_vocab_inv is not None
                else f"root_{root_idx[b].item()}"
            )

            tags: list[str] = []
            curr_embed = root_embed[b : b + 1]
            h = hidden[:, b : b + 1, :].contiguous()
            enc_out = encoder_outputs[b : b + 1]

            for _ in range(self.max_decode_len - 2):
                logit, h, _ = self.tag_decoder.forward_step(curr_embed, h, enc_out)
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
                ).squeeze(1)

            results.append(root_str + (" " + " ".join(tags) if tags else ""))

        return results

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
