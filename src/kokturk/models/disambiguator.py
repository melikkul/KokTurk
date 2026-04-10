"""BERTurk-based morphological disambiguator.

Instead of generating a parse from characters (hard, ~85% ceiling),
this model SELECTS the best parse from Zeyrek's candidate list using
sentence context from frozen BERTurk embeddings.

Architecture:
    Sentence -> BERTurk (frozen) -> per-word embedding (768-d)
                                        |
                                  bert_proj -> (256-d)
                                        |
    Candidates -> tag_vocab embed -> BiGRU -> cand_proj -> (256-d)
                                        |
                         concat(bert_proj, cand_proj) -> MLP -> score
                                        |
                             softmax -> select best candidate

Training: cross-entropy over candidate indices.
Inference: argmax over candidate scores.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CandidateEncoder(nn.Module):
    """Encode a canonical parse string (as token IDs) into a fixed-dim vector.

    Each candidate like "ev +PLU +POSS.3SG +ABL" is pre-tokenized into
    tag_vocab indices and passed through embedding + bidirectional GRU.
    The final hidden states (forward + backward) are concatenated.
    """

    def __init__(
        self,
        tag_vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            tag_vocab_size, embed_dim, padding_idx=0,
        )
        self.gru = nn.GRU(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.output_dim = hidden_dim * 2  # bidirectional

    def forward(self, tag_ids: torch.Tensor) -> torch.Tensor:
        """Encode candidate parse sequences.

        Args:
            tag_ids: (B, K, L) padded tag token indices.

        Returns:
            (B, K, output_dim) one vector per candidate.
        """
        B, K, L = tag_ids.shape
        flat = tag_ids.view(B * K, L)

        embedded = self.embedding(flat)  # (B*K, L, embed_dim)
        _, hidden = self.gru(embedded)   # hidden: (2*layers, B*K, hidden_dim)

        # Concat forward and backward final hidden states
        fwd = hidden[-2]  # (B*K, hidden_dim)
        bwd = hidden[-1]  # (B*K, hidden_dim)
        encoded = torch.cat([fwd, bwd], dim=-1)  # (B*K, 2*hidden_dim)

        return encoded.view(B, K, self.output_dim)


class BERTurkDisambiguator(nn.Module):
    """Select the best morphological parse from Zeyrek candidates
    using BERTurk sentence context.

    BERTurk is frozen — only the candidate encoder, projections, and
    score head are trained (~960K parameters).

    Supports injectable bert_model/tokenizer for testing without
    downloading the full model (same pattern as BERTurkContext).
    """

    def __init__(
        self,
        tag_vocab_size: int,
        bert_path: str = "models/berturk",
        cand_embed_dim: int = 64,
        cand_hidden_dim: int = 128,
        projection_dim: int = 256,
        dropout: float = 0.3,
        bert_model: object | None = None,
        tokenizer: object | None = None,
        skip_bert_loading: bool = False,
    ) -> None:
        super().__init__()
        self.bert_dim = 768

        # Frozen BERTurk for sentence context
        # skip_bert_loading=True when using pre-cached embeddings (saves ~440 MB)
        if skip_bert_loading:
            self.bert = None  # type: ignore[assignment]
            self.tokenizer = None  # type: ignore[assignment]
        elif bert_model is not None:
            self.bert = bert_model  # type: ignore[assignment]
            if tokenizer is not None:
                self.tokenizer = tokenizer
            else:
                from transformers import AutoTokenizer  # type: ignore[import]
                self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
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

        # Candidate parse encoder
        self.candidate_encoder = CandidateEncoder(
            tag_vocab_size, cand_embed_dim, cand_hidden_dim,
        )

        # Project both to shared space
        self.bert_proj = nn.Sequential(
            nn.Linear(self.bert_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.Dropout(dropout),
        )
        self.cand_proj = nn.Sequential(
            nn.Linear(self.candidate_encoder.output_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.Dropout(dropout),
        )

        # Score head: context + candidate -> scalar
        self.score_head = nn.Sequential(
            nn.Linear(projection_dim * 2, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, 1),
        )

    def _extract_target_embeddings(
        self,
        hidden: torch.Tensor,
        encoding: object,
        target_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Extract BERTurk embedding for each target word position.

        Uses first-subword-token alignment (same approach as
        BERTurkContext in context_encoder.py).

        Args:
            hidden: (B, S, 768) BERTurk hidden states.
            encoding: Tokenizer BatchEncoding with word_ids().
            target_positions: (B,) word positions (0-based).

        Returns:
            (B, 768) target word embeddings.
        """
        B = hidden.size(0)
        positions = target_positions.tolist()
        target_embeds: list[torch.Tensor] = []

        for b in range(B):
            try:
                word_ids_b = encoding.word_ids(b)  # type: ignore[union-attr]
                subword_idx = next(
                    (
                        i for i, wid in enumerate(word_ids_b)
                        if wid == int(positions[b])
                    ),
                    1,  # fallback: first non-special token
                )
            except Exception:
                # Fallback if fast-tokenizer word_ids() not available
                subword_idx = min(int(positions[b]) + 1, hidden.size(1) - 1)
            target_embeds.append(hidden[b, subword_idx])

        return torch.stack(target_embeds)  # (B, 768)

    def forward(
        self,
        sentence_texts: list[str],
        target_positions: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_mask: torch.Tensor,
        gold_indices: torch.Tensor | None = None,
        cached_bert_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Score each candidate parse against BERTurk context.

        Args:
            sentence_texts: List of B raw sentence strings.
            target_positions: (B,) word positions in sentences.
            candidate_ids: (B, K, L) padded candidate tag indices.
            candidate_mask: (B, K) True for real candidates.
            gold_indices: (B,) correct candidate index (for training).
            cached_bert_embeds: (B, 768) pre-computed BERTurk embeddings
                (skips BERT forward pass when provided).

        Returns:
            (logits, loss) where logits is (B, K) and loss is scalar or None.
        """
        device = next(self.score_head.parameters()).device
        B, K, L = candidate_ids.shape

        # 1. Get BERTurk context for target word
        if cached_bert_embeds is not None:
            context = cached_bert_embeds.to(device)
        else:
            if self.bert is None or self.tokenizer is None:
                raise RuntimeError(
                    "BERTurk not loaded — pass cached_bert_embeds or "
                    "instantiate model with skip_bert_loading=False"
                )
            encoding = self.tokenizer(
                sentence_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            with torch.no_grad():
                outputs = self.bert(
                    input_ids=input_ids, attention_mask=attention_mask,
                )
            hidden = outputs.last_hidden_state  # (B, S, 768)
            context = self._extract_target_embeddings(
                hidden, encoding, target_positions,
            )  # (B, 768)

        # 2. Encode all candidates
        cand_encoded = self.candidate_encoder(
            candidate_ids.to(device),
        )  # (B, K, cand_dim)

        # 3. Project both to shared space
        context_proj = self.bert_proj(context)    # (B, proj_dim)
        cand_proj = self.cand_proj(cand_encoded)  # (B, K, proj_dim)

        # 4. Score each candidate against context
        context_expanded = context_proj.unsqueeze(1).expand(-1, K, -1)
        combined = torch.cat(
            [context_expanded, cand_proj], dim=-1,
        )  # (B, K, 2*proj_dim)
        logits = self.score_head(combined).squeeze(-1)  # (B, K)

        # Mask invalid candidates
        logits = logits.masked_fill(~candidate_mask.to(device), float("-inf"))

        # 5. Loss
        loss = None
        if gold_indices is not None:
            loss = F.cross_entropy(logits, gold_indices.to(device))

        return logits, loss
