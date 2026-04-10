"""Composite acquisition function for active learning.

Scores sentences by morphological complexity and model uncertainty
to select the most informative examples for human annotation.

A(x) = lambda_bald * BALD(x) + lambda_mad * MAD(x) + lambda_conf * CONF(x)

When TRMorph is unavailable (no foma), CONF=0 and weights are redistributed:
A(x) = 0.55 * BALD(x) + 0.45 * MAD(x)

When no neural model exists (seed stage), BALD=0 and scoring is MAD-only.
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


class MorphAcquisition:
    """Composite acquisition function for active learning sentence selection.

    Args:
        lambda_bald: Weight for BALD (model uncertainty). 0.0 if no model.
        lambda_mad: Weight for MAD (morphological ambiguity density).
        lambda_conf: Weight for CONF (analyzer conflict density). 0.0 if no TRMorph.
        max_parse_count: Maximum parse count for MAD normalization.
        model: Trained MorphAtomizer model for BALD. None = MAD-only.
        char_vocab: Character vocabulary for model input encoding.
        tag_vocab: Tag vocabulary for model output decoding.
        mc_samples: Number of MC dropout forward passes for BALD.
    """

    def __init__(
        self,
        lambda_bald: float = 0.55,
        lambda_mad: float = 0.45,
        lambda_conf: float = 0.0,
        max_parse_count: int = 10,
        model: Any = None,
        char_vocab: Any = None,
        tag_vocab: Any = None,
        mc_samples: int = 20,
    ) -> None:
        self.lambda_bald = lambda_bald
        self.lambda_mad = lambda_mad
        self.lambda_conf = lambda_conf
        self.max_parse_count = max_parse_count
        self.model = model
        self.char_vocab = char_vocab
        self.tag_vocab = tag_vocab
        self.mc_samples = mc_samples

    def compute_mad(self, parse_count: int) -> float:
        """Morphological Ambiguity Density for a single token.

        MAD = log(parse_count) / log(max_parse_count), normalized to [0,1].
        Unambiguous tokens (1 parse) → 0.0.
        Highly ambiguous tokens (many parses) → ~1.0.

        Args:
            parse_count: Number of candidate parses for the token.

        Returns:
            MAD score in [0.0, 1.0].
        """
        if parse_count <= 1:
            return 0.0
        if self.max_parse_count <= 1:
            return 0.0
        return math.log(parse_count) / math.log(self.max_parse_count)

    def compute_bald(
        self, tokens: list[str],
    ) -> float:
        """BALD (Bayesian Active Learning by Disagreement) via MC dropout.

        Runs T forward passes with dropout enabled, computes mutual
        information between model predictions and parameters:
        BALD = H[y|x] - E_theta[H[y|x,theta]]
             = total_entropy - mean_individual_entropy

        Returns 0.0 when no neural model is available.

        Args:
            tokens: List of surface forms in the sentence.

        Returns:
            BALD score (non-negative). 0.0 if no model.
        """
        if self.model is None or self.char_vocab is None:
            return 0.0

        try:
            import torch
            import torch.nn.functional as functional  # noqa: N812
        except ImportError:
            return 0.0

        device = next(self.model.parameters()).device
        bald_scores: list[float] = []

        for surface in tokens:
            # Encode characters
            char_ids = [self.char_vocab.encode(c) for c in surface]
            char_ids.append(2)  # EOS
            char_ids += [0] * (64 - len(char_ids))
            chars = torch.tensor(
                [char_ids[:64]], dtype=torch.long, device=device,
            )

            # MC dropout: T forward passes with dropout ON
            self.model.train()  # enable dropout
            predictions: list[torch.Tensor] = []
            for _ in range(self.mc_samples):
                with torch.no_grad():
                    logits = self.model(chars, teacher_forcing_ratio=0.0)
                    # logits: (1, L, V) → softmax probabilities
                    probs = functional.softmax(logits[0], dim=-1)  # (L, V)
                    predictions.append(probs)

            # Stack: (T, L, V)
            stacked = torch.stack(predictions)
            mean_pred = stacked.mean(dim=0)  # (L, V)

            # Total entropy: H[E[p]]
            total_ent = -(
                mean_pred * (mean_pred + 1e-10).log()
            ).sum(dim=-1).mean()

            # Mean individual entropy: E[H[p]]
            indiv_ents: list[torch.Tensor] = []
            for pred in predictions:
                ent = -(pred * (pred + 1e-10).log()).sum(dim=-1).mean()
                indiv_ents.append(ent)
            avg_ent = torch.stack(indiv_ents).mean()

            bald = (total_ent - avg_ent).clamp(min=0.0).item()
            bald_scores.append(bald)

        self.model.eval()
        return sum(bald_scores) / max(len(bald_scores), 1)

    def compute_conf(self, token_candidates: list[dict[str, object]]) -> float:
        """Analyzer conflict density for a single token.

        CONF = 1.0 if Zeyrek and TRMorph disagree, 0.0 otherwise.
        Returns 0.0 when TRMorph is unavailable.

        Args:
            token_candidates: List of candidate parse dicts.

        Returns:
            CONF score: 0.0 or 1.0.
        """
        if self.lambda_conf == 0.0:
            return 0.0

        zeyrek_ids = {
            (c.get("root"), tuple(c.get("tags", [])))
            for c in token_candidates
            if c.get("source") == "zeyrek"
        }
        trmorph_ids = {
            (c.get("root"), tuple(c.get("tags", [])))
            for c in token_candidates
            if c.get("source") == "trmorph"
        }

        if not zeyrek_ids or not trmorph_ids:
            return 0.0

        # Disagree if no overlap
        return 0.0 if zeyrek_ids & trmorph_ids else 1.0

    def score_sentence(
        self,
        sentence_candidates: list[dict[str, object]],
    ) -> float:
        """Composite acquisition score for a sentence.

        Args:
            sentence_candidates: List of candidate dicts for each token
                in the sentence (from candidates.jsonl).

        Returns:
            Acquisition score (higher = more informative for annotation).
        """
        if not sentence_candidates:
            return 0.0

        # MAD: average across tokens
        mad_scores = [
            self.compute_mad(int(c.get("parse_count", 0)))
            for c in sentence_candidates
        ]
        avg_mad = sum(mad_scores) / len(mad_scores)

        # BALD: MC dropout (0.0 if no model)
        surfaces = [str(c.get("surface", "")) for c in sentence_candidates]
        bald_score = self.compute_bald(surfaces)

        # CONF: average across tokens
        conf_scores = [
            self.compute_conf(c.get("analyses", []))  # type: ignore[arg-type]
            for c in sentence_candidates
        ]
        avg_conf = sum(conf_scores) / len(conf_scores) if conf_scores else 0.0

        return (
            self.lambda_bald * bald_score
            + self.lambda_mad * avg_mad
            + self.lambda_conf * avg_conf
        )

    def rank_sentences(
        self,
        corpus: list[dict[str, object]],
        candidates_by_sent: dict[str, list[dict[str, object]]],
        exclude_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Rank all sentences by acquisition score.

        Args:
            corpus: List of sentence dicts.
            candidates_by_sent: Candidates indexed by sentence_id.
            exclude_ids: Sentence IDs to skip (already annotated).

        Returns:
            List of (sentence_id, score) sorted descending by score.
        """
        if exclude_ids is None:
            exclude_ids = set()

        scored: list[tuple[str, float]] = []
        for sent in corpus:
            sid = str(sent["sentence_id"])
            if sid in exclude_ids:
                continue
            cands = candidates_by_sent.get(sid, [])
            score = self.score_sentence(cands)
            scored.append((sid, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


def select_batch(
    corpus: list[dict[str, object]],
    candidates_by_sent: dict[str, list[dict[str, object]]],
    exclude_ids: set[str] | None = None,
    batch_size: int = 50,
    lambda_bald: float = 0.55,
    lambda_mad: float = 0.45,
    lambda_conf: float = 0.0,
) -> list[tuple[str, float]]:
    """Select a batch of sentences for annotation.

    Args:
        corpus: All sentences.
        candidates_by_sent: Prelabeled candidates indexed by sentence_id.
        exclude_ids: Already-annotated sentence IDs.
        batch_size: Number of sentences to select.
        lambda_bald: BALD weight.
        lambda_mad: MAD weight.
        lambda_conf: CONF weight.

    Returns:
        List of (sentence_id, score) for the selected batch.
    """
    acq = MorphAcquisition(
        lambda_bald=lambda_bald,
        lambda_mad=lambda_mad,
        lambda_conf=lambda_conf,
    )
    ranked = acq.rank_sentences(corpus, candidates_by_sent, exclude_ids)
    return ranked[:batch_size]
