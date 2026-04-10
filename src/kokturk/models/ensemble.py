"""Majority-vote ensemble over independently trained atomizer models.

Loads N model checkpoints trained with different random seeds and
combines their predictions via majority vote at the full-sequence level.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch

from kokturk.models.char_gru import MorphAtomizer
from train.datasets import EOS_IDX, Vocab


class AtomizerEnsemble:
    """Majority-vote ensemble over N MorphAtomizer models."""

    def __init__(
        self,
        model_paths: list[str | Path],
        vocab_dir: str | Path,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        vocab_dir = Path(vocab_dir)
        self.char_vocab = Vocab.load(vocab_dir / "char_vocab.json")
        self.tag_vocab = Vocab.load(vocab_dir / "tag_vocab.json")

        self.models: list[MorphAtomizer] = []
        self.model_ems: list[float] = []

        for path in model_paths:
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
            model = MorphAtomizer(
                char_vocab_size=ckpt["char_vocab_size"],
                tag_vocab_size=ckpt["tag_vocab_size"],
                embed_dim=ckpt.get("embed_dim", 64),
                hidden_dim=ckpt.get("hidden_dim", 128),
                num_layers=ckpt.get("num_layers", 2),
            )
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(self.device)
            model.eval()
            self.models.append(model)
            self.model_ems.append(ckpt.get("best_em", 0.0))

    def _encode_word(self, word: str) -> torch.Tensor:
        """Encode a single word to character tensor."""
        char_ids = [self.char_vocab.encode(c) for c in word]
        char_ids.append(EOS_IDX)
        max_len = 64
        if len(char_ids) > max_len:
            char_ids = char_ids[:max_len]
        char_ids += [0] * (max_len - len(char_ids))
        return torch.tensor([char_ids], dtype=torch.long, device=self.device)

    def _decode_prediction(self, pred: torch.Tensor) -> str:
        """Decode a prediction tensor to a tag string."""
        tokens: list[str] = []
        for idx in pred[0].tolist():
            if idx == EOS_IDX:
                break
            if idx > 3:  # skip PAD/SOS/EOS/UNK
                tokens.append(self.tag_vocab.decode(idx))
        return " ".join(tokens)

    def analyze(self, word: str) -> str:
        """Analyze a word using majority vote across all models.

        Args:
            word: Turkish surface form.

        Returns:
            Predicted analysis string (e.g., "ev +PLU +POSS.3SG +ABL").
        """
        chars = self._encode_word(word)
        predictions: list[str] = []
        for model in self.models:
            pred = model.greedy_decode(chars)
            predictions.append(self._decode_prediction(pred))

        # Majority vote on the full output string
        counts = Counter(predictions)
        best_pred, _ = counts.most_common(1)[0]

        # Tie-breaking: prefer prediction from model with highest val EM
        if len(counts) > 1:
            max_count = counts[best_pred]
            tied = [p for p, c in counts.items() if c == max_count]
            if len(tied) > 1:
                # Pick from the model with highest EM among tied predictions
                for _em, pred in sorted(
                    zip(self.model_ems, predictions, strict=False),
                    reverse=True,
                ):
                    if pred in tied:
                        best_pred = pred
                        break

        return best_pred

    def analyze_batch(self, words: list[str]) -> list[str]:
        """Analyze a batch of words."""
        return [self.analyze(w) for w in words]
