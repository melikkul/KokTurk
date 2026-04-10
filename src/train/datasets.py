"""PyTorch datasets for morphological atomizer training.

Combines gold-labeled tokens (from seed annotation) with high-confidence
weakly-labeled tokens for mixed-quality training.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]

TIER_MAP: dict[str, int] = {"gold": 0, "silver-auto": 1, "silver-agreed": 2}


class Vocab:
    """Simple token-to-index vocabulary."""

    def __init__(self, tokens: list[str] | None = None) -> None:
        self.token2idx: dict[str, int] = {}
        self.idx2token: list[str] = []
        for tok in SPECIAL_TOKENS:
            self._add(tok)
        if tokens:
            for tok in tokens:
                self._add(tok)

    def _add(self, token: str) -> int:
        if token not in self.token2idx:
            idx = len(self.idx2token)
            self.token2idx[token] = idx
            self.idx2token.append(token)
            return idx
        return self.token2idx[token]

    def encode(self, token: str) -> int:
        return self.token2idx.get(token, UNK_IDX)

    def decode(self, idx: int) -> str:
        if 0 <= idx < len(self.idx2token):
            return self.idx2token[idx]
        return "<UNK>"

    def __len__(self) -> int:
        return len(self.idx2token)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.idx2token, f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path: Path) -> Vocab:
        with open(path, encoding="utf-8") as f:
            tokens_list = json.load(f)
        v = cls.__new__(cls)
        v.idx2token = tokens_list
        v.token2idx = {t: i for i, t in enumerate(tokens_list)}
        return v


def build_vocabs(
    gold_path: Path,
    weak_labels_path: Path,
    weak_confidence_threshold: float = 0.90,
) -> tuple[Vocab, Vocab]:
    """Build character and tag vocabularies from gold + weak data.

    Args:
        gold_path: Path to gold seed JSONL (with gold_label set).
        weak_labels_path: Path to probabilistic_labels.jsonl.
        weak_confidence_threshold: Min confidence for weak labels.

    Returns:
        (char_vocab, tag_vocab) tuple.
    """
    chars: set[str] = set()
    tags: set[str] = set()

    # Collect from gold
    with open(gold_path, encoding="utf-8") as f:
        for line in f:
            sent = json.loads(line)
            for token in sent["tokens"]:
                surface = token["surface"]
                for ch in surface:
                    chars.add(ch)
                gold = token.get("gold_label")
                if gold and isinstance(gold, str):
                    parts = gold.split()
                    for p in parts:
                        tags.add(p)

    # Collect from weak labels
    with open(weak_labels_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            conf = record.get("confidence", 0.0)
            if conf < weak_confidence_threshold:
                continue
            surface = record.get("surface", "")
            for ch in surface:
                chars.add(ch)
            label = record.get("predicted_label", "")
            if label:
                parts = label.split()
                for p in parts:
                    tags.add(p)

    char_vocab = Vocab(sorted(chars))
    tag_vocab = Vocab(sorted(tags))
    return char_vocab, tag_vocab


class MorphAtomizerDataset(Dataset):  # type: ignore[type-arg]
    """Training dataset combining gold and weak labels.

    Each sample is (char_indices, tag_indices) where:
    - char_indices: character-level encoding of the surface form
    - tag_indices: encoding of "root +TAG1 +TAG2 ..." as token sequence

    Sampling: gold_ratio of each batch from gold data (upsampled),
    remainder from high-confidence weak labels.
    """

    def __init__(
        self,
        gold_path: Path,
        weak_labels_path: Path,
        char_vocab: Vocab,
        tag_vocab: Vocab,
        weak_confidence_threshold: float = 0.90,
        max_char_len: int = 64,
        max_tag_len: int = 15,
        max_weak_samples: int = 0,
    ) -> None:
        self.char_vocab = char_vocab
        self.tag_vocab = tag_vocab
        self.max_char_len = max_char_len
        self.max_tag_len = max_tag_len

        self.samples: list[tuple[str, str]] = []  # (surface, label)

        # Load gold samples
        n_gold = 0
        with open(gold_path, encoding="utf-8") as f:
            for line in f:
                sent = json.loads(line)
                for token in sent["tokens"]:
                    gold = token.get("gold_label")
                    if gold and isinstance(gold, str):
                        self.samples.append((token["surface"], gold))
                        n_gold += 1

        # Load weak samples (optionally capped for CPU training)
        n_weak = 0
        with open(weak_labels_path, encoding="utf-8") as f:
            for line in f:
                if max_weak_samples > 0 and n_weak >= max_weak_samples:
                    break
                record = json.loads(line)
                conf = record.get("confidence", 0.0)
                if conf < weak_confidence_threshold:
                    continue
                label = record.get("predicted_label", "")
                if not label:
                    continue
                surface = record.get("surface", "")
                if not surface:
                    continue
                self.samples.append((surface, label))
                n_weak += 1

        self.n_gold = n_gold
        self.n_weak = n_weak

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        surface, label = self.samples[idx]

        # Encode characters: surface → [c1, c2, ..., EOS]
        char_ids = [self.char_vocab.encode(c) for c in surface]
        char_ids.append(EOS_IDX)
        if len(char_ids) > self.max_char_len:
            char_ids = char_ids[: self.max_char_len]
        char_ids += [PAD_IDX] * (self.max_char_len - len(char_ids))

        # Encode tags: label → [SOS, root, +TAG1, +TAG2, ..., EOS]
        parts = label.split()
        tag_ids = [SOS_IDX]
        for p in parts:
            tag_ids.append(self.tag_vocab.encode(p))
        tag_ids.append(EOS_IDX)
        if len(tag_ids) > self.max_tag_len:
            tag_ids = tag_ids[: self.max_tag_len]
        tag_ids += [PAD_IDX] * (self.max_tag_len - len(tag_ids))

        return (
            torch.tensor(char_ids, dtype=torch.long),
            torch.tensor(tag_ids, dtype=torch.long),
        )


class TieredCorpusDataset(Dataset):  # type: ignore[type-arg]
    """Dataset from the flat tiered corpus (tr_gold_morph_v1.jsonl or splits).

    Each line is a single token record with surface, label, and tier.
    Gold tokens can be weighted higher via sample_weight.
    """

    def __init__(
        self,
        path: Path,
        char_vocab: Vocab,
        tag_vocab: Vocab,
        max_char_len: int = 64,
        max_tag_len: int = 15,
        gold_weight: float = 2.0,
        root_vocab: dict[str, int] | None = None,
        augmenter: object | None = None,
    ) -> None:
        self.char_vocab = char_vocab
        self.tag_vocab = tag_vocab
        self.max_char_len = max_char_len
        self.max_tag_len = max_tag_len
        self.root_vocab = root_vocab
        self.augmenter = augmenter
        self.augment_count = 0

        self.samples: list[tuple[str, str]] = []
        self.weights: list[float] = []
        self.tiers: list[int] = []
        self.char_lengths: list[int] = []
        tier_counts: dict[str, int] = {}

        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                surface = rec.get("surface", "")
                label = rec.get("label", "")
                tier = rec.get("tier", "silver-auto")
                if not surface or not label:
                    continue
                self.samples.append((surface, label))
                w = gold_weight if tier == "gold" else 1.0
                self.weights.append(w)
                self.tiers.append(TIER_MAP.get(tier, 1))
                # Actual char count + EOS (before padding) for bucket batching
                self.char_lengths.append(min(len(surface) + 1, max_char_len))
                tier_counts[tier] = tier_counts.get(tier, 0) + 1

        self.tier_counts = tier_counts

        # Precompute per-tier index lists for curriculum sampling
        self.tier_indices: dict[int, list[int]] = {}
        for i, t in enumerate(self.tiers):
            self.tier_indices.setdefault(t, []).append(i)

    def _extract_root(self, label: str) -> int:
        """Return root_vocab index for first token in label.

        For example, ``"ev +PLU +ABL"`` returns ``root_vocab["ev"]``.

        Args:
            label: Full morphological label string.

        Returns:
            Root vocabulary index, or ``0`` when ``root_vocab`` is ``None``.
        """
        if not self.root_vocab:
            return 0
        root = label.split()[0] if label.strip() else ""
        return self.root_vocab.get(root, self.root_vocab.get("<UNK_ROOT>", 1))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        surface, label = self.samples[idx]

        if self.augmenter is not None:
            root_str = label.split()[0] if label.strip() else ""
            new_surface, _, _ = self.augmenter.augment(  # type: ignore[attr-defined]
                surface, root=root_str, tags=label,
            )
            if new_surface != surface:
                self.augment_count += 1
            surface = new_surface

        char_ids = [self.char_vocab.encode(c) for c in surface]
        char_ids.append(EOS_IDX)
        if len(char_ids) > self.max_char_len:
            char_ids = char_ids[: self.max_char_len]
        char_ids += [PAD_IDX] * (self.max_char_len - len(char_ids))

        parts = label.split()
        tag_ids = [SOS_IDX]
        for p in parts:
            tag_ids.append(self.tag_vocab.encode(p))
        tag_ids.append(EOS_IDX)
        if len(tag_ids) > self.max_tag_len:
            tag_ids = tag_ids[: self.max_tag_len]
        tag_ids += [PAD_IDX] * (self.max_tag_len - len(tag_ids))

        root_idx = self._extract_root(label)

        return (
            torch.tensor(char_ids, dtype=torch.long),
            torch.tensor(tag_ids, dtype=torch.long),
            self.tiers[idx],
            root_idx,
        )


# ---------------------------------------------------------------------------
# Word vocabulary helper
# ---------------------------------------------------------------------------

def build_word_vocab(path: Path, min_count: int = 2) -> dict[str, int]:
    """Build a surface-form vocabulary from a JSONL corpus file.

    Special tokens: ``"<PAD>"`` → 0, ``"<UNK>"`` → 1.
    Remaining tokens are sorted by descending frequency and assigned indices ≥ 2.

    Args:
        path: Path to a JSONL file with ``"surface"`` fields.
        min_count: Minimum occurrence count to include a surface in the vocab.

    Returns:
        Mapping of ``{surface_lower: index}``.
    """
    from collections import Counter

    counter: Counter[str] = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            surface = rec.get("surface", "").lower().strip()
            if surface:
                counter[surface] += 1

    vocab: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    for word, count in counter.most_common():
        if count >= min_count and word not in vocab:
            vocab[word] = len(vocab)
    return vocab


# ---------------------------------------------------------------------------
# ContextualTieredDataset
# ---------------------------------------------------------------------------

class ContextualTieredDataset(Dataset):  # type: ignore[type-arg]
    """TieredCorpusDataset extended with per-token sentence context.

    Groups records by ``sentence_id`` at init time, then for each token stores:

    - ``sentence_word_ids``: ``(max_sent_len,)`` word indices for the full sentence.
    - ``target_pos``: position of the current token in the sentence.
    - ``sentence_text``: raw space-joined sentence string (for BERTurkContext).

    All four context encoders work with this dataset:

    - ``Word2VecContext`` / ``SentenceBiGRUContext``: use ``sentence_word_ids``
      + ``target_pos``.
    - ``POSBigramContext``: caller extracts neighbour word ids from
      ``sentence_word_ids``.
    - ``BERTurkContext``: uses ``sentence_text`` for its own WordPiece
      tokenisation — no subclass needed.

    Records without a ``sentence_id`` are treated as single-token sentences.

    Returns a 7-tuple per sample::

        (char_ids, tag_ids, tier_int, root_idx, sentence_word_ids,
         target_pos, sentence_text)

    Args:
        path: Path to JSONL corpus file (same format as
            :class:`TieredCorpusDataset`).
        char_vocab: Character vocabulary.
        tag_vocab: Tag vocabulary.
        word_vocab: Surface-form vocabulary (built with
            :func:`build_word_vocab`).
        max_char_len: Maximum character sequence length.
        max_tag_len: Maximum tag sequence length (including SOS/EOS).
        max_sent_len: Maximum sentence length for context tensors.
        gold_weight: Sample weight multiplier for gold-tier tokens.
        root_vocab: Optional root vocabulary for :class:`TieredCorpusDataset`
            compatibility.
    """

    def __init__(
        self,
        path: Path,
        char_vocab: Vocab,
        tag_vocab: Vocab,
        word_vocab: dict[str, int],
        max_char_len: int = 64,
        max_tag_len: int = 15,
        max_sent_len: int = 64,
        gold_weight: float = 2.0,
        root_vocab: dict[str, int] | None = None,
    ) -> None:
        self.char_vocab = char_vocab
        self.tag_vocab = tag_vocab
        self.word_vocab = word_vocab
        self.max_char_len = max_char_len
        self.max_tag_len = max_tag_len
        self.max_sent_len = max_sent_len
        self.root_vocab = root_vocab

        # Core sample lists (mirrors TieredCorpusDataset)
        self.samples: list[tuple[str, str]] = []
        self.weights: list[float] = []
        self.tiers: list[int] = []

        # Context lists
        self.sentence_word_ids_list: list[torch.Tensor] = []
        self.target_positions: list[int] = []
        self.sentence_texts: list[str] = []

        # ---- First pass: group records by sentence_id ----
        sentences: dict[str, list[dict]] = {}
        _singleton_counter = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                sent_id = rec.get("sentence_id")
                if sent_id is None:
                    # Give each singleton a unique key
                    sent_id = f"__singleton_{_singleton_counter}__"
                    _singleton_counter += 1
                sentences.setdefault(sent_id, []).append(rec)

        # Sort within each sentence by token_idx
        for sent_recs in sentences.values():
            sent_recs.sort(key=lambda r: r.get("token_idx", 0))

        # ---- Second pass: build samples and context tensors ----
        tier_counts: dict[str, int] = {}

        for sent_recs in sentences.values():
            # Build word-id sequence for this sentence
            raw_ids = [
                word_vocab.get(r.get("surface", "").lower(), 1)  # 1 = UNK
                for r in sent_recs
            ]
            if len(raw_ids) > max_sent_len:
                raw_ids = raw_ids[:max_sent_len]
            padded_ids = raw_ids + [0] * (max_sent_len - len(raw_ids))
            sent_tensor = torch.tensor(padded_ids, dtype=torch.long)

            # Build raw sentence text (space-joined surfaces)
            sent_text = " ".join(r.get("surface", "") for r in sent_recs)

            for i, rec in enumerate(sent_recs):
                surface = rec.get("surface", "")
                label = rec.get("label", "")
                if not surface or not label:
                    continue

                tier = rec.get("tier", "silver-auto")
                w = gold_weight if tier == "gold" else 1.0

                self.samples.append((surface, label))
                self.weights.append(w)
                self.tiers.append(TIER_MAP.get(tier, 1))

                self.sentence_word_ids_list.append(sent_tensor)
                self.target_positions.append(min(i, max_sent_len - 1))
                self.sentence_texts.append(sent_text)

                tier_counts[tier] = tier_counts.get(tier, 0) + 1

        self.tier_counts = tier_counts

        # Precompute per-tier index lists (same as TieredCorpusDataset)
        self.tier_indices: dict[int, list[int]] = {}
        for i, t in enumerate(self.tiers):
            self.tier_indices.setdefault(t, []).append(i)

    def _extract_root(self, label: str) -> int:
        """Return root vocab index for the first token in label."""
        if not self.root_vocab:
            return 0
        root = label.split()[0] if label.strip() else ""
        return self.root_vocab.get(root, self.root_vocab.get("<UNK_ROOT>", 1))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> tuple[
        torch.Tensor, torch.Tensor, int, int, torch.Tensor, int, str
    ]:
        surface, label = self.samples[idx]

        # --- Character encoding ---
        char_ids = [self.char_vocab.encode(c) for c in surface]
        char_ids.append(EOS_IDX)
        if len(char_ids) > self.max_char_len:
            char_ids = char_ids[: self.max_char_len]
        char_ids += [PAD_IDX] * (self.max_char_len - len(char_ids))

        # --- Tag encoding ---
        parts = label.split()
        tag_ids = [SOS_IDX]
        for p in parts:
            tag_ids.append(self.tag_vocab.encode(p))
        tag_ids.append(EOS_IDX)
        if len(tag_ids) > self.max_tag_len:
            tag_ids = tag_ids[: self.max_tag_len]
        tag_ids += [PAD_IDX] * (self.max_tag_len - len(tag_ids))

        root_idx = self._extract_root(label)

        return (
            torch.tensor(char_ids, dtype=torch.long),
            torch.tensor(tag_ids, dtype=torch.long),
            self.tiers[idx],
            root_idx,
            self.sentence_word_ids_list[idx],   # (max_sent_len,)
            self.target_positions[idx],          # int
            self.sentence_texts[idx],            # str
        )


def contextual_collate(batch: list) -> tuple:
    """Collate function for :class:`ContextualTieredDataset`.

    Handles the ``sentence_text`` strings by keeping them as a plain list
    instead of attempting to stack them into a tensor.

    Args:
        batch: List of 7-tuples from :class:`ContextualTieredDataset`.

    Returns:
        7-tuple of:
        ``(char_ids, tag_ids, tiers, root_idxs, sent_word_ids,
           target_positions, sentence_texts)``
    """
    char_ids, tag_ids, tiers, roots, sent_wids, tgt_pos, sent_texts = zip(*batch)
    return (
        torch.stack(char_ids),
        torch.stack(tag_ids),
        torch.tensor(tiers, dtype=torch.long),
        torch.tensor(roots, dtype=torch.long),
        torch.stack(sent_wids),
        torch.tensor(tgt_pos, dtype=torch.long),
        list(sent_texts),
    )
