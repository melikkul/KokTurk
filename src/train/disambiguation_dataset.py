"""PyTorch dataset for morphological disambiguation training.

Each sample represents a single token within its sentence context, paired
with Zeyrek candidate parses and a gold index indicating the correct parse.

The dataset pre-computes Zeyrek candidates at init time (slow but done once)
and caches them for fast epoch iteration.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from train.datasets import PAD_IDX, UNK_IDX, Vocab

logger = logging.getLogger(__name__)


class DisambiguationDataset(Dataset):
    """Dataset for BERTurk disambiguation training.

    Each sample = one token with:
        - sentence_text: full sentence for BERTurk context
        - target_position: word position in sentence (0-based)
        - candidates: list of canonical parse strings
        - gold_idx: index of correct candidate

    For tokens where gold is NOT in Zeyrek candidates, the gold label
    is appended as an extra candidate so the model can learn to select it.
    """

    def __init__(
        self,
        data_path: Path | str,
        tag_vocab: Vocab,
        max_candidates: int = 10,
        max_parse_len: int = 15,
    ) -> None:
        self.tag_vocab = tag_vocab
        self.max_candidates = max_candidates
        self.max_parse_len = max_parse_len
        self.samples: list[dict] = []
        self._load(Path(data_path))

    def _load(self, data_path: Path) -> None:
        """Load data, group by sentence, generate Zeyrek candidates."""
        from kokturk.core.analyzer import ZeyrekBackend

        # Silence Zeyrek's verbose per-token WARNING output — it dumps
        # every parse candidate to stderr which fills up log files.
        logging.getLogger("zeyrek").setLevel(logging.ERROR)
        logging.getLogger("zeyrek.rulebasedanalyzer").setLevel(logging.ERROR)

        backend = ZeyrekBackend()

        # Group tokens by sentence_id, sort by token_idx
        sentences: dict[str, list[dict]] = defaultdict(list)
        for line in data_path.open():
            item = json.loads(line)
            sid = item.get("sentence_id", "")
            if sid:
                sentences[sid].append(item)

        # Sort tokens within each sentence
        for sid in sentences:
            sentences[sid].sort(key=lambda x: x.get("token_idx", 0))

        # Stats
        total = 0
        gold_found = 0
        gold_added = 0
        unambiguous = 0
        skipped_unk = 0

        for sid, tokens in sentences.items():
            sentence_text = " ".join(t["surface"] for t in tokens)

            for token in tokens:
                surface = token["surface"]
                gold_label = token["label"]

                # Skip tokens with Unk gold label — no learning signal
                if gold_label == "Unk":
                    skipped_unk += 1
                    continue

                total += 1

                # Get Zeyrek candidates
                analyses = backend.analyze(surface)
                candidates = [a.to_str() for a in analyses]

                # Deduplicate candidates (preserving order)
                seen: set[str] = set()
                deduped: list[str] = []
                for c in candidates:
                    if c not in seen:
                        seen.add(c)
                        deduped.append(c)
                candidates = deduped

                # Find gold index
                gold_idx = -1
                for i, cand in enumerate(candidates):
                    if cand == gold_label:
                        gold_idx = i
                        break

                if gold_idx >= 0:
                    gold_found += 1
                else:
                    # Gold not in candidates — add it as extra candidate
                    candidates.append(gold_label)
                    gold_idx = len(candidates) - 1
                    gold_added += 1

                num_candidates = len(candidates)

                if num_candidates == 1:
                    unambiguous += 1

                # Truncate to max_candidates, keeping gold
                if num_candidates > self.max_candidates:
                    if gold_idx >= self.max_candidates:
                        # Swap gold into last valid position
                        candidates[self.max_candidates - 1] = candidates[gold_idx]
                        gold_idx = self.max_candidates - 1
                    candidates = candidates[: self.max_candidates]
                    num_candidates = self.max_candidates

                self.samples.append({
                    "sentence_text": sentence_text,
                    "target_position": token.get("token_idx", 0),
                    "surface": surface,
                    "candidates": candidates,
                    "gold_idx": gold_idx,
                    "num_candidates": num_candidates,
                })

        # Report stats
        ambiguous = total - unambiguous
        logger.info(
            "DisambiguationDataset: %d tokens (%d skipped Unk), "
            "%d unambiguous (%.1f%%), %d ambiguous",
            total, skipped_unk, unambiguous,
            unambiguous / total * 100 if total else 0, ambiguous,
        )
        logger.info(
            "  Gold in Zeyrek: %d (%.1f%%), Gold added: %d (%.1f%%)",
            gold_found, gold_found / total * 100 if total else 0,
            gold_added, gold_added / total * 100 if total else 0,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # Encode each candidate as tag_vocab token IDs
        candidate_ids = torch.zeros(
            self.max_candidates, self.max_parse_len, dtype=torch.long,
        )
        candidate_mask = torch.zeros(self.max_candidates, dtype=torch.bool)

        for i, cand in enumerate(sample["candidates"]):
            parts = cand.split()
            for j, part in enumerate(parts[: self.max_parse_len]):
                candidate_ids[i, j] = self.tag_vocab.encode(part)
            candidate_mask[i] = True

        return {
            "sample_idx": idx,  # for BERTurk cache lookup
            "sentence_text": sample["sentence_text"],
            "target_position": sample["target_position"],
            "surface": sample["surface"],
            "candidate_ids": candidate_ids,
            "candidate_mask": candidate_mask,
            "gold_idx": sample["gold_idx"],
            "num_candidates": sample["num_candidates"],
        }


def disambiguation_collate(batch: list[dict]) -> dict:
    """Collate function for DisambiguationDataset.

    Handles variable-length sentence strings alongside tensor fields.
    """
    return {
        "sample_indices": torch.tensor(
            [b["sample_idx"] for b in batch], dtype=torch.long,
        ),
        "sentence_texts": [b["sentence_text"] for b in batch],
        "target_positions": torch.tensor(
            [b["target_position"] for b in batch], dtype=torch.long,
        ),
        "surfaces": [b["surface"] for b in batch],
        "candidate_ids": torch.stack([b["candidate_ids"] for b in batch]),
        "candidate_mask": torch.stack([b["candidate_mask"] for b in batch]),
        "gold_indices": torch.tensor(
            [b["gold_idx"] for b in batch], dtype=torch.long,
        ),
        "num_candidates": torch.tensor(
            [b["num_candidates"] for b in batch], dtype=torch.long,
        ),
    }
