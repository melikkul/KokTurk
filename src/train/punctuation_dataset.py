"""Noktalama restorasyonu eğitim veri kümesi.

Kaynak: BOUN ağaç yapısı CoNLL-U dosyaları veya kök-türk gold CoNLL-U.
Her cümle için sözcük düzeyinde token ve her sözcüğü takip eden
noktalama işaretini çıkarır.

CoNLL-U'da noktalama sözcükleri ``UPOS=PUNCT`` ile işaretlenir.
Bunlar sözcük başına etiketlere dönüştürülür: her PUNCT-olmayan
sözcükten sonra hangi noktalama geldiği.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from kokturk.models.punctuation_restorer import _SYMBOL_TO_LABEL, PUNCT_LABELS

# Varsayılan CoNLL-U arama yolları
_DEFAULT_CONLLU_PATHS = [
    "data/external/boun_treebank",
    "data/gold/tr_gold_morph_v1.conllu",
]


def find_conllu_file(data_path: str | None = None) -> Path:
    """CoNLL-U dosyasını bul.

    Verilen yol yoksa varsayılan konumları dener.

    Raises:
        FileNotFoundError: Hiçbir konumda CoNLL-U bulunamazsa.
    """
    if data_path:
        p = Path(data_path)
        if p.exists():
            return p

    for candidate in _DEFAULT_CONLLU_PATHS:
        p = Path(candidate)
        if p.is_file():
            return p
        if p.is_dir():
            conllu_files = list(p.glob("*.conllu"))
            if conllu_files:
                return conllu_files[0]

    msg = (
        "CoNLL-U dosyası bulunamadı. Aranan konumlar:\n"
        + "\n".join(f"  - {p}" for p in _DEFAULT_CONLLU_PATHS)
    )
    if data_path:
        msg = f"'{data_path}' bulunamadı. " + msg
    raise FileNotFoundError(msg)


class PunctuationDataset(Dataset):
    """Her örnek = bir cümle ve sözcük başına noktalama etiketleri.

    Args:
        data_path: CoNLL-U veya JSONL dosya yolu.
        max_len: Cümle başına maksimum sözcük sayısı.
    """

    def __init__(
        self,
        data_path: str,
        max_len: int = 64,
    ) -> None:
        self.max_len = max_len
        self.samples: list[dict] = []

        path = Path(data_path)
        if path.suffix == ".conllu":
            self._load_conllu(path)
        else:
            self._load_jsonl(path)

    def _load_conllu(self, path: Path) -> None:
        """CoNLL-U dosyasından sözcük + noktalama çiftlerini ayrıştır."""
        current_words: list[str] = []
        current_puncts: list[int] = []

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    if current_words:
                        self.samples.append({
                            "words": current_words,
                            "punct_labels": current_puncts,
                            "sentence": " ".join(current_words),
                        })
                        current_words = []
                        current_puncts = []
                    continue

                if line.startswith("#"):
                    continue

                fields = line.split("\t")
                if len(fields) < 4:
                    continue

                # Çok sözcüklü aralıkları atla (1-2 gibi)
                if "-" in fields[0]:
                    continue

                form = fields[1]
                upos = fields[3]

                if upos == "PUNCT":
                    # Noktalamayi önceki sözcüğe iliştir
                    if current_puncts:
                        label_name = _SYMBOL_TO_LABEL.get(form, "none")
                        current_puncts[-1] = PUNCT_LABELS.get(label_name, 0)
                else:
                    current_words.append(form)
                    current_puncts.append(0)  # varsayılan: noktalama yok

        if current_words:
            self.samples.append({
                "words": current_words,
                "punct_labels": current_puncts,
                "sentence": " ".join(current_words),
            })

    def _load_jsonl(self, path: Path) -> None:
        """Cümle bazlı JSONL'den yükle."""
        sentences: dict[str, list[dict]] = defaultdict(list)
        with open(path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                sid = item.get("sentence_id", "")
                if sid:
                    sentences[sid].append(item)

        punct_symbol_map = {",": 1, ".": 2, "?": 3, "!": 4, ":": 5, ";": 6}

        for _sid, tokens in sentences.items():
            tokens.sort(key=lambda x: x.get("token_idx", 0))
            words: list[str] = []
            puncts: list[int] = []

            for token in tokens:
                surface = token["surface"]
                if surface in ".,?!:;":
                    if puncts:
                        puncts[-1] = punct_symbol_map.get(surface, 0)
                else:
                    words.append(surface)
                    puncts.append(0)

            if words:
                self.samples.append({
                    "words": words,
                    "punct_labels": puncts,
                    "sentence": " ".join(words),
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        words = sample["words"][: self.max_len]
        labels = sample["punct_labels"][: self.max_len]

        pad_len = self.max_len - len(words)
        labels_tensor = torch.tensor(labels + [-1] * pad_len, dtype=torch.long)

        return {
            "sentence": " ".join(words),
            "words": words,
            "labels": labels_tensor,
            "num_words": len(words),
        }


def punctuation_collate(batch: list[dict]) -> dict:
    """Değişken uzunluktaki cümleleri topla."""
    return {
        "sentences": [b["sentence"] for b in batch],
        "labels": torch.stack([b["labels"] for b in batch]),
        "num_words": torch.tensor([b["num_words"] for b in batch]),
    }
