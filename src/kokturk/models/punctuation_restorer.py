"""Türkçe noktalama işareti restorasyonu — donmuş BERTurk tabanlı.

Her sözcük için ardından gelmesi gereken noktalama işaretini tahmin eder.
Tahmini mevcut noktalama ile karşılaştırarak hataları bulur.

Mimari::

    Cümle → BERTurk (donmuş) → sözcük başına gömme (768d)
                                        ↓
                                  Linear(768 → 256) + LayerNorm
                                        ↓
                                  Linear(256 → 7)  ← 7 noktalama sınıfı
                                        ↓
                            {yok, virgül, nokta, soru, ünlem, iki nokta, noktalı virgül}

Eğitim verisi: BOUN ağaç yapısı (CoNLL-U formatı, noktalama etiketli)
Eğitilebilir parametreler: ~200K
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

# Noktalama sınıfları
PUNCT_LABELS: dict[str, int] = {
    "none": 0,
    "comma": 1,
    "period": 2,
    "question": 3,
    "excl": 4,
    "colon": 5,
    "semicolon": 6,
}

PUNCT_SYMBOLS: dict[int, str] = {
    0: "",
    1: ",",
    2: ".",
    3: "?",
    4: "!",
    5: ":",
    6: ";",
}

# Ters eşleme: sembol → sınıf adı
_SYMBOL_TO_LABEL: dict[str, str] = {
    ",": "comma",
    ".": "period",
    "?": "question",
    "!": "excl",
    ":": "colon",
    ";": "semicolon",
    "...": "period",
    "\u2026": "period",  # …
}

# Minimum güven eşiği — altındaki tahminler göz ardı edilir
_CONFIDENCE_THRESHOLD = 0.7


@dataclass(frozen=True, slots=True)
class PunctuationSuggestion:
    """Bir noktalama düzeltme önerisi."""

    position: int
    after: str
    expected: str
    actual: str
    action: str  # "add" | "remove" | "replace"
    confidence: float


class PunctuationRestorer(nn.Module):
    """Donmuş BERTurk bağlamını kullanarak sözcük başına noktalama tahmini.

    Kullanım::

        restorer = PunctuationRestorer("models/berturk")
        corrections = restorer.check("Bugün hava güzel yarın yağmur yağacak")
        restored = restorer.restore("Bugün hava güzel yarın yağmur yağacak")

    Test için bert_model ve tokenizer enjekte edilebilir::

        restorer = PunctuationRestorer(
            bert_model=mock_bert,
            tokenizer=mock_tokenizer,
        )
    """

    def __init__(
        self,
        bert_path: str = "models/berturk",
        dropout: float = 0.3,
        bert_model: object | None = None,
        tokenizer: object | None = None,
        skip_bert_loading: bool = False,
    ) -> None:
        super().__init__()
        self.bert_dim = 768
        num_classes = len(PUNCT_LABELS)

        # Donmuş BERTurk
        if skip_bert_loading:
            self.bert = None  # type: ignore[assignment]
            self.tokenizer = None  # type: ignore[assignment]
        elif bert_model is not None:
            self.bert = bert_model  # type: ignore[assignment]
            self.tokenizer = tokenizer  # type: ignore[assignment]
        else:
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
            self.bert = AutoModel.from_pretrained(bert_path)
            for p in self.bert.parameters():
                p.requires_grad = False
            self.bert.eval()

        # Sınıflandırma başlığı (~200K parametre)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert_dim, 256),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        word_embeddings: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Sözcük gömmeleri üzerinden noktalama tahmini.

        Args:
            word_embeddings: ``(B, N, 768)`` — sözcük başına BERTurk gömmeleri.
            labels: ``(B, N)`` — noktalama sınıf indeksleri, -1 ile dolgulu.

        Returns:
            ``(logits, loss)`` çifti.  *loss*, etiketler verilmezse ``None``.
        """
        logits = self.classifier(word_embeddings)  # (B, N, num_classes)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, len(PUNCT_LABELS)),
                labels.view(-1),
                ignore_index=-1,
            )
        return logits, loss

    # ------------------------------------------------------------------
    # BERTurk gömmeleri
    # ------------------------------------------------------------------

    def get_word_embeddings(self, sentence: str) -> torch.Tensor:
        """Cümledeki her sözcük için BERTurk gömmesi al.

        Alt-sözcük → sözcük hizalaması için her sözcüğün ilk
        alt-sözcük gömmesini kullanır.

        Args:
            sentence: Boşlukla ayrılmış sözcüklerden oluşan cümle.

        Returns:
            ``(num_words, 768)`` tensör.
        """
        words = sentence.split()
        if not words:
            return torch.zeros(0, self.bert_dim)

        with torch.no_grad():
            encoding = self.tokenizer(
                sentence,
                return_tensors="pt",
                truncation=True,
                max_length=128,
            )
            output = self.bert(**encoding)
            hidden = output.last_hidden_state[0]  # (seq_len, 768)

            # Her sözcüğün ilk alt-sözcük gömmesi
            word_ids = encoding.word_ids(0)
            word_embeddings: list[torch.Tensor] = []
            seen: set[int] = set()
            for i, wid in enumerate(word_ids):
                if wid is not None and wid not in seen:
                    seen.add(wid)
                    word_embeddings.append(hidden[i])

            # Sözcük sayısına dolgu/kırpma
            while len(word_embeddings) < len(words):
                word_embeddings.append(torch.zeros(self.bert_dim))
            word_embeddings = word_embeddings[: len(words)]

        return torch.stack(word_embeddings)

    # ------------------------------------------------------------------
    # Çıkarım
    # ------------------------------------------------------------------

    def check(self, text: str) -> list[PunctuationSuggestion]:
        """Metindeki noktalama hatalarını denetle.

        Yalnızca güven skoru >= {_CONFIDENCE_THRESHOLD} olan öneriler döner.

        Args:
            text: Ham Türkçe metin (yanlış/eksik noktalama olabilir).

        Returns:
            Düzeltme önerileri listesi.
        """
        # Sözcükleri ve mevcut noktalamaları ayır
        tokens = re.findall(r"\w+|[^\w\s]", text)

        words: list[str] = []
        actual_punct: dict[int, str] = {}
        word_idx = 0

        for token in tokens:
            if re.match(r"\w+", token):
                words.append(token)
                actual_punct[word_idx] = ""
                word_idx += 1
            elif word_idx > 0:
                actual_punct[word_idx - 1] = token

        if not words:
            return []

        # BERTurk gömmeleri
        clean_sentence = " ".join(words)
        embeddings = self.get_word_embeddings(clean_sentence).unsqueeze(0)

        # Tahmin
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(embeddings)
            probs = F.softmax(logits[0], dim=-1)
            preds = probs.argmax(dim=-1)

        # Karşılaştır
        corrections: list[PunctuationSuggestion] = []
        for i, (word, pred_idx) in enumerate(zip(words, preds, strict=True)):
            confidence = probs[i].max().item()
            if confidence < _CONFIDENCE_THRESHOLD:
                continue

            expected = PUNCT_SYMBOLS.get(pred_idx.item(), "")
            actual = actual_punct.get(i, "")

            if expected != actual:
                if expected and not actual:
                    action = "add"
                elif not expected and actual:
                    action = "remove"
                else:
                    action = "replace"

                corrections.append(PunctuationSuggestion(
                    position=i,
                    after=word,
                    expected=expected,
                    actual=actual,
                    action=action,
                    confidence=confidence,
                ))

        return corrections

    def restore(self, text: str) -> str:
        """Noktalamasız metne doğru noktalama ekle.

        Yalnızca güven skoru >= {_CONFIDENCE_THRESHOLD} olan tahminler uygulanır.

        Args:
            text: ``"Bugün hava güzel yarın yağmur yağacak"``

        Returns:
            ``"Bugün hava güzel, yarın yağmur yağacak."``
        """
        words = text.split()
        if not words:
            return text

        embeddings = self.get_word_embeddings(text).unsqueeze(0)

        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(embeddings)
            probs = F.softmax(logits[0], dim=-1)
            preds = probs.argmax(dim=-1)

        result: list[str] = []
        for word, pred_idx, prob_row in zip(words, preds, probs, strict=True):
            result.append(word)
            confidence = prob_row.max().item()
            if confidence >= _CONFIDENCE_THRESHOLD:
                punct = PUNCT_SYMBOLS.get(pred_idx.item(), "")
                if punct:
                    result.append(punct)

        # Noktalamaları sözcüklere yapıştır
        text_out = " ".join(result)
        for sym in ",.?!:;":
            text_out = text_out.replace(f" {sym}", sym)
        return text_out
