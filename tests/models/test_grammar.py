"""Dilbilgisi denetleyicisi modülleri için testler.

Kapsam: ünlü uyumu, yazım denetleyicisi, dilbilgisi denetleyicisi,
noktalama restorasyonu ve birleşik arayüz.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import torch

# ---------------------------------------------------------------------------
# 1. Ünlü uyumu (4-way)
# ---------------------------------------------------------------------------


class TestVowelHarmony:
    def test_valid_front(self):
        """Ön ünlü sözcük — uyum geçerli."""
        from kokturk.core.phonology import check_vowel_harmony

        # "gülüm" → ü+ü, her ikisi ön-yuvarlak, tam uyum
        result = check_vowel_harmony("gülüm")
        assert result.ok is True
        assert result.severity == "none"

    def test_valid_back(self):
        """Arka ünlü sözcük — uyum geçerli."""
        from kokturk.core.phonology import check_vowel_harmony

        result = check_vowel_harmony("araba")
        assert result.ok is True
        assert result.severity == "none"

    def test_two_way_violation(self):
        """Ön/arka karışımı → error."""
        from kokturk.core.phonology import check_vowel_harmony

        # Yapay sözcük: ön + arka ünlü
        result = check_vowel_harmony("gülar")
        assert result.ok is False
        assert result.severity == "error"

    def test_four_way_warning(self):
        """Yuvarlak/düz karışımı (ama ön/arka tutarlı) → warning."""
        from kokturk.core.phonology import check_vowel_harmony

        # Ön ünlüler ama yuvarlak + düz karışımı: ö + e
        result = check_vowel_harmony("göle")
        # ö=front-rounded, e=front-unrounded → 4-way warning
        assert result.ok is False
        assert result.severity == "warning"

    def test_single_vowel(self):
        """Tek ünlü — her zaman geçerli."""
        from kokturk.core.phonology import check_vowel_harmony

        result = check_vowel_harmony("ev")
        assert result.ok is True


# ---------------------------------------------------------------------------
# 2. Yazım denetleyicisi
# ---------------------------------------------------------------------------


class TestSpellChecker:
    def test_deasciify_guzel(self):
        """'guzel' → 'güzel' dönüşümü."""
        from kokturk.models.spell_checker import TurkishSpellChecker

        checker = TurkishSpellChecker()
        results = checker.check("guzel")
        assert any(s.suggestion == "güzel" for s in results), (
            f"'güzel' beklendi, bulunan: {[s.suggestion for s in results]}"
        )

    def test_casing_istanbul(self):
        """'Istanbul' → 'İstanbul' büyük I düzeltmesi."""
        from kokturk.models.spell_checker import TurkishSpellChecker

        checker = TurkishSpellChecker()
        results = checker.check("Istanbul")
        assert any(s.suggestion == "İstanbul" for s in results), (
            f"'İstanbul' beklendi, bulunan: {[s.suggestion for s in results]}"
        )

    def test_apostrophe_turkiyede(self):
        """'Turkiyede' → deasciify ile düzeltme."""
        from kokturk.models.spell_checker import TurkishSpellChecker

        checker = TurkishSpellChecker()
        # "Turkiyede" ASCII Türkçe karakter içerir (u→ü, i→İ vb.)
        # Deasciify akışına girmeli
        results = checker.check("Turkiyede")
        # En az bir öneri olmalı (deasciify)
        # Not: Zeyrek "Turkiyede"yi tanıyabilir ama ASCII kontrol
        # deasciify akışını tetikler
        assert len(results) > 0, (
            "Turkiyede için öneri bekleniyor"
            f" — has_ascii={checker._has_ascii_turkish_chars('Turkiyede')}"
        )

    def test_correct_applies_high_confidence(self):
        """correct() yalnızca yüksek güvenilirlikli düzeltmeleri uygular."""
        from kokturk.models.spell_checker import TurkishSpellChecker

        checker = TurkishSpellChecker()
        corrected = checker.correct("Istanbul")
        assert corrected == "İstanbul"

    def test_valid_word_no_suggestion(self):
        """Geçerli sözcük için öneri yapılmaz."""
        from kokturk.models.spell_checker import TurkishSpellChecker

        checker = TurkishSpellChecker()
        results = checker.check("ev")
        assert len(results) == 0


# ---------------------------------------------------------------------------
# 3. Dilbilgisi denetleyicisi
# ---------------------------------------------------------------------------


class TestGrammarChecker:
    def test_genitive_possessive(self):
        """'benim ev' → tamlayan-iyelik uyum hatası."""
        from kokturk.models.grammar_checker import TurkishGrammarChecker

        checker = TurkishGrammarChecker()
        errors = checker.check("benim ev")
        agreement_errors = [e for e in errors if e.error_type == "agreement"]
        assert len(agreement_errors) > 0, (
            "Tamlayan-iyelik uyum hatası bekleniyor"
        )

    def test_no_false_positive_possessive(self):
        """'benim evim' → uyum hatası olmamalı."""
        from kokturk.models.grammar_checker import TurkishGrammarChecker

        checker = TurkishGrammarChecker()
        errors = checker.check("benim evim")
        agreement_errors = [e for e in errors if e.error_type == "agreement"]
        assert len(agreement_errors) == 0, (
            f"Hatalı uyarı: {[e.message for e in agreement_errors]}"
        )

    def test_de_da_voiceless_rule(self):
        """Sert ünsüz sonrası 'de' → 'te' önerisi."""
        from kokturk.models.grammar_checker import TurkishGrammarChecker

        checker = TurkishGrammarChecker()
        # "kitap de" → "kitap" sert ünsüz p ile bitiyor
        errors = checker.check("kitap de")
        de_da_errors = [
            e for e in errors
            if e.error_type == "agreement" and e.suggestion in ("te", "ta")
        ]
        assert len(de_da_errors) > 0, (
            f"de/da→te/ta hatası bekleniyor, bulunan: {errors}"
        )

    def test_well_formed_sentence(self):
        """Düzgün cümle için kritik hata yok."""
        from kokturk.models.grammar_checker import TurkishGrammarChecker

        checker = TurkishGrammarChecker()
        errors = checker.check("bugün hava güzel")
        # Uyum hatası olmamalı (morfoloji uyarıları olabilir)
        agreement_errors = [e for e in errors if e.error_type == "agreement"]
        assert len(agreement_errors) == 0


# ---------------------------------------------------------------------------
# 4. Noktalama etiketleri
# ---------------------------------------------------------------------------


class TestPunctuationLabels:
    def test_label_count(self):
        """7 noktalama sınıfı tanımlı."""
        from kokturk.models.punctuation_restorer import PUNCT_LABELS

        assert len(PUNCT_LABELS) == 7

    def test_symbol_mapping(self):
        """Sembol eşlemesi doğru."""
        from kokturk.models.punctuation_restorer import PUNCT_SYMBOLS

        assert PUNCT_SYMBOLS[0] == ""
        assert PUNCT_SYMBOLS[1] == ","
        assert PUNCT_SYMBOLS[2] == "."
        assert PUNCT_SYMBOLS[3] == "?"


# ---------------------------------------------------------------------------
# 5. Noktalama restorasyonu modeli (mock BERT)
# ---------------------------------------------------------------------------


def _make_mock_punct_restorer():
    """Mock BERTurk ile PunctuationRestorer oluştur."""
    from kokturk.models.punctuation_restorer import PunctuationRestorer

    seq_len = 20

    mock_output = MagicMock()
    mock_output.last_hidden_state = torch.randn(1, seq_len, 768)

    mock_bert = MagicMock()
    mock_bert.return_value = mock_output
    mock_bert.parameters = lambda: iter([])

    mock_encoding = MagicMock()
    mock_encoding.__getitem__ = MagicMock(
        side_effect=lambda k: {
            "input_ids": torch.zeros(1, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
        }[k]
    )
    mock_encoding.word_ids = MagicMock(
        side_effect=lambda b: [None] + list(range(seq_len - 2)) + [None],
    )

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = mock_encoding

    model = PunctuationRestorer(
        bert_model=mock_bert,
        tokenizer=mock_tokenizer,
    )
    return model


class TestPunctuationRestorer:
    def test_forward_shape(self):
        """Forward logits şekli (B, N, 7) olmalı."""
        from kokturk.models.punctuation_restorer import PUNCT_LABELS

        model = _make_mock_punct_restorer()
        bs, nw = 2, 5
        embeddings = torch.randn(bs, nw, 768)
        logits, loss = model(embeddings)

        assert logits.shape == (bs, nw, len(PUNCT_LABELS))
        assert loss is None

    def test_forward_with_labels(self):
        """Etiketlerle forward loss döndürmeli."""
        model = _make_mock_punct_restorer()
        bs, nw = 2, 5
        embeddings = torch.randn(bs, nw, 768)
        labels = torch.zeros(bs, nw, dtype=torch.long)

        logits, loss = model(embeddings, labels)
        assert loss is not None
        assert loss.item() > 0

    def test_loss_decreases(self):
        """5 gradient adımında loss düşmeli."""
        model = _make_mock_punct_restorer()
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )

        bs, nw = 4, 8
        embeddings = torch.randn(bs, nw, 768)
        labels = torch.randint(0, 7, (bs, nw))

        losses: list[float] = []
        for _ in range(5):
            optimizer.zero_grad()
            _, loss = model(embeddings, labels)
            assert loss is not None
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss düşmedi: {losses[0]:.4f} → {losses[-1]:.4f}"
        )


# ---------------------------------------------------------------------------
# 6. Birleşik arayüz
# ---------------------------------------------------------------------------


class TestUnifiedGrammarChecker:
    def test_check_aggregates(self):
        """check() tüm modüllerden sorunları toplar."""
        from kokturk.core.grammar import GrammarChecker

        checker = GrammarChecker(enable_punctuation=False)
        result = checker.check("Istanbul guzel")
        assert result.error_count > 0

    def test_correct_returns_fixed(self):
        """correct() düzeltilmiş metin döndürür."""
        from kokturk.core.grammar import GrammarChecker

        checker = GrammarChecker(enable_punctuation=False)
        corrected = checker.correct("Istanbul")
        assert "İstanbul" in corrected

    def test_result_dataclass(self):
        """GrammarResult özellikleri doğru çalışır."""
        from kokturk.core.grammar import GrammarResult

        r = GrammarResult(original="test", corrected="test", issues=[])
        assert not r.has_errors
        assert r.error_count == 0

    def test_disabled_modules_skipped(self):
        """Devre dışı modüller atlanır."""
        from kokturk.core.grammar import GrammarChecker

        checker = GrammarChecker(
            enable_punctuation=False,
            enable_grammar=False,
            enable_spelling=False,
        )
        result = checker.check("test")
        assert result.error_count == 0
        assert result.corrected == "test"
