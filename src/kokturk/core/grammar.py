"""Birleşik Türkçe dilbilgisi denetim arayüzü.

Üç modülü birleştirir:
1. TurkishSpellChecker — yazım ve ASCII düzeltme
2. TurkishGrammarChecker — uyum ve yapı kontrolü
3. PunctuationRestorer — noktalama düzeltme

Kullanım::

    from kokturk import GrammarChecker

    checker = GrammarChecker()
    result = checker.check("Turkiyede benim ev cok guzel yarin yagmur yagacak")
    print(result.corrected)
    # → "Türkiye'de benim evim çok güzel, yarın yağmur yağacak."

    for issue in result.issues:
        print(f"  {issue.severity}: {issue.message}")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GrammarResult:
    """Dilbilgisi denetim sonucu."""

    original: str
    corrected: str
    issues: list[Any] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Ciddi hata (error) var mı?"""
        return any(
            getattr(i, "severity", None) == "error"
            or getattr(i, "error_type", None) is not None
            for i in self.issues
        )

    @property
    def error_count(self) -> int:
        """Toplam sorun sayısı."""
        return len(self.issues)


class GrammarChecker:
    """Hepsi-bir-arada Türkçe dilbilgisi denetleyicisi.

    kök-türk morfolojik analizi ile şunları birleştirir:
    - Yazım denetimi (kök-türk sözlüğü + edit mesafesi)
    - Dilbilgisi kuralları (uyum kontrolü)
    - Noktalama restorasyonu (BERTurk tabanlı)

    Args:
        berturk_path: BERTurk model dizini.
        punctuation_model: Eğitilmiş noktalama modeli checkpoint yolu.
        enable_punctuation: Noktalama denetimini etkinleştir.
        enable_grammar: Dilbilgisi denetimini etkinleştir.
        enable_spelling: Yazım denetimini etkinleştir.
    """

    def __init__(
        self,
        berturk_path: str = "models/berturk",
        punctuation_model: str | None = None,
        enable_punctuation: bool = True,
        enable_grammar: bool = True,
        enable_spelling: bool = True,
    ) -> None:
        self._spell_checker = None
        self._grammar_checker = None
        self._punct_restorer = None

        if enable_spelling:
            from kokturk.models.spell_checker import TurkishSpellChecker

            self._spell_checker = TurkishSpellChecker()

        if enable_grammar:
            from kokturk.models.grammar_checker import TurkishGrammarChecker

            # Yazım denetleyicisi ile aynı çözümleyiciyi paylaş
            analyzer = (
                self._spell_checker._analyzer
                if self._spell_checker is not None
                else None
            )
            self._grammar_checker = TurkishGrammarChecker(
                analyzer=analyzer, berturk_path=berturk_path
            )

        if enable_punctuation and punctuation_model:
            import torch

            from kokturk.models.punctuation_restorer import PunctuationRestorer

            self._punct_restorer = PunctuationRestorer(berturk_path)
            ckpt = torch.load(punctuation_model, map_location="cpu")
            self._punct_restorer.load_state_dict(ckpt["model_state_dict"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, text: str) -> GrammarResult:
        """Tüm etkin denetimleri çalıştır.

        Args:
            text: Denetlenecek Türkçe metin.

        Returns:
            Düzeltilmiş metin ve sorun listesi içeren sonuç.
        """
        all_issues: list[Any] = []
        corrected = text

        # 1. Önce yazım (karakterleri düzeltir, diğer kontroller için zemin hazırlar)
        if self._spell_checker is not None:
            spell_results = self._spell_checker.check(corrected)
            all_issues.extend(spell_results)
            corrected = self._spell_checker.correct(corrected)

        # 2. Dilbilgisi kontrolü (düzeltilmiş metin üzerinde)
        if self._grammar_checker is not None:
            grammar_results = self._grammar_checker.check(corrected)
            all_issues.extend(grammar_results)

        # 3. Noktalama en son
        if self._punct_restorer is not None:
            punct_results = self._punct_restorer.check(corrected)
            all_issues.extend(punct_results)
            corrected = self._punct_restorer.restore(corrected)

        return GrammarResult(
            original=text,
            corrected=corrected,
            issues=all_issues,
        )

    def correct(self, text: str) -> str:
        """Yüksek güvenilirlikli düzeltmeleri otomatik uygula.

        Args:
            text: Düzeltilecek Türkçe metin.

        Returns:
            Düzeltilmiş metin.
        """
        result = self.check(text)
        return result.corrected
