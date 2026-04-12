"""Türkçe dilbilgisi denetleyicisi — kök-türk morfolojisi + kural tabanlı.

Üç tür dilbilgisi hatası tespit edilir:

1. MORFOLOJİK HATALAR — ünlü uyumu ihlalleri, tanınmayan kelimeler
2. UYUM HATALARI — tamlayan-iyelik uyumu, de/da-te/ta kuralı
3. ANOMALİLER — BERTurk perplexity tabanlı (v2'de eklenecek)

Bu modül şunları YAPMAZ:
- Üslup veya okunabilirlik kontrolü
- Alternatif kelime önerisi
- Anlamsal hata tespiti
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kokturk.core.analyzer import MorphoAnalyzer


# Sert ünsüzler — de/da yerine te/ta gerektiren sesler
_VOICELESS = frozenset("çfhkpsşt")

# Tamlayan zamirleri (genitif eki taşıyan)
_GENITIVE_PRONOUNS = frozenset(
    ("benim", "senin", "onun", "bizim", "sizin", "onların")
)


@dataclass(frozen=True, slots=True)
class GrammarError:
    """Tespit edilen bir dilbilgisi hatası."""

    position: int
    token: str
    error_type: str  # "morphology" | "agreement" | "anomaly"
    severity: str  # "error" | "warning"
    message: str
    suggestion: str | None = None
    confidence: float = 0.0


class TurkishGrammarChecker:
    """Türkçe metin için dilbilgisi denetleyicisi.

    Üç sinyal kullanır:
    1. kök-türk morfolojik analizi (ek geçerliliği)
    2. Morfolojik uyum kuralları (tamlayan-iyelik vb.)
    3. BERTurk anomali tespiti (v2'de eklenecek)

    Kullanım::

        checker = TurkishGrammarChecker()
        errors = checker.check("Benim ev çok güzel")
        for e in errors:
            print(f"{e.severity}: {e.message}")
    """

    def __init__(
        self,
        analyzer: MorphoAnalyzer | None = None,
        berturk_path: str = "models/berturk",
    ) -> None:
        if analyzer is None:
            from kokturk.core.analyzer import MorphoAnalyzer

            analyzer = MorphoAnalyzer(backends=["zeyrek"])
        self._analyzer = analyzer
        self._berturk_path = berturk_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, text: str) -> list[GrammarError]:
        """Metindeki dilbilgisi hatalarını denetle.

        Args:
            text: Ham Türkçe metin.

        Returns:
            Konuma göre sıralı hata listesi.
        """
        errors: list[GrammarError] = []
        words = text.split()
        if not words:
            return errors

        # Morfolojik analiz
        analyses = []
        for word in words:
            result = self._analyzer.analyze(word)
            analyses.append(result)

        # 1. Morfolojik geçerlilik
        errors.extend(self._check_morphology(words, analyses))

        # 2. Uyum kuralları
        errors.extend(self._check_agreement(words, analyses))

        # 3. Anomali tespiti (v2'de eklenecek)
        errors.extend(self._check_anomalies(text, words))

        return sorted(errors, key=lambda e: e.position)

    # ------------------------------------------------------------------
    # Morfolojik kontroller
    # ------------------------------------------------------------------

    def _check_morphology(self, words: list[str], analyses: list) -> list[GrammarError]:
        """Morfolojik olarak geçersiz kelimeleri kontrol et."""
        from kokturk.core.phonology import check_vowel_harmony

        errors: list[GrammarError] = []

        for i, (word, analysis) in enumerate(zip(words, analyses, strict=True)):
            # Noktalama atla
            if all(ch in ".,?!:;-–—…'\"()" for ch in word):
                continue

            # kök-türk sözlüğünde bulunamayan kelime
            if analysis is None or not analysis.analyses:
                errors.append(GrammarError(
                    position=i,
                    token=word,
                    error_type="morphology",
                    severity="warning",
                    message=(
                        f"'{word}' kök-türk sözlüğünde bulunamadı"
                        " — yazım hatası olabilir"
                    ),
                    confidence=0.7,
                ))
                continue

            # Ünlü uyumu kontrolü
            harmony = check_vowel_harmony(word)
            if not harmony.ok:
                errors.append(GrammarError(
                    position=i,
                    token=word,
                    error_type="morphology",
                    severity=harmony.severity,
                    message=(
                        f"'{word}' ünlü uyumu ihlali içerebilir"
                        if harmony.severity == "warning"
                        else f"'{word}' ünlü uyumu ihlali"
                    ),
                    confidence=0.5 if harmony.severity == "warning" else 0.7,
                ))

        return errors

    # ------------------------------------------------------------------
    # Uyum kontrolleri
    # ------------------------------------------------------------------

    def _check_agreement(self, words: list[str], analyses: list) -> list[GrammarError]:
        """Komşu sözcükler arasındaki morfolojik uyumu kontrol et."""
        errors: list[GrammarError] = []

        for i in range(len(words) - 1):
            # de/da - te/ta kuralı (bağımsız sözcük olarak)
            de_da_err = self._check_de_da(words, analyses, i + 1)
            if de_da_err is not None:
                errors.append(de_da_err)

            # Tamlayan-iyelik uyumu
            gen_err = self._check_genitive_possessive(words, analyses, i)
            if gen_err is not None:
                errors.append(gen_err)

        return errors

    def _check_de_da(
        self,
        words: list[str],
        analyses: list,
        idx: int,
    ) -> GrammarError | None:
        """de/da ve te/ta kullanımını kontrol et.

        1. Bağımsız "de"/"da" sözcüğü: önceki kelimenin son ünsüzüne bakarak
           sert ünsüz sonrası "te"/"ta" gerekip gerekmediğini kontrol et.
        2. Ayrı yazılmış ek: "ev de" yazılmışsa ama "evde" (bulunma eki)
           kastedilmişse uyarı ver.
        """
        if idx < 1 or idx >= len(words):
            return None

        word_lower = words[idx].lower()
        if word_lower not in ("de", "da"):
            return None

        prev_word = words[idx - 1]
        prev_lower = prev_word.lower().rstrip(".,?!:;")

        if not prev_lower:
            return None

        # Kural 1: Sert ünsüz sonrası te/ta
        last_char = prev_lower[-1]
        if last_char in _VOICELESS:
            correct = "te" if word_lower == "de" else "ta"
            return GrammarError(
                position=idx,
                token=words[idx],
                error_type="agreement",
                severity="error",
                message=(
                    f"Sert ünsüzden sonra '{correct}' kullanılmalı,"
                    f" '{words[idx]}' değil"
                ),
                suggestion=correct,
                confidence=0.95,
            )

        # Kural 2: Ayrı yazılmış bulunma eki tespiti
        # "ev de" → "evde" olabilir mi?
        merged = prev_lower + word_lower
        result = self._analyzer.analyze(merged)
        if result and result.analyses:
            # Birleşik yazım geçerli — olası ayrı yazım hatası
            has_loc = any("+LOC" in str(a) or "+Loc" in str(a) for a in result.analyses)
            if has_loc:
                return GrammarError(
                    position=idx,
                    token=words[idx],
                    error_type="agreement",
                    severity="warning",
                    message=(
                        f"'{prev_word} {words[idx]}' ayrı yazılmış"
                        f" — bulunma eki ise '{merged}' şeklinde"
                        " bitişik yazılmalı"
                    ),
                    suggestion=merged,
                    confidence=0.6,
                )

        return None

    def _check_genitive_possessive(
        self,
        words: list[str],
        analyses: list,
        idx: int,
    ) -> GrammarError | None:
        """Tamlayan eki sonrası iyelik eki gereksinimi kontrol et.

        "benim ev" → "benim evim" olmalı.
        """
        if idx + 1 >= len(words):
            return None

        word_lower = words[idx].lower()

        # Tamlayan zamiri mi?
        is_genitive = word_lower in _GENITIVE_PRONOUNS

        if not is_genitive:
            # Morfolojik analizde +GEN eki var mı?
            a1 = analyses[idx]
            if a1 and a1.analyses:
                is_genitive = any(
                    "+GEN" in str(a) or "+Gen" in str(a)
                    for a in a1.analyses
                )

        if not is_genitive:
            return None

        # Sonraki kelime isim mi ve iyelik eki var mı?
        a2 = analyses[idx + 1]
        if a2 is None or not a2.analyses:
            return None

        # İsim olup olmadığını kontrol et
        is_noun = any(
            "Noun" in str(a) or "+PLU" in str(a)
            for a in a2.analyses
        )
        if not is_noun:
            return None

        # İyelik eki var mı?
        has_poss = any(
            "+POSS" in str(a) or "+Poss" in str(a) or "+P1sg" in str(a)
            or "+P2sg" in str(a) or "+P3sg" in str(a) or "+P1pl" in str(a)
            or "+P2pl" in str(a) or "+P3pl" in str(a)
            for a in a2.analyses
        )

        if has_poss:
            return None

        return GrammarError(
            position=idx + 1,
            token=words[idx + 1],
            error_type="agreement",
            severity="error",
            message=(
                f"'{words[idx]}' tamlayan eki sonrası"
                f" '{words[idx + 1]}' iyelik eki gerektirir"
            ),
            confidence=0.8,
        )

    # ------------------------------------------------------------------
    # Anomali tespiti (v2)
    # ------------------------------------------------------------------

    def _check_anomalies(
        self, text: str, words: list[str]
    ) -> list[GrammarError]:
        """BERTurk tabanlı anomali tespiti.

        TODO(v2): AutoModelForMaskedLM yükleyerek düşük olasılıklı
        sözcükleri tespit et. Şu an devre dışı.
        """
        return []
