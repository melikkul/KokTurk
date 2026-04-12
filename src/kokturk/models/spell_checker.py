"""Türkçe yazım denetleyicisi — kök-türk morfolojik sözlüğü tabanlı.

Üç temel yetenek:
1. ASCII düzeltme — "turkce" → "türkçe"
2. I/İ büyük-küçük harf düzeltme — "Istanbul" → "İstanbul"
3. Yazım önerisi — edit mesafesi ile en yakın geçerli kelime

Morfolojik doğrulama için kök-türk çözümleyicisi kullanılır.
Sinir ağı modeli gerektirmez — kural tabanlı, morfolojik farkındalıklı.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kokturk.core.analyzer import MorphoAnalyzer

# Turkish alphabet (29 letters)
TURKISH_ALPHABET = "abcçdefgğhıijklmnoöprsştuüvyz"

# Common ASCII → Turkish single-char substitutions (most frequent typos)
_TURKISH_CHAR_SUBS: dict[str, str] = {
    "c": "ç",
    "g": "ğ",
    "i": "ı",
    "o": "ö",
    "s": "ş",
    "u": "ü",
}

# Max morphological validation calls per word (performance cap)
_MAX_VALIDATE_CALLS = 50


@dataclass(frozen=True, slots=True)
class SpellSuggestion:
    """Bir yazım düzeltme önerisi."""

    position: int
    original: str
    suggestion: str
    error_type: str  # "deasciify" | "casing" | "apostrophe" | "spelling"
    confidence: float
    message: str


class TurkishSpellChecker:
    """Türkçe yazım denetleyicisi.

    kök-türk morfolojik çözümleyicisini sözlük olarak kullanarak
    yazım hatalarını tespit eder ve düzeltme önerisi sunar.

    Kullanım::

        checker = TurkishSpellChecker()
        results = checker.check("Turkiyede yasiyoruz Istanbul guzel")
        for s in results:
            print(f"{s.original} → {s.suggestion}  ({s.message})")
    """

    def __init__(self, analyzer: MorphoAnalyzer | None = None) -> None:
        if analyzer is None:
            from kokturk.core.analyzer import MorphoAnalyzer

            analyzer = MorphoAnalyzer(backends=["zeyrek"])
        self._analyzer = analyzer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, text: str) -> list[SpellSuggestion]:
        """Metindeki yazım hatalarını denetle.

        Args:
            text: Denetlenecek Türkçe metin.

        Returns:
            Bulunan hatalara ilişkin öneri listesi.
        """
        suggestions: list[SpellSuggestion] = []
        words = text.split()

        for i, word in enumerate(words):
            # 1. I/İ büyük-küçük harf kontrolü
            casing_fix = self._check_casing(word)
            if casing_fix:
                suggestions.append(SpellSuggestion(
                    position=i,
                    original=word,
                    suggestion=casing_fix,
                    error_type="casing",
                    confidence=0.95,
                    message=f"Türkçe büyük harf düzeltme: '{word}' → '{casing_fix}'",
                ))
                word = casing_fix  # sonraki kontrollerden önce uygula

            # 2. ASCII düzeltme — Zeyrek ASCII formları da tanıyabilir,
            #    bu yüzden geçerlilik kontrolünden ÖNCE deasciify dene
            if self._has_ascii_turkish_chars(word):
                deasc = self._deasciify(word)
                if deasc is not None and deasc != word:
                    suggestions.append(SpellSuggestion(
                        position=i,
                        original=word,
                        suggestion=deasc,
                        error_type="deasciify",
                        confidence=0.9,
                        message=f"ASCII düzeltme: '{word}' → '{deasc}'",
                    ))
                    continue

            # 3. Morfolojik doğrulama — kelime geçerliyse atla
            if self._is_valid_word(word):
                continue

            # 4. Apostrof ekleme dene (özel isim + ek)
            apost = self._try_apostrophe(word)
            if apost is not None:
                suggestions.append(SpellSuggestion(
                    position=i,
                    original=word,
                    suggestion=apost,
                    error_type="apostrophe",
                    confidence=0.85,
                    message=f"Apostrof eksik: '{word}' → '{apost}'",
                ))
                continue

            # 5. Edit mesafesi ile yakın kelime ara
            closest = self._find_closest(word)
            if closest is not None:
                suggestions.append(SpellSuggestion(
                    position=i,
                    original=word,
                    suggestion=closest,
                    error_type="spelling",
                    confidence=0.6,
                    message=f"Yazım önerisi: '{word}' → '{closest}'",
                ))

        return suggestions

    def correct(self, text: str) -> str:
        """Yüksek güvenilirlikli düzeltmeleri otomatik uygula.

        Yalnızca güven skoru >= 0.8 olan öneriler uygulanır.

        Args:
            text: Düzeltilecek Türkçe metin.

        Returns:
            Düzeltilmiş metin.
        """
        suggestions = self.check(text)
        words = text.split()

        for sugg in sorted(suggestions, key=lambda s: -s.position):
            if sugg.confidence >= 0.8 and 0 <= sugg.position < len(words):
                words[sugg.position] = sugg.suggestion

        return " ".join(words)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _has_ascii_turkish_chars(word: str) -> bool:
        """Kelimede Türkçe karşılığı olan ASCII karakterler var mı?

        Eğer kelime sadece ASCII kullanıyorsa ve c/g/i/o/s/u içeriyorsa
        deasciify denenmelidir.
        """
        turkish_specific = set("çğışöüÇĞİŞÖÜ")
        # Zaten Türkçe karakter varsa deasciify gerekmez
        if any(ch in turkish_specific for ch in word):
            return False
        # ASCII'de Türkçe karşılığı olan karakterler var mı?
        return any(ch in _TURKISH_CHAR_SUBS for ch in word.lower())

    def _is_valid_word(self, word: str) -> bool:
        """Kelimenin morfolojik olarak geçerli olup olmadığını kontrol et."""
        # Noktalama tek başına geçerlidir
        if all(ch in ".,?!:;-–—…'\"()" for ch in word):
            return True
        result = self._analyzer.analyze(word)
        return bool(result and result.analyses)

    def _check_casing(self, word: str) -> str | None:
        """Türkçe I/İ büyük-küçük harf hatalarını düzelt."""
        from ariturk.normalize import turkish_upper

        # "Istanbul" → "İstanbul" (kelime başı I + küçük harf)
        if word.startswith("I") and len(word) > 1 and word[1].islower():
            fix = "İ" + word[1:]
            return fix

        # "ISTANBUL" → "İSTANBUL"
        if word.isupper() and "I" in word:
            fix = turkish_upper(word.replace("I", "İ").lower())
            # Basit kontrol: sadece I→İ dönüşümü yap
            fix = word.replace("I", "İ")
            if fix != word:
                return fix

        return None

    def _deasciify(self, word: str) -> str | None:
        """ASCII Türkçe karakterlerini düzelt.

        Önce ariturk lookup tablosunu dene, sonra tek-karakter
        değişikliklerini sırayla dene (üstel değil, doğrusal).
        """
        from ariturk.normalize import restore_diacritics

        # 1. Lookup tablosu (O(1))
        restored = restore_diacritics(word)
        if restored != word and self._is_valid_word(restored):
            return restored

        # 2. Tek karakter değişiklikleri (doğrusal)
        lower = word.lower()
        calls = 0
        for i, ch in enumerate(lower):
            if ch in _TURKISH_CHAR_SUBS and calls < _MAX_VALIDATE_CALLS:
                candidate = word[:i] + _TURKISH_CHAR_SUBS[ch] + word[i + 1:]
                calls += 1
                if self._is_valid_word(candidate):
                    return candidate

        # 3. İki karakter değişikliği kombinasyonları (sınırlı)
        positions = [
            (i, _TURKISH_CHAR_SUBS[ch])
            for i, ch in enumerate(lower)
            if ch in _TURKISH_CHAR_SUBS
        ]
        for j in range(len(positions)):
            for k in range(j + 1, len(positions)):
                if calls >= _MAX_VALIDATE_CALLS:
                    return None
                pos_j, repl_j = positions[j]
                pos_k, repl_k = positions[k]
                chars = list(word)
                chars[pos_j] = repl_j
                chars[pos_k] = repl_k
                candidate = "".join(chars)
                calls += 1
                if self._is_valid_word(candidate):
                    return candidate

        return None

    def _try_apostrophe(self, word: str) -> str | None:
        """Özel isim + ek arasına apostrof eklemeyi dene."""
        positions = range(2, len(word) - 1)
        for calls, i in enumerate(positions):
            if calls >= _MAX_VALIDATE_CALLS:
                break
            with_apost = f"{word[:i]}'{word[i:]}"
            if self._is_valid_word(with_apost):
                return with_apost
        return None

    def _find_closest(self, word: str, max_distance: int = 1) -> str | None:
        """Edit mesafesi 1 ile en yakın geçerli kelimeyi bul.

        Önce Türkçe'ye özgü karakter ikame kalıplarını dener, sonra
        genel düzenleme işlemlerini (takas, silme, ekleme).
        Kelime başına en fazla ``_MAX_VALIDATE_CALLS`` morfolojik sorgu yapar.
        """
        if len(word) > 15:
            return None

        calls = 0
        lower = word.lower()

        # 1. Türkçe'ye özgü ikameler (ı↔i, ö↔o, ü↔u, ç↔c, ş↔s, ğ↔g)
        turkish_pairs = [
            ("ı", "i"), ("i", "ı"),
            ("ö", "o"), ("o", "ö"),
            ("ü", "u"), ("u", "ü"),
            ("ç", "c"), ("c", "ç"),
            ("ş", "s"), ("s", "ş"),
            ("ğ", "g"), ("g", "ğ"),
        ]
        for i, ch in enumerate(lower):
            for src, dst in turkish_pairs:
                if ch == src and calls < _MAX_VALIDATE_CALLS:
                    candidate = word[:i] + dst + word[i + 1:]
                    calls += 1
                    if self._is_valid_word(candidate):
                        return candidate

        # 2. Bitişik karakter takası
        for i in range(len(word) - 1):
            if calls >= _MAX_VALIDATE_CALLS:
                return None
            swapped = word[:i] + word[i + 1] + word[i] + word[i + 2:]
            calls += 1
            if self._is_valid_word(swapped):
                return swapped

        # 3. Bir karakter silme
        for i in range(len(word)):
            if calls >= _MAX_VALIDATE_CALLS:
                return None
            deleted = word[:i] + word[i + 1:]
            if len(deleted) >= 2:
                calls += 1
                if self._is_valid_word(deleted):
                    return deleted

        # 4. Bir karakter ekleme (Türkçe alfabe)
        for i in range(len(word) + 1):
            for ch in TURKISH_ALPHABET:
                if calls >= _MAX_VALIDATE_CALLS:
                    return None
                inserted = word[:i] + ch + word[i:]
                calls += 1
                if self._is_valid_word(inserted):
                    return inserted

        return None
