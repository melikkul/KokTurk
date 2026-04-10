"""Performance regression tests.

Ensure analyzer inference doesn't silently slow down.
Uses generous time bounds to catch gross regressions only.
"""

from __future__ import annotations

import time

import pytest

from kokturk.core.analyzer import MorphoAnalyzer

TEST_WORDS = [
    "evlerinden", "gidiyorum", "yapılmış", "kitabı", "çocuklar",
    "gelecek", "okudum", "güzel", "yüz", "yazar",
    "alan", "koşuyor", "gelmeli", "aldım", "yazıyor",
    "okuyacak", "büyük", "küçük", "gözlük", "başladı",
    "okullarında", "gelmeyecek", "gördüm", "gelmiş", "yazmıştım",
    "okunur", "düşünüyorum", "anlamadım", "seviyorum", "bakıyoruz",
    "yapacağız", "gidecekler", "gelmediler", "yazılacak", "öğretmen",
    "öğretmenler", "kitapları", "arkadaşım", "insanlar", "güzellik",
    "anlatıyor", "söylüyor", "bilgi", "gözleri", "ağaçlar",
    "çalışıyor", "okul", "ev", "su", "gel",
    "iyi", "büyük", "küçük", "güzel", "kötü",
    "gelmek", "yazmak", "okumak", "görmek", "almak",
    "vermek", "demek", "bilmek", "istemek", "başlamak",
    "anne", "baba", "kardeş", "dost", "yol",
    "gün", "zaman", "yer", "şey", "para",
    "adam", "kadın", "çocuk", "kız", "oğul",
    "masa", "kapı", "pencere", "duvar", "bahçe",
    "araba", "yemek", "içmek", "uyumak", "gülmek",
    "ağlamak", "koşmak", "yürümek", "oturmak", "kalkmak",
    "düşmek", "çıkmak", "inmek", "binmek", "dönmek",
]


@pytest.mark.slow
def test_inference_throughput() -> None:
    """Analyzer must process >=100 tokens in <5 seconds on CPU."""
    analyzer = MorphoAnalyzer(backends=["zeyrek"])
    tokens = TEST_WORDS[:100]

    start = time.perf_counter()
    for token in tokens:
        analyzer.analyze(token)
    elapsed = time.perf_counter() - start

    assert elapsed < 5.0, (
        f"Throughput regression: 100 tokens took {elapsed:.2f}s (limit: 5s)"
    )


@pytest.mark.slow
def test_analyzer_startup_time() -> None:
    """MorphoAnalyzer initialization must complete in <10 seconds."""
    start = time.perf_counter()
    MorphoAnalyzer(backends=["zeyrek"])
    elapsed = time.perf_counter() - start

    assert elapsed < 10.0, (
        f"Startup regression: analyzer init took {elapsed:.2f}s (limit: 10s)"
    )
