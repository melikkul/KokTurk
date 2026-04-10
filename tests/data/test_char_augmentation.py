"""Tests for character-level augmentation (Category C Task 3)."""
from __future__ import annotations

import random

from data.char_augmentation import (
    ASCII_TO_DIACRITIC,
    DIACRITIC_TO_ASCII,
    TURKISH_QWERTY_COORDS,
    CompositeAugmenter,
    DiacriticAugmenter,
    KeyboardAugmenter,
    StemCorruptAugmenter,
)


def test_keyboard_identity_when_prob_zero():
    aug = KeyboardAugmenter(noise_prob=0.0)
    assert aug.augment("merhaba") == "merhaba"


def test_keyboard_only_substitutes_adjacent_keys():
    random.seed(0)
    aug = KeyboardAugmenter(noise_prob=1.0)
    out = aug.augment("kalem")
    for orig, new in zip("kalem", out):
        if orig == new:
            continue
        assert new in aug.adjacency[orig], f"{new!r} not adjacent to {orig!r}"


def test_keyboard_handles_turkish_chars():
    random.seed(1)
    aug = KeyboardAugmenter(noise_prob=1.0)
    out = aug.augment("çocuk")
    # "ç" is a known key — output differs but still has length 5
    assert len(out) == 5


def test_diacritic_strip_mode():
    aug = DiacriticAugmenter(prob=1.0, mode="strip")
    out = aug.augment("çocuğa")
    for ch in out:
        assert ch not in DIACRITIC_TO_ASCII


def test_diacritic_identity_when_prob_zero():
    aug = DiacriticAugmenter(prob=0.0)
    assert aug.augment("ışık") == "ışık"


def test_diacritic_harmony_safe_swap():
    random.seed(2)
    aug = DiacriticAugmenter(prob=1.0, mode="swap", harmony_safe=True)
    # ı is back-vowel; it must not turn into a front-class letter like ü or ö
    from data.char_augmentation import BACK_VOWELS, FRONT_VOWELS
    for _ in range(20):
        out = aug.augment("ı")
        assert out in BACK_VOWELS, f"harmony violated: {out!r}"


def test_stemcorrupt_tags_unchanged():
    random.seed(3)
    aug = StemCorruptAugmenter(corrupt_prob=1.0, preserve_length=True)
    new_surface, root, tags = aug.augment(
        "evlerinden", root="ev", tags="ev +PLU +POSS.3SG +ABL",
    )
    assert tags == "ev +PLU +POSS.3SG +ABL"
    assert root == "ev"
    # Suffix portion ("lerinden") preserved
    assert new_surface.endswith("lerinden")
    # Stem portion ("ev") is length-preserved
    assert len(new_surface) == len("evlerinden")


def test_stemcorrupt_noop_when_prob_zero():
    aug = StemCorruptAugmenter(corrupt_prob=0.0)
    s, r, t = aug.augment("kitap", root="kitap", tags="kitap")
    assert s == "kitap"


def test_composite_chains_augmenters():
    random.seed(4)
    comp = CompositeAugmenter([
        (DiacriticAugmenter(prob=1.0, mode="strip"), 1.0),
        (KeyboardAugmenter(noise_prob=0.0), 1.0),  # no-op: noise_prob 0
    ])
    s, _, _ = comp.augment("çocuk")
    # Diacritic strip ran; keyboard no-op
    assert "ç" not in s


def test_composite_probability_zero_skips():
    random.seed(5)
    strict = DiacriticAugmenter(prob=1.0, mode="strip")
    comp = CompositeAugmenter([(strict, 0.0)])
    s, _, _ = comp.augment("çiçek")
    assert s == "çiçek"


def test_empty_and_single_char():
    ka = KeyboardAugmenter(noise_prob=1.0)
    da = DiacriticAugmenter(prob=1.0)
    sc = StemCorruptAugmenter(corrupt_prob=1.0)
    assert ka.augment("") == ""
    assert da.augment("") == ""
    new, _, _ = sc.augment("")
    assert new == ""
    # Single char doesn't crash
    ka.augment("a")
    da.augment("ç")
    sc.augment("a", root="a", tags="a")


def test_turkish_qwerty_coords_shape():
    # Sanity: table has the Turkish letters we expect
    for ch in "çğıöşü":
        assert ch in TURKISH_QWERTY_COORDS
    assert set(ASCII_TO_DIACRITIC.values()) <= set(DIACRITIC_TO_ASCII.keys())
