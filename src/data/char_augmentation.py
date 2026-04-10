"""Character-level augmentation for Turkish morphological training.

Three strategies are provided:

1. :class:`KeyboardAugmenter` — substitutes characters with neighbours on
   a Turkish QWERTY layout (Euclidean distance ≤ √2).
2. :class:`DiacriticAugmenter` — strips or swaps diacritics (ç/ğ/ı/ö/ş/ü).
   In ``harmony_safe`` mode the swap stays inside the same vowel harmony
   class (front↔front, back↔back) so we never create harmony violations.
3. :class:`StemCorruptAugmenter` — replaces the stem characters of the
   surface form with random characters while leaving the gold tags
   untouched. Forces the decoder to decompose affixes independently of
   root identity.

:class:`CompositeAugmenter` chains the above with per-augmenter probabilities.
"""

from __future__ import annotations

import random
from typing import Sequence

__all__ = [
    "TURKISH_QWERTY_COORDS",
    "DIACRITIC_TO_ASCII",
    "ASCII_TO_DIACRITIC",
    "FRONT_VOWELS",
    "BACK_VOWELS",
    "KeyboardAugmenter",
    "DiacriticAugmenter",
    "StemCorruptAugmenter",
    "CompositeAugmenter",
]


TURKISH_QWERTY_COORDS: dict[str, tuple[int, int]] = {
    "q": (0, 0), "w": (0, 1), "e": (0, 2), "r": (0, 3), "t": (0, 4),
    "y": (0, 5), "u": (0, 6), "ı": (0, 7), "o": (0, 8), "p": (0, 9),
    "ğ": (0, 10), "ü": (0, 11),
    "a": (1, 0), "s": (1, 1), "d": (1, 2), "f": (1, 3), "g": (1, 4),
    "h": (1, 5), "j": (1, 6), "k": (1, 7), "l": (1, 8), "ş": (1, 9),
    "i": (1, 10),
    "z": (2, 0), "x": (2, 1), "c": (2, 2), "v": (2, 3), "b": (2, 4),
    "n": (2, 5), "m": (2, 6), "ö": (2, 7), "ç": (2, 8),
}

DIACRITIC_TO_ASCII: dict[str, str] = {
    "ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u",
}
ASCII_TO_DIACRITIC: dict[str, str] = {
    "c": "ç", "g": "ğ", "i": "ı", "o": "ö", "s": "ş", "u": "ü",
}

# Turkish vowel harmony classes.
FRONT_VOWELS = set("eiöüİEÖÜ")
BACK_VOWELS = set("aıouAIOU")


def _harmony_class(ch: str) -> str | None:
    if ch in FRONT_VOWELS:
        return "front"
    if ch in BACK_VOWELS:
        return "back"
    return None


class KeyboardAugmenter:
    """Substitute chars with physically adjacent keys on Turkish QWERTY.

    Args:
        noise_prob: per-character substitution probability.
    """

    def __init__(self, noise_prob: float = 0.1) -> None:
        if not 0.0 <= noise_prob <= 1.0:
            raise ValueError(f"noise_prob out of range: {noise_prob}")
        self.noise_prob = noise_prob
        self.adjacency = self._build_adjacency()

    def _build_adjacency(self) -> dict[str, list[str]]:
        adj: dict[str, list[str]] = {}
        for ch, (r, c) in TURKISH_QWERTY_COORDS.items():
            neighbours = []
            for other, (r2, c2) in TURKISH_QWERTY_COORDS.items():
                if other == ch:
                    continue
                if abs(r - r2) <= 1 and abs(c - c2) <= 1:
                    neighbours.append(other)
            adj[ch] = neighbours
        return adj

    def augment(self, word: str, seed: int | None = None) -> str:
        if not word or self.noise_prob <= 0:
            return word
        rng = random.Random(seed) if seed is not None else random
        out_chars: list[str] = []
        for ch in word:
            lower = ch.lower()
            if lower in self.adjacency and rng.random() < self.noise_prob:
                neighbours = self.adjacency[lower]
                if neighbours:
                    sub = rng.choice(neighbours)
                    out_chars.append(sub if ch == lower else sub.upper())
                    continue
            out_chars.append(ch)
        return "".join(out_chars)


class DiacriticAugmenter:
    """Strip or swap Turkish diacritics.

    Args:
        prob: per-character application probability.
        mode: ``"strip"`` (ç→c), ``"swap"`` (swap within harmony class), or
            ``"both"`` (50/50).
        harmony_safe: when True, swaps stay inside the same vowel harmony
            class. When False all swaps are allowed (hypercorrection).
    """

    def __init__(
        self,
        prob: float = 0.15,
        mode: str = "strip",
        harmony_safe: bool = True,
    ) -> None:
        if mode not in {"strip", "swap", "both"}:
            raise ValueError(f"mode must be strip/swap/both, got {mode!r}")
        self.prob = prob
        self.mode = mode
        self.harmony_safe = harmony_safe

    def augment(self, word: str, seed: int | None = None) -> str:
        if not word or self.prob <= 0:
            return word
        rng = random.Random(seed) if seed is not None else random
        out: list[str] = []
        for ch in word:
            if rng.random() >= self.prob:
                out.append(ch)
                continue
            lower = ch.lower()
            op = self.mode
            if op == "both":
                op = "strip" if rng.random() < 0.5 else "swap"
            if op == "strip" and lower in DIACRITIC_TO_ASCII:
                sub = DIACRITIC_TO_ASCII[lower]
                out.append(sub if ch == lower else sub.upper())
            elif op == "swap":
                target_class = _harmony_class(lower) if self.harmony_safe else None
                candidates = [
                    c for c in DIACRITIC_TO_ASCII
                    if c != lower and (
                        target_class is None
                        or _harmony_class(c) == target_class
                    )
                ]
                if candidates:
                    sub = rng.choice(candidates)
                    out.append(sub if ch == lower else sub.upper())
                else:
                    out.append(ch)
            else:
                out.append(ch)
        return "".join(out)


class StemCorruptAugmenter:
    """Replace stem characters with random characters; tags unchanged.

    Args:
        corrupt_prob: probability of corrupting a given sample.
        preserve_length: when True, replacement has the same length as the
            original stem.
        alphabet: character pool to sample from. Defaults to Turkish lower.
    """

    DEFAULT_ALPHABET = "abcçdefgğhıijklmnoöprsştuüvyz"

    def __init__(
        self,
        corrupt_prob: float = 0.3,
        preserve_length: bool = True,
        alphabet: str | None = None,
    ) -> None:
        self.corrupt_prob = corrupt_prob
        self.preserve_length = preserve_length
        self.alphabet = alphabet or self.DEFAULT_ALPHABET

    def augment(
        self,
        surface_form: str,
        root: str = "",
        tags: str = "",
        seed: int | None = None,
    ) -> tuple[str, str, str]:
        if not surface_form or self.corrupt_prob <= 0:
            return surface_form, root, tags
        rng = random.Random(seed) if seed is not None else random
        if rng.random() >= self.corrupt_prob:
            return surface_form, root, tags

        # Identify stem portion: longest common prefix with `root` when given,
        # otherwise corrupt the whole surface form.
        if root and surface_form.lower().startswith(root.lower()):
            stem_len = len(root)
        else:
            stem_len = len(surface_form)

        n = stem_len if self.preserve_length else rng.randint(1, stem_len)
        corrupted_stem = "".join(rng.choice(self.alphabet) for _ in range(n))
        new_surface = corrupted_stem + surface_form[stem_len:]
        return new_surface, root, tags


class CompositeAugmenter:
    """Chain of ``(augmenter, probability)`` pairs.

    Each augmenter's ``augment`` is called with the current surface form.
    ``StemCorruptAugmenter`` is the only one that also returns a ``root``
    and ``tags`` tuple — it is detected by signature.
    """

    def __init__(self, augmenters: Sequence[tuple[object, float]]) -> None:
        self.augmenters = list(augmenters)

    def augment(
        self,
        surface_form: str,
        root: str = "",
        tags: str = "",
        seed: int | None = None,
    ) -> tuple[str, str, str]:
        rng = random.Random(seed) if seed is not None else random
        surf, r, t = surface_form, root, tags
        for aug, prob in self.augmenters:
            if rng.random() >= prob:
                continue
            if isinstance(aug, StemCorruptAugmenter):
                surf, r, t = aug.augment(surf, r, t)
            else:
                surf = aug.augment(surf)  # type: ignore[attr-defined]
        return surf, r, t
