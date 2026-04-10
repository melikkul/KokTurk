"""Preprocessing for abbreviations, numerics, and emphatic reduplication.

Three classes of Turkish tokens that the standard char-level analyzer
mis-handles:

* **Abbreviations** like ``NATO'nun`` need vowel harmony from the
  *pronunciation* of the abbreviation, not the orthography.
* **Numerics** like ``1990'larda`` need a digit-to-spoken conversion
  before harmony can be determined.
* **Emphatic reduplications** like ``masmavi`` are derived from a base
  adjective by prefixing the first CV(C) with a linker consonant
  (``p / s / m / r``); the linker is unpredictable per base.

The :func:`preprocess_special_token` dispatcher returns a
:class:`SpecialTokenResult` if any pattern matches, else ``None``. This
module is opt-in via :py:meth:`MorphoAnalyzer.analyze`'s
``handle_special_tokens=True`` flag — default behavior is unchanged.
"""

from __future__ import annotations

from kokturk.core.phonology import ALL_VOWELS, last_vowel
from kokturk.core.special_token_types import SpecialTokenResult

# ---------------------------------------------------------------------------
# Abbreviation handling
# ---------------------------------------------------------------------------

# Turkish letter pronunciations for spelled-out initialisms.
LETTER_PRONUNCIATIONS: dict[str, str] = {
    "A": "a", "B": "be", "C": "ce", "D": "de", "E": "e",
    "F": "fe", "G": "ge", "H": "he", "I": "ı", "İ": "i",
    "J": "je", "K": "ke", "L": "le", "M": "me", "N": "ne",
    "O": "o", "Ö": "ö", "P": "pe", "Q": "ku", "R": "re",
    "S": "se", "Ş": "şe", "T": "te", "U": "u", "Ü": "ü",
    "V": "ve", "W": "çift ve", "X": "iks", "Y": "ye", "Z": "ze",
}

# Common Turkish acronyms read as words rather than spelled out.
ACRONYM_PRONUNCIATIONS: dict[str, str] = {
    "NATO": "nato",
    "NASA": "nasa",
    "ASELSAN": "aselsan",
    "UNESCO": "unesko",
    "ODTÜ": "ödtü",
    "YÖK": "yök",
    "İTÜ": "itü",
    "İSO": "iso",
    "TOFAŞ": "tofaş",
    "TÜBİTAK": "tübitak",
    "ÇEVKO": "çevko",
}


def get_abbreviation_final_vowel(abbrev: str) -> str | None:
    """Return the harmony-driving vowel for an abbreviation, or None."""
    if not abbrev:
        return None
    upper = abbrev.upper()
    pron = ACRONYM_PRONUNCIATIONS.get(upper)
    if pron is not None:
        return last_vowel(pron)

    if not upper.isalpha():
        return None

    last_letter = upper[-1]
    letter_pron = LETTER_PRONUNCIATIONS.get(last_letter)
    if letter_pron is None:
        return None
    return last_vowel(letter_pron)


def split_abbreviation_suffix(token: str) -> tuple[str, str] | None:
    """Split ``"NATO'nun"`` → ``("NATO", "nun")``.

    Handles:

    * apostrophe-separated suffixes,
    * period-ending abbreviations (``Alm.lar`` → ``("Alm.", "lar")``),
    * the bare-uppercase no-apostrophe form (``NATOnun`` → ``("NATO", "nun")``)
      provided the uppercase head is at least two characters.
    """
    if not token:
        return None

    if "'" in token:
        head, _, tail = token.partition("'")
        if head and tail:
            return head, tail

    if "." in token:
        idx = token.index(".")
        head, tail = token[: idx + 1], token[idx + 1:]
        if head[:-1] and head[0].isupper() and tail and tail.islower() and tail.isalpha():
            return head, tail

    # No apostrophe / no period — try to peel off a leading uppercase run
    # followed by a lowercase suffix tail.
    head_chars: list[str] = []
    for ch in token:
        if ch.isupper() or ch.isdigit():
            head_chars.append(ch)
        else:
            break
    if len(head_chars) >= 2 and len(head_chars) < len(token):
        head = "".join(head_chars)
        tail = token[len(head):]
        if tail.isalpha() and tail.islower():
            return head, tail
    return None


# ---------------------------------------------------------------------------
# Numeric handling
# ---------------------------------------------------------------------------

DIGIT_PRONUNCIATIONS: dict[str, str] = {
    "0": "sıfır", "1": "bir", "2": "iki", "3": "üç", "4": "dört",
    "5": "beş", "6": "altı", "7": "yedi", "8": "sekiz", "9": "dokuz",
}

TENS_PRONUNCIATIONS: dict[str, str] = {
    "1": "on", "2": "yirmi", "3": "otuz", "4": "kırk", "5": "elli",
    "6": "altmış", "7": "yetmiş", "8": "seksen", "9": "doksan",
}


def get_numeric_final_vowel(number_str: str) -> str | None:
    """Return the harmony-driving vowel for a numeric token.

    Strategy: read the number, find the **last non-zero positional
    component** (units, tens, hundreds, …), and look up its spoken form.
    The last vowel of that spoken form is the harmony vowel.

    Examples:
        >>> get_numeric_final_vowel("1990")  # ends in "doksan"
        'a'
        >>> get_numeric_final_vowel("3")     # "üç"
        'ü'
        >>> get_numeric_final_vowel("100")   # "yüz"
        'ü'
    """
    if not number_str.isdigit():
        return None

    # Strip leading zeros so "007" behaves like "7".
    stripped = number_str.lstrip("0") or "0"
    units = stripped[-1]
    if units != "0":
        return last_vowel(DIGIT_PRONUNCIATIONS[units])

    if len(stripped) >= 2 and stripped[-2] != "0":
        return last_vowel(TENS_PRONUNCIATIONS[stripped[-2]])

    if len(stripped) >= 3 and stripped[-3] != "0":
        return last_vowel("yüz")  # 'ü'

    if len(stripped) >= 4 and stripped[-4] != "0":
        return last_vowel("bin")  # 'i'

    # Fallback: treat as "sıfır".
    return "ı"


def split_numeric_suffix(token: str) -> tuple[str, str] | None:
    """Split ``"1990'larda"`` → ``("1990", "larda")`` or ``"3'ün"`` → ``("3", "ün")``."""
    if not token or not token[0].isdigit():
        return None
    if "'" not in token:
        return None
    head, _, tail = token.partition("'")
    if head.isdigit() and tail and tail.isalpha():
        return head, tail
    return None


# ---------------------------------------------------------------------------
# Reduplication handling
# ---------------------------------------------------------------------------

REDUPLICATION_LEXICON: dict[str, str] = {
    "masmavi": "mavi",
    "bembeyaz": "beyaz",
    "kapkara": "kara",
    "sapasağlam": "sağlam",
    "yapayalnız": "yalnız",
    "güpegündüz": "gündüz",
    "tertemiz": "temiz",
    "kupkuru": "kuru",
    "sımsıcak": "sıcak",
    "büsbütün": "bütün",
    "düpedüz": "düz",
    "çırılçıplak": "çıplak",
    "çepeçevre": "çevre",
    "apaçık": "açık",
    "upuzun": "uzun",
    "yemyeşil": "yeşil",
    "kıpkırmızı": "kırmızı",
    "kıpkızıl": "kızıl",
    "mosmor": "mor",
    "sapsarı": "sarı",
    "apayrı": "ayrı",
    "dimdik": "dik",
    "dümdüz": "düz",
    "yepyeni": "yeni",
    "tıpatıp": "tıp",
    "darmadağın": "dağın",
    "paramparça": "parça",
    "kıskıvrak": "kıvrak",
    "boşboğaz": "boğaz",
}

REDUPLICATION_LINKERS: frozenset[str] = frozenset({"p", "s", "m", "r"})


def decompose_reduplication(token: str) -> tuple[str, str] | None:
    """Return ``(base_adjective, linker_consonant)`` if ``token`` is reduplicated.

    First consults :data:`REDUPLICATION_LEXICON`. If the token is not in
    the lexicon, falls back to a simple generative rule: the first
    syllable of an emphatic reduplication is ``base[0] + (optional vowel)
    + linker``, where ``linker`` is one of ``p / s / m / r``. The
    function attempts to peel that prefix off and verify the remainder
    matches the lexicon-base candidate space.
    """
    if not token:
        return None

    if token in REDUPLICATION_LEXICON:
        base = REDUPLICATION_LEXICON[token]
        # The linker is the consonant immediately before the base in the
        # reduplicated form.
        idx = token.find(base)
        if idx >= 1:
            linker = token[idx - 1]
            if linker in REDUPLICATION_LINKERS:
                return base, linker
        return base, ""

    # Generative attempt: assume the form is C V? L base, where L is a
    # linker. Try every (length-of-prefix, linker-position) split.
    for prefix_len in (2, 3):
        if len(token) <= prefix_len:
            continue
        prefix = token[:prefix_len]
        rest = token[prefix_len:]
        linker = prefix[-1]
        if linker not in REDUPLICATION_LINKERS:
            continue
        # The first char of the reduplicant must equal the first char of
        # the candidate base.
        if not rest or rest[0] != prefix[0]:
            continue
        return rest, linker

    return None


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------


def preprocess_special_token(token: str) -> SpecialTokenResult | None:
    """Identify and decompose a special token, or return ``None``.

    Priority order:

    1. **Numeric with suffix** — must start with a digit.
    2. **Abbreviation with suffix** — uppercase head + lowercase tail.
    3. **Reduplication** — lexicon lookup, then generative rule.
    4. **None** — pass through to the regular morphological analyzer.
    """
    if not token:
        return None

    # 1. numeric
    if token[0].isdigit():
        split = split_numeric_suffix(token)
        if split is not None:
            base, suffix = split
            vowel = get_numeric_final_vowel(base) or ""
            return SpecialTokenResult(
                token_type="numeric",
                base=base,
                suffix_part=suffix,
                harmony_vowel=vowel,
            )
        if token.isdigit():
            vowel = get_numeric_final_vowel(token) or ""
            return SpecialTokenResult(
                token_type="numeric",
                base=token,
                suffix_part="",
                harmony_vowel=vowel,
            )
        return None

    # 2. abbreviation — head must be at least 2 uppercase chars.
    if token[0].isupper():
        split = split_abbreviation_suffix(token)
        if split is not None:
            base, suffix = split
            head_alpha = base.rstrip(".")
            if len(head_alpha) >= 2 and head_alpha.isupper():
                vowel = get_abbreviation_final_vowel(head_alpha) or ""
                return SpecialTokenResult(
                    token_type="abbreviation",
                    base=base,
                    suffix_part=suffix,
                    harmony_vowel=vowel,
                )
        # Bare uppercase acronym with no suffix.
        if token.isupper() and len(token) >= 2 and token.isalpha():
            vowel = get_abbreviation_final_vowel(token) or ""
            return SpecialTokenResult(
                token_type="abbreviation",
                base=token,
                suffix_part="",
                harmony_vowel=vowel,
            )

    # 3. reduplication
    if token.islower():
        redup = decompose_reduplication(token)
        if redup is not None:
            base, _linker = redup
            vowel = last_vowel(base) or ""
            # Guard against generative false positives: require the base
            # to end in a vowel typical of Turkish adjectives, or to be
            # in the lexicon.
            if token in REDUPLICATION_LEXICON or (
                len(base) >= 3 and any(c in ALL_VOWELS for c in base)
            ):
                return SpecialTokenResult(
                    token_type="reduplication",
                    base=base,
                    suffix_part="",
                    harmony_vowel=vowel,
                )

    return None
