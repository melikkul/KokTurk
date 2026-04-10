"""Code-switch preprocessing for Turkish text with foreign roots.

Handles the common pattern: ForeignRoot'TurkishSuffixes
Examples: Google'ladım, iPhone'un, Biden'a, Netflix'ten, tweet'ledim

Strategy:

1. Detect apostrophe boundary in token.
2. Split into (foreign_root, suffix_part).
3. Classify the foreign root as verb_stem / proper_noun / noun.
4. Analyze suffix_part with RELAXED vowel harmony
   (foreign root phonology is unpredictable).
5. Return analysis: foreign_root +FOREIGN +suffix_tags.

This mirrors Zemberek's "runtime dictionary generator" pattern.
The module is opt-in via :py:meth:`MorphoAnalyzer.analyze`'s
``handle_code_switch=True`` flag — default behavior is unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

TURKISH_SPECIFIC_CHARS = frozenset("çğışöüÇĞİŞÖÜ")

APOSTROPHE_VARIANTS: tuple[str, ...] = (
    "'",       # ASCII apostrophe
    "\u2019",  # RIGHT SINGLE QUOTATION MARK (most common in Turkish text)
    "\u02BC",  # MODIFIER LETTER APOSTROPHE
    "\u2018",  # LEFT SINGLE QUOTATION MARK
)


@dataclass(frozen=True, slots=True)
class CodeSwitchResult:
    """Result of code-switch detection and splitting.

    Attributes:
        is_code_switched: Always ``True`` when this object exists.
        foreign_root: The foreign stem before the apostrophe (e.g. "Google").
        suffix_part: Turkish suffix chain after the apostrophe (e.g. "ladım").
        original: The original token (e.g. "Google'ladım").
        root_type: ``"verb_stem"`` | ``"proper_noun"`` | ``"noun"``.
    """

    is_code_switched: bool
    foreign_root: str
    suffix_part: str
    original: str
    root_type: str


# ---------------------------------------------------------------------------
# Suffix analysis patterns
# ---------------------------------------------------------------------------

# Ordered list of (regex, canonical_tag) for left-to-right suffix matching.
# Patterns are tried at the *current position* in the suffix string.
# Longer / more-specific patterns come first to prevent short-match greediness.
FOREIGN_SUFFIX_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # -- Verbalizer -lA: must be followed by tense/person, NOT by r+end (plural) --
    (re.compile(r"l[ae](?=[dym])"), "+VERB.LA"),
    (re.compile(r"l[ae](?=r[ıiuü])"), "+VERB.LA"),

    # -- Tense / aspect (after verbalizer) --
    (re.compile(r"d[ıiuü]"), "+PAST"),
    (re.compile(r"[yı]or"), "+PROG"),
    (re.compile(r"m[ıiuü]ş"), "+EVID"),
    (re.compile(r"[yaeıiouü]?[ae]c[ae]k"), "+FUT"),
    (re.compile(r"r(?=[ıiuü]|$)"), "+AOR"),

    # -- Nominal: plural (before agreement to prevent ler/lar → +3PL) --
    (re.compile(r"l[ae]r"), "+PLU"),

    # -- Agreement markers --
    (re.compile(r"n[ıiuü]z"), "+2PL"),
    (re.compile(r"m$"), "+1SG"),
    (re.compile(r"n$"), "+2SG"),
    (re.compile(r"k$"), "+1PL"),

    # -- Nominal: case markers (longer before shorter) --
    (re.compile(r"d[ae]n"), "+ABL"),
    (re.compile(r"t[ae]n"), "+ABL"),
    (re.compile(r"d[ae]"), "+LOC"),
    (re.compile(r"t[ae]"), "+LOC"),
    (re.compile(r"n[ıiuü]n"), "+GEN"),
    (re.compile(r"[ıiuü]n"), "+GEN"),
    (re.compile(r"y[aeıiuü]"), "+DAT"),
    (re.compile(r"[ae]"), "+DAT"),
    (re.compile(r"n[ıiuü]"), "+POSS.3SG"),
    (re.compile(r"y[ıiuü]"), "+ACC"),
    (re.compile(r"[ıiuü]"), "+ACC"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def split_foreign_suffix(token: str) -> tuple[str, str] | None:
    """Split a token on the first apostrophe variant.

    Handles ``'``, ``\\u2019``, ``\\u02BC``, ``\\u2018``.

    Returns:
        ``(root, suffix)`` if an apostrophe is found and both parts
        are non-empty, else ``None``.
    """
    if not token:
        return None
    for apo in APOSTROPHE_VARIANTS:
        idx = token.find(apo)
        if idx > 0 and idx < len(token) - 1:
            return (token[:idx], token[idx + 1 :])
    return None


def classify_foreign_root(root: str, suffix: str) -> str:
    """Classify the foreign root type based on the suffix pattern.

    Returns:
        ``"verb_stem"`` if the suffix starts with a Turkish verbalizer
        (``-la/-le``), ``"proper_noun"`` if the root starts uppercase
        and contains no Turkish-specific characters, else ``"noun"``.
    """
    # Verbalizer check: suffix begins with la/le followed by tense/person marker.
    # Bare "lar"/"ler" is plural, not verbalizer — require len > 3.
    if (
        suffix
        and len(suffix) > 3
        and suffix[0] == "l"
        and suffix[1] in "ae"
        and suffix[2] in "dymrıiuü"
    ):
        return "verb_stem"
    # Proper noun: starts uppercase, no Turkish-specific characters
    if root and root[0].isupper() and not any(c in TURKISH_SPECIFIC_CHARS for c in root):
        return "proper_noun"
    return "noun"


def detect_code_switch(token: str) -> CodeSwitchResult | None:
    """Detect if a token contains a foreign root with Turkish suffixes.

    Heuristics:

    1. Token must contain an apostrophe variant with non-empty parts on
       both sides.
    2. Root that is all-uppercase (len >= 2) is treated as an abbreviation,
       NOT a code-switch — defers to ``handle_special_tokens``.
    3. Root containing Turkish-specific characters (çğışöü) is treated as
       a native Turkish proper noun — defers to standard backends.

    Returns:
        :class:`CodeSwitchResult` or ``None`` if the token is not
        code-switched.
    """
    split = split_foreign_suffix(token)
    if split is None:
        return None

    root, suffix = split

    # Abbreviation disambiguation: all-uppercase root → defer to special_tokens
    root_alpha = root.replace(".", "")
    if len(root_alpha) >= 2 and root_alpha.isupper() and root_alpha.isalpha():
        return None

    # Turkish proper noun disambiguation: root contains çğışöü → not foreign
    if any(c in TURKISH_SPECIFIC_CHARS for c in root):
        return None

    root_type = classify_foreign_root(root, suffix)
    return CodeSwitchResult(
        is_code_switched=True,
        foreign_root=root,
        suffix_part=suffix,
        original=token,
        root_type=root_type,
    )


def analyze_foreign_suffixes(
    suffix_part: str,
    root_final_vowel: str | None = None,
    *,
    relaxed_harmony: bool = True,
) -> list[str]:
    """Analyze the Turkish suffix chain attached to a foreign root.

    Uses left-to-right ordered regex matching against
    :data:`FOREIGN_SUFFIX_PATTERNS`. When ``relaxed_harmony=True``
    (default for code-switched tokens), front/back vowel variants
    are treated as equivalent (already encoded in the regex classes).

    Args:
        suffix_part: The suffix string after the apostrophe.
        root_final_vowel: Last vowel of the foreign root (informational).
        relaxed_harmony: Accept both front and back vowel variants.

    Returns:
        List of canonical tags, e.g. ``["+VERB.LA", "+PAST", "+1SG"]``.
    """
    if not suffix_part:
        return []

    tags: list[str] = []
    pos = 0

    while pos < len(suffix_part):
        matched = False
        for pattern, tag in FOREIGN_SUFFIX_PATTERNS:
            m = pattern.match(suffix_part, pos)
            if m:
                tags.append(tag)
                pos = m.end()
                matched = True
                break
        if not matched:
            # Skip unrecognized character and continue
            pos += 1

    return tags
