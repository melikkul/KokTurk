"""Fused Light Verb Construction (LVC) decomposition lexicon.

Turkish fuses many Noun+etmek/olmak compounds into single orthographic
tokens via consonant gemination (``ret`` → ``redd-etmek``) or vowel elision
(``kayıp`` → ``kayb-olmak``). The standard analyzer treats these as opaque
verbs, hiding the internal nominal root. This module reverses the fusion.

Two public entry points:

* :func:`decompose_fused_lvc` — given a surface form (e.g. ``"reddetti"``)
  return ``(nominal, light_verb, suffix_remainder)`` if a fused LVC root
  can be matched at the start of the surface, else ``None``.
* :func:`is_fused_lvc` — quick membership check for stems.

The lexicon is opt-in (the analyzer only consults it when
``decompose_lvc=True``) so existing pipelines remain bit-for-bit identical.
"""

from __future__ import annotations

# Surface verbal stem (without infinitive ``-mek``/``-mak``) →
# (nominal_component, light_verb_root). Light verb is always ``et`` or ``ol``.
#
# Each entry is verified by either consonant gemination (single→double, e.g.
# ``ret`` + ``et`` → ``redd-et``), vowel elision (medial-vowel drop, e.g.
# ``kayıp`` + ``ol`` → ``kayb-ol``), or a consonant voicing alternation.
# Plain ``noun + et/ol`` without any morphophonology is NOT a fused LVC and
# is excluded from the table on principle.
# Each entry maps a surface verbal stem prefix → (nominal, light_verb).
# Both the consonant-final form (``...et``) and the voiced/vowel-initial
# form (``...ed``) are listed when the light verb ``et`` triggers t→d
# voicing before a vowel-initial inflection (e.g. ``redd-et`` → ``redd-ed-iyor``).
FUSED_LVC_TABLE: dict[str, tuple[str, str]] = {
    # gemination forms (single consonant doubles before -et)
    "reddet": ("ret", "et"),
    "redded": ("ret", "et"),
    "hisset": ("his", "et"),
    "hissed": ("his", "et"),
    "zannet": ("zan", "et"),
    "zanned": ("zan", "et"),
    "affet": ("af", "et"),
    "affed": ("af", "et"),
    "hallet": ("hal", "et"),
    "halled": ("hal", "et"),
    "haccet": ("hac", "et"),
    "şakket": ("şak", "et"),
    # elision / syncope forms (medial vowel drops before -et / -ol)
    "sabret": ("sabır", "et"),
    "sabred": ("sabır", "et"),
    "emret": ("emir", "et"),
    "emred": ("emir", "et"),
    "azmet": ("azim", "et"),
    "azmed": ("azim", "et"),
    "devret": ("devir", "et"),
    "devred": ("devir", "et"),
    "hükmet": ("hüküm", "et"),
    "hükmed": ("hüküm", "et"),
    "naklet": ("nakil", "et"),
    "nakled": ("nakil", "et"),
    "seyret": ("seyir", "et"),
    "seyred": ("seyir", "et"),
    "fethet": ("fetih", "et"),
    "fethed": ("fetih", "et"),
    "kahret": ("kahır", "et"),
    "kahred": ("kahır", "et"),
    "akset": ("akis", "et"),
    "aksed": ("akis", "et"),
    "şükret": ("şükür", "et"),
    "şükred": ("şükür", "et"),
    "fikret": ("fikir", "et"),
    "fikred": ("fikir", "et"),
    "kaybol": ("kayıp", "ol"),
    # voicing alternation (final voiceless → voiced before vowel-initial)
    "bahset": ("bahis", "et"),
    "bahsed": ("bahis", "et"),
    "mahvet": ("mahiv", "et"),
    "mahved": ("mahiv", "et"),
    "neşret": ("neşir", "et"),
    "neşred": ("neşir", "et"),
    "vazet": ("vaaz", "et"),
    "vazed": ("vaaz", "et"),
}


def is_fused_lvc(root: str) -> bool:
    """Return True if ``root`` is a known fused light-verb stem.

    Args:
        root: a verbal stem candidate (no infinitive suffix).
    """
    return root in FUSED_LVC_TABLE


def decompose_fused_lvc(surface_form: str) -> tuple[str, str, str] | None:
    """Decompose a fused LVC surface token into its underlying components.

    The function does a longest-prefix match against
    :data:`FUSED_LVC_TABLE`. If a match is found, the matched prefix is
    interpreted as the surface form of ``nominal + light_verb``, and the
    rest of the surface string is treated as the suffix sequence attached
    to the light verb.

    Args:
        surface_form: a raw input token, e.g. ``"reddetti"`` or
            ``"hissediyorum"``.

    Returns:
        ``(nominal, light_verb, suffix_remainder)`` if the surface starts
        with a known fused LVC root, otherwise ``None``.

    Examples:
        >>> decompose_fused_lvc("reddetti")
        ('ret', 'et', 'ti')
        >>> decompose_fused_lvc("hissediyorum")
        ('his', 'et', 'iyorum')
        >>> decompose_fused_lvc("kayboldu")
        ('kayıp', 'ol', 'du')
        >>> decompose_fused_lvc("geliyor") is None
        True
    """
    if not surface_form:
        return None

    lower = surface_form.lower()
    # Longest-prefix match — sort keys by length descending so that longer
    # stems shadow their prefixes (e.g. "naklet" before "nakl").
    for stem in sorted(FUSED_LVC_TABLE, key=len, reverse=True):
        if lower.startswith(stem):
            nominal, light_verb = FUSED_LVC_TABLE[stem]
            remainder = surface_form[len(stem):]
            return nominal, light_verb, remainder
    return None
