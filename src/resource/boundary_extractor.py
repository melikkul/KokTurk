"""Extract morpheme boundaries from canonical tag representations.

Input:  surface="evlerinden", canonical="ev +PLU +POSS.3SG +ABL"
Output: "ev|ler|i|nden"

Strategy:
1. Root is always a prefix of the surface (after allomorphy)
2. Each canonical tag maps to known surface allomorphs
3. Greedily match allomorphs left-to-right after the root
"""
from __future__ import annotations


# Suffix allomorphs for each canonical tag (longest first for greedy match)
TAG_SURFACE_MAP: dict[str, list[str]] = {
    "+PLU": ["lar", "ler"],
    "+ACC": ["yı", "yi", "yu", "yü", "ı", "i", "u", "ü"],
    "+DAT": ["ya", "ye", "na", "ne", "a", "e"],
    "+LOC": ["nda", "nde", "da", "de", "ta", "te"],
    "+ABL": ["ndan", "nden", "dan", "den", "tan", "ten"],
    "+GEN": ["nın", "nin", "nun", "nün", "ın", "in", "un", "ün"],
    "+INS": ["yla", "yle", "la", "le"],
    "+POSS.1SG": ["ım", "im", "um", "üm", "m"],
    "+POSS.2SG": ["ın", "in", "un", "ün", "n"],
    "+POSS.3SG": ["sı", "si", "su", "sü", "ı", "i", "u", "ü"],
    "+POSS.1PL": ["ımız", "imiz", "umuz", "ümüz", "mız", "miz", "muz", "müz"],
    "+POSS.2PL": ["ınız", "iniz", "unuz", "ünüz", "nız", "niz", "nuz", "nüz"],
    "+POSS.3PL": ["ları", "leri"],
    "+PAST": ["dı", "di", "du", "dü", "tı", "ti", "tu", "tü"],
    "+PROG1": ["ıyor", "iyor", "uyor", "üyor"],
    "+PROG2": ["makta", "mekte"],
    "+AOR": ["ır", "ir", "ur", "ür", "ar", "er", "r"],
    "+FUT": ["acak", "ecek"],
    "+COND": ["sa", "se"],
    "+NEG": ["ma", "me"],
    "+CAUS": ["dır", "dir", "tır", "tir", "dur", "dür", "tur", "tür", "t", "ır", "ir"],
    "+PASS": ["ıl", "il", "ul", "ül", "ın", "in", "un", "ün", "n"],
    "+BECOME": ["laş", "leş"],
    "+AGT": ["cı", "ci", "cu", "cü", "çı", "çi", "çu", "çü"],
    "+INF": ["mek", "mak", "ma", "me", "ış", "iş", "uş", "üş"],
    "+PASTPART": ["dık", "dik", "duk", "dük", "tık", "tik", "tuk", "tük"],
    "+FUTPART": ["acak", "ecek"],
    "+A1SG": ["ım", "im", "um", "üm", "m"],
    "+A2SG": ["sın", "sin", "sun", "sün", "n"],
    "+A3SG": [],  # zero morpheme
    "+A1PL": ["ız", "iz", "uz", "üz", "yız", "yiz", "yuz", "yüz", "k"],
    "+A2PL": ["sınız", "siniz", "sunuz", "sünüz", "nız", "niz", "nuz", "nüz"],
    "+A3PL": ["lar", "ler"],
    "+NOM": [],
    "+WITH": ["lı", "li", "lu", "lü"],
    "+WITHOUT": ["sız", "siz", "suz", "süz"],
    "+NESS": ["lık", "lik", "luk", "lük"],
    "+REL": ["ki"],
    "+WHILE": ["arak", "erek"],
    "+ABLE": ["abil", "ebil"],
}

# Consonant mutations at morpheme boundaries
_MUTATIONS: dict[str, str] = {"p": "b", "ç": "c", "t": "d", "k": "ğ"}


def extract_boundaries(surface: str, canonical: str) -> str:
    """Extract morpheme boundaries from surface + canonical tags.

    Returns pipe-separated segments: ``"ev|ler|i|nden"``.
    Returns the surface unchanged if extraction fails or there are no tags.

    Args:
        surface: Turkish surface form (e.g., ``"evlerinden"``).
        canonical: Canonical tag string (e.g., ``"ev +PLU +POSS.3SG +ABL"``).
    """
    parts = canonical.split()
    if not parts:
        return surface

    root = parts[0]
    tags = parts[1:]

    if not tags:
        return surface

    root_end = _find_root_end(surface, root)
    if root_end <= 0:
        return surface

    segments = [surface[:root_end]]
    remaining = surface[root_end:]

    for tag in tags:
        allomorphs = TAG_SURFACE_MAP.get(tag, [])
        matched = False
        for allomorph in sorted(allomorphs, key=len, reverse=True):
            if remaining.lower().startswith(allomorph):
                segments.append(remaining[: len(allomorph)])
                remaining = remaining[len(allomorph):]
                matched = True
                break
        if not matched and remaining:
            segments.append(remaining)
            remaining = ""
            break

    if remaining:
        segments[-1] += remaining

    return "|".join(segments)


def _find_root_end(surface: str, root: str) -> int:
    """Find where the root ends in the surface form."""
    sl = surface.lower()
    rl = root.lower()

    # Direct prefix
    if sl.startswith(rl):
        return len(root)

    # Consonant mutation: kitap→kitab
    if rl and rl[-1] in _MUTATIONS:
        mutated = rl[:-1] + _MUTATIONS[rl[-1]]
        if sl.startswith(mutated):
            return len(mutated)

    # Vowel deletion: burun→burn, ağız→ağz
    if len(rl) >= 4:
        shortened = rl[:-2] + rl[-1]
        if sl.startswith(shortened):
            return len(shortened)

    return len(root)
