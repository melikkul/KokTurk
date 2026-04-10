"""Canonical tag mappings for morphological feature normalization.

This is the Functional Canonization — the core innovation of morpho-tr.
Maps backend-specific feature names (Zeyrek, TRMorph) to a unified
canonical tag set using the +TAG format.

If this mapping is wrong, everything downstream breaks. Every entry
is tested in tests/core/test_constants.py.
"""

# Zeyrek feature tags → canonical morpho-tr tags
# Zeyrek uses Zemberek-style notation: "A3sg", "Pnon", "Loc", etc.
ZEYREK_TO_CANONICAL: dict[str, str] = {
    # === POS tags (kept as-is for classification, prefixed with +) ===
    "Noun": "+Noun",
    "Verb": "+Verb",
    "Adj": "+Adj",
    "Adv": "+Adv",
    "Det": "+Det",
    "Pron": "+Pron",
    "Postp": "+Postp",
    "Conj": "+Conj",
    "Interj": "+Interj",
    "Num": "+Num",
    "Ques": "+Ques",
    "Punc": "+Punc",
    "Dup": "+Dup",
    "Prop": "+Prop",
    # === Case suffixes ===
    "Nom": "+NOM",
    "Acc": "+ACC",
    "Dat": "+DAT",
    "Loc": "+LOC",
    "Abl": "+ABL",
    "Gen": "+GEN",
    "Ins": "+INS",
    "Equ": "+EQU",
    # === Number ===
    "A1sg": "",  # unmarked 1st person singular
    "A2sg": "",  # unmarked 2nd person singular
    "A3sg": "",  # unmarked 3rd person singular
    "A1pl": "+1PL",
    "A2pl": "+2PL",
    "A3pl": "+PLU",
    # === Possession ===
    "P1sg": "+POSS.1SG",
    "P2sg": "+POSS.2SG",
    "P3sg": "+POSS.3SG",
    "P1pl": "+POSS.1PL",
    "P2pl": "+POSS.2PL",
    "P3pl": "+POSS.3PL",
    "Pnon": "",  # no possession (unmarked)
    # === Tense/Aspect/Mood ===
    "Past": "+PAST",
    "Narr": "+EVID",
    "Aor": "+AOR",
    "Prog1": "+PROG",
    "Prog2": "+PROG",
    "Fut": "+FUT",
    "Pres": "+PRES",
    "Imp": "+IMP",
    "Opt": "+OPT",
    "Desr": "+DESR",
    "Neces": "+NECES",
    "Cond": "+COND",
    "Abil": "+ABIL",
    "Neg": "+NEG",
    "Caus": "+CAUS",
    "Pass": "+PASS",
    "Recip": "+RECIP",
    "Reflex": "+REFLEX",
    # === Copula and auxiliary ===
    "Cop": "+COP",
    # === Derivational suffixes ===
    "Become": "+BECOME",
    "Acquire": "+ACQUIRE",
    "Dim": "+DIM",
    "Agt": "+AGT",
    "Ness": "+NESS",
    "With": "+WITH",
    "Without": "+WITHOUT",
    "Related": "+REL",
    "FitFor": "+FITFOR",
    "Ly": "+LY",
    "Inf1": "+INF",
    "Inf2": "+INF",
    "Inf3": "+INF",
    "PastPart": "+PASTPART",
    "FutPart": "+FUTPART",
    "PresPart": "+PRESPART",
    "NarrPart": "+NARRPART",
    "AorPart": "+AORPART",
    "NotState": "+NOTSTATE",
    "FeelLike": "+FEELLIKE",
    "JustLike": "+JUSTLIKE",
    "AsIf": "+ASIF",
    "While": "+WHILE",
    "When": "+WHEN",
    "SinceDoingSo": "+SINCEDOINGSO",
    "ByDoingSo": "+BYDOINGSO",
    "AdamantlyDoingSo": "+ADAMANTLY",
    "AfterDoingSo": "+AFTERDOINGSO",
    "WithoutDoingSo": "+WITHOUTDOINGSO",
    "AsLongAs": "+ASLONGAS",
    "InsteadOfDoingSo": "+INSTEAD",
    "Zero": "",  # zero derivation (unmarked)
    "Rel": "+REL",  # relative clause marker (-ki)
    # Internal Zeyrek markers for unknown/unparseable tokens
    "U": "",  # unknown
    "n": "",  # unknown component
    "k": "",  # unknown component
}

# TRMorph FST feature tags → canonical morpho-tr tags
# TRMorph uses lowercase notation: "pl", "loc", "abl", etc.
TRMORPH_TO_CANONICAL: dict[str, str] = {
    # === POS tags ===
    "N": "+Noun",
    "V": "+Verb",
    "Adj": "+Adj",
    "Adv": "+Adv",
    "Det": "+Det",
    "Pron": "+Pron",
    "Postp": "+Postp",
    "Conj": "+Conj",
    "Interj": "+Interj",
    "Num": "+Num",
    # === Case ===
    "nom": "+NOM",
    "acc": "+ACC",
    "dat": "+DAT",
    "loc": "+LOC",
    "abl": "+ABL",
    "gen": "+GEN",
    "ins": "+INS",
    "equ": "+EQU",
    # === Number ===
    "sg": "",  # unmarked singular
    "pl": "+PLU",
    # === Possession ===
    "p1s": "+POSS.1SG",
    "p2s": "+POSS.2SG",
    "p3s": "+POSS.3SG",
    "p1p": "+POSS.1PL",
    "p2p": "+POSS.2PL",
    "p3p": "+POSS.3PL",
    # === Tense/Aspect/Mood ===
    "past": "+PAST",
    "narr": "+EVID",
    "aor": "+AOR",
    "prog": "+PROG",
    "fut": "+FUT",
    "pres": "+PRES",
    "imp": "+IMP",
    "opt": "+OPT",
    "neces": "+NECES",
    "cond": "+COND",
    "abil": "+ABIL",
    "neg": "+NEG",
    "caus": "+CAUS",
    "pass": "+PASS",
    "recip": "+RECIP",
    "reflex": "+REFLEX",
    "cop": "+COP",
}

# Canonical tags that map to empty string (unmarked features) — used in tests
UNMARKED_FEATURES: frozenset[str] = frozenset(
    k for k, v in ZEYREK_TO_CANONICAL.items() if v == ""
)
