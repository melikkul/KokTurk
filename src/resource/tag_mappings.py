"""Tag mapping tables: UD CoNLL-U features → canonical morpho-tr tags.
Also UniMorph → canonical.

The canonical tag format uses the +TAG convention defined in
kokturk/core/constants.py. This module provides the bridge from
external annotation schemes to that canonical representation.
"""
from __future__ import annotations


# UD feature key=value → canonical tag (empty string means unmarked/skip)
UD_TO_CANONICAL: dict[str, str] = {
    # Case (nominative is unmarked)
    "Case=Nom": "",
    "Case=Gen": "+GEN",
    "Case=Dat": "+DAT",
    "Case=Acc": "+ACC",
    "Case=Loc": "+LOC",
    "Case=Abl": "+ABL",
    "Case=Ins": "+INS",
    "Case=Equ": "+EQU",
    "Case=Voc": "",        # vocative rare in Turkish
    # Number (singular unmarked)
    "Number=Sing": "",
    "Number=Plur": "+PLU",
    # Tense
    "Tense=Past": "+PAST",
    "Tense=Pres": "+PRES",
    "Tense=Fut": "+FUT",
    # Aspect
    "Aspect=Perf": "",     # perfective subsumed by tense
    "Aspect=Prog": "+PROG",
    "Aspect=Hab": "+AOR",
    # Mood
    "Mood=Ind": "",
    "Mood=Imp": "+IMP",
    "Mood=Opt": "+OPT",
    "Mood=Desr": "+DESR",
    "Mood=Neces": "+NECES",
    "Mood=Cond": "+COND",
    # Evidence
    "Evident=Fh": "",      # first-hand evidence (default)
    "Evident=Nfh": "+EVID",
    # Polarity
    "Polarity=Pos": "",
    "Polarity=Neg": "+NEG",
    # VerbForm
    "VerbForm=Fin": "",
    "VerbForm=Inf": "+INF",
    "VerbForm=Part": "+AORPART",   # approximate; more specific type ignored here
    "VerbForm=Conv": "+BYDOINGSO", # approximate
    # Voice
    "Voice=Cau": "+CAUS",
    "Voice=Pass": "+PASS",
    "Voice=Rcp": "+RECIP",
    "Voice=Rfl": "+REFLEX",
    # POS (Universal POS → canonical)
    "NOUN": "+Noun",
    "VERB": "+Verb",
    "ADJ": "+Adj",
    "ADV": "+Adv",
    "DET": "+Det",
    "PRON": "+Pron",
    "ADP": "+Postp",
    "CCONJ": "+Conj",
    "SCONJ": "+Conj",
    "INTJ": "+Interj",
    "NUM": "+Num",
    "PROPN": "+Prop",
    "PUNCT": "+Punc",
    "AUX": "+Verb",       # Turkish auxiliaries treated as verbs
    "PART": "",
    "X": "",
}

# UniMorph feature tag → canonical tag
# UniMorph tags from the 'tur' repository use semicolon-delimited features
# on a single column. This maps individual features.
UNIMORPH_TO_CANONICAL: dict[str, str] = {
    # POS
    "N": "+Noun",
    "V": "+Verb",
    "ADJ": "+Adj",
    "ADV": "+Adv",
    "DET": "+Det",
    "PRO": "+Pron",
    "POST": "+Postp",
    "CONJ": "+Conj",
    "PROPN": "+Prop",
    # Case
    "NOM": "",
    "GEN": "+GEN",
    "DAT": "+DAT",
    "ACC": "+ACC",
    "LOC": "+LOC",
    "ABL": "+ABL",
    "INS": "+INS",
    "EQU": "+EQU",
    # Number
    "SG": "",
    "PL": "+PLU",
    # Possessive (UniMorph uses PSS1S, PSS2S, etc.)
    "PSS1S": "+POSS.1SG",
    "PSS2S": "+POSS.2SG",
    "PSS3S": "+POSS.3SG",
    "PSS1P": "+POSS.1PL",
    "PSS2P": "+POSS.2PL",
    "PSS3P": "+POSS.3PL",
    # Tense
    "PST": "+PAST",
    "PRS": "+PRES",
    "FUT": "+FUT",
    # Aspect
    "PROG": "+PROG",
    "HAB": "+AOR",
    "PERF": "",
    # Mood
    "IND": "",
    "IMP": "+IMP",
    "OPT": "+OPT",
    "DES": "+DESR",
    "NEC": "+NECES",
    "COND": "+COND",
    # Evidence
    "NFHEVID": "+EVID",
    # Polarity
    "POS": "",
    "NEG": "+NEG",
    # Person
    "1": "",   # person alone not enough, handled with number
    "2": "",
    "3": "",
    # VerbForm
    "PART": "+AORPART",
    "CONV": "+BYDOINGSO",
    "INF": "+INF",
    # Plurals for verb agreement
    "1PL": "+1PL",
    "2PL": "+2PL",
}

# Maps for UD agreement suffixes: (Person, Number) → canonical
_UD_AGREEMENT: dict[tuple[str, str], str] = {
    ("1", "Plur"): "+1PL",
    ("2", "Plur"): "+2PL",
    # 3rd person plural for nouns is already handled by Number=Plur → +PLU
    # 1sg, 2sg, 3sg are unmarked in Turkish (no canonical tag)
}

# Maps for UD possessive: (Person[psor], Number[psor]) → canonical
_UD_POSSESSIVE: dict[tuple[str, str], str] = {
    ("1", "Sing"): "+POSS.1SG",
    ("2", "Sing"): "+POSS.2SG",
    ("3", "Sing"): "+POSS.3SG",
    ("1", "Plur"): "+POSS.1PL",
    ("2", "Plur"): "+POSS.2PL",
    ("3", "Plur"): "+POSS.3PL",
}


def ud_feats_to_canonical(lemma: str, pos: str, feats: str) -> str:
    """Convert CoNLL-U FORM+UPOS+FEATS to a canonical morpho-tr tag string.

    Example::

        ud_feats_to_canonical("ev", "NOUN", "Case=Loc|Number=Sing|Person=3")
        # → "ev +Noun +LOC"

    Args:
        lemma: Token lemma from CoNLL-U LEMMA column.
        pos: Universal POS tag from UPOS column.
        feats: Feature string from FEATS column (e.g. "Case=Loc|Number=Sing").
               Pass "_" or "" if absent.

    Returns:
        Canonical label string starting with lemma, followed by space-separated
        +TAG tokens in morphotactic order.
    """
    tags: list[str] = []

    # POS tag
    pos_tag = UD_TO_CANONICAL.get(pos, "")
    if pos_tag:
        tags.append(pos_tag)

    if feats and feats != "_":
        feat_dict: dict[str, str] = {}
        for feat in feats.split("|"):
            if "=" in feat:
                k, v = feat.split("=", 1)
                feat_dict[k] = v

        # Plural number
        if feat_dict.get("Number") == "Plur":
            tags.append("+PLU")

        # Possessive: requires BOTH Person[psor] AND Number[psor]
        psor_person = feat_dict.get("Person[psor]")
        psor_number = feat_dict.get("Number[psor]", "Sing")
        if psor_person:
            poss_tag = _UD_POSSESSIVE.get((psor_person, psor_number))
            if poss_tag:
                tags.append(poss_tag)

        # Case
        case_tag = UD_TO_CANONICAL.get(f"Case={feat_dict.get('Case', 'Nom')}", "")
        if case_tag:
            tags.append(case_tag)

        # Tense
        tense_tag = UD_TO_CANONICAL.get(f"Tense={feat_dict.get('Tense', '')}", "")
        if tense_tag:
            tags.append(tense_tag)

        # Evidence (narrative/hearsay)
        evident_tag = UD_TO_CANONICAL.get(f"Evident={feat_dict.get('Evident', '')}", "")
        if evident_tag:
            tags.append(evident_tag)

        # Aspect
        aspect_tag = UD_TO_CANONICAL.get(f"Aspect={feat_dict.get('Aspect', '')}", "")
        if aspect_tag:
            tags.append(aspect_tag)

        # Mood
        mood_tag = UD_TO_CANONICAL.get(f"Mood={feat_dict.get('Mood', '')}", "")
        if mood_tag:
            tags.append(mood_tag)

        # Polarity
        if feat_dict.get("Polarity") == "Neg":
            tags.append("+NEG")

        # Verb agreement (person × number for plural only; singular unmarked)
        person = feat_dict.get("Person", "")
        number = feat_dict.get("Number", "Sing")
        agr_tag = _UD_AGREEMENT.get((person, number))
        if agr_tag:
            tags.append(agr_tag)

        # Voice
        voice_tag = UD_TO_CANONICAL.get(f"Voice={feat_dict.get('Voice', '')}", "")
        if voice_tag:
            tags.append(voice_tag)

        # VerbForm (participles, infinitives, converbs)
        vf_tag = UD_TO_CANONICAL.get(f"VerbForm={feat_dict.get('VerbForm', '')}", "")
        if vf_tag:
            tags.append(vf_tag)

    return lemma + (" " + " ".join(tags) if tags else "")


def unimorph_tags_to_canonical(lemma: str, form: str, tag_string: str) -> str:
    """Convert UniMorph tag string to canonical morpho-tr label.

    UniMorph uses semicolon-delimited tags (e.g., ``"N;NOM;PL"``).

    Args:
        lemma: Dictionary form of the word.
        form: Inflected surface form.
        tag_string: Semicolon-delimited UniMorph feature tags.

    Returns:
        Canonical label string.
    """
    tags: list[str] = []
    for raw_tag in tag_string.split(";"):
        raw_tag = raw_tag.strip()
        canonical = UNIMORPH_TO_CANONICAL.get(raw_tag, "")
        if canonical:
            tags.append(canonical)
    return lemma + (" " + " ".join(tags) if tags else "")
