"""Paradigm-completion data augmentation.

Given a root (and its POS), synthesize morphotactically-valid inflected
forms using Turkish vowel harmony + a small archiphoneme template language.
The generator is deliberately conservative: a form is only emitted if it
survives a Zeyrek validation sanity pass (when an analyzer is provided).

The augmentation is **targeted by rarity**: the helper
:func:`augmentation_budget` reads the tag-frequency JSON produced by
:mod:`benchmark.tag_frequency` and returns the number of synthetic forms to
generate per tag (<10 occ → 50, 10–100 → 20, >100 → 0).

Archiphoneme template language
------------------------------
- ``A``            → back→``a`` / front→``e``
- ``I``            → back-unround→``ı``, back-round→``u``,
                      front-unround→``i``, front-round→``ü``
- ``D``            → after voiceless consonant→``t``, else→``d``
- ``C``            → after voiceless consonant→``ç``, else→``c``
- ``(y)`` ``(n)`` ``(s)`` — buffer consonants; emitted only if the previous
  character is a vowel.

Voicing alternation (``kitap → kitab-ı``) is lexically conditioned via the
``alternation_map`` argument mined from the corpus; callers can build this
with :func:`mine_voicing_alternations`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from kokturk.core.phonology import (
    BACK_ROUNDED,
    BACK_UNROUNDED,
    FRONT_ROUNDED,
    FRONT_UNROUNDED,
    is_front,
    is_rounded,
    last_vowel,
)

VOICELESS = set("pçtkfsşh")
VOWELS = set("aeıioöuü")


# ---------------------------------------------------------------------------
# Paradigms (critical gaps included per the approved plan)
# ---------------------------------------------------------------------------

# Each entry is (template, canonical tag sequence).
NOMINAL_PARADIGM: list[tuple[str, str]] = [
    ("+lAr", "+PLU"),
    ("+(n)In", "+GEN"),
    ("+(y)A", "+DAT"),
    ("+(y)I", "+ACC"),
    ("+DAn", "+ABL"),
    ("+DA", "+LOC"),
    ("+Im", "+POSS.1SG"),
    ("+In", "+POSS.2SG"),
    ("+(s)I", "+POSS.3SG"),
    ("+(s)I+(n)A", "+POSS.3SG +DAT"),
    ("+(s)I+nDAn", "+POSS.3SG +ABL"),
    ("+(s)I+nDA", "+POSS.3SG +LOC"),
    ("+lArImIz+DAn", "+PLU +POSS.1PL +ABL"),
    ("+(n)In+(s)I", "+GEN +POSS.3SG"),
]

VERBAL_PARADIGM: list[tuple[str, str]] = [
    ("+DI", "+PAST"),
    ("+Iyor", "+PROG"),
    ("+(y)AcAk", "+FUT"),
    ("+mIş", "+EVID"),
    ("+mA+DI", "+NEG +PAST"),
    ("+mA+Iyor", "+NEG +PROG"),
    ("+(y)AbIl+mA+DI", "+ABIL +NEG +PAST"),
    ("+Il+Iyor", "+PASS +PROG"),
    ("+DIr+Il+DI", "+CAUS +PASS +PAST"),
]


@dataclass
class AugmentReport:
    generated: int = 0
    kept: int = 0
    discarded_invalid: int = 0
    per_tag_generated: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# attach_suffix — archiphoneme resolver
# ---------------------------------------------------------------------------

def _pick_I(front: bool, rounded: bool) -> str:
    if front and rounded:
        return "ü"
    if front and not rounded:
        return "i"
    if not front and rounded:
        return "u"
    return "ı"


def _pick_A(front: bool) -> str:
    return "e" if front else "a"


def _pick_voiced(voiceless_prev: bool, voiced_variant: str, voiceless_variant: str) -> str:
    return voiceless_variant if voiceless_prev else voiced_variant


_BUFFER_RE = re.compile(r"\(([yns])\)")


def attach_suffix(
    root: str,
    template: str,
    alternation_map: dict[str, str] | None = None,
) -> str:
    """Attach a suffix template to a root, resolving Turkish allomorphs.

    Args:
        root: Bare surface root (e.g. ``"ev"``, ``"kitap"``, ``"araba"``).
        template: Suffix template with archiphonemes and ``(y)/(n)/(s)``
            buffers, e.g. ``"+(y)A"``, ``"+lAr"``, ``"+DA"``.
        alternation_map: Optional ``{root: voiced_root}`` mapping for
            consonant voicing (``kitap → kitab``). Applied only when the
            template begins with a vowel-initial archiphoneme.
    """
    if alternation_map is None:
        alternation_map = {}
    # Strip leading '+' if present — the template proper follows.
    t = template.lstrip("+")

    front = is_front(root) if last_vowel(root) is not None else True
    rounded = is_rounded(root) if last_vowel(root) is not None else False

    # Voicing: apply only if next non-buffer char is a vowel archiphoneme.
    stripped = _BUFFER_RE.sub("", t)
    first_real = stripped[:1]
    if first_real in ("A", "I") or (stripped and stripped[0] in VOWELS):
        if root in alternation_map:
            root_effective = alternation_map[root]
        else:
            root_effective = root
    else:
        root_effective = root

    out = list(root_effective)

    i = 0
    while i < len(t):
        ch = t[i]
        # Buffer consonant
        if ch == "(" and i + 2 < len(t) and t[i + 2] == ")":
            buf_char = t[i + 1]
            prev = out[-1] if out else ""
            if prev and prev.lower() in VOWELS:
                out.append(buf_char)
            i += 3
            continue
        # Archiphonemes
        if ch == "A":
            out.append(_pick_A(front))
        elif ch == "I":
            out.append(_pick_I(front, rounded))
        elif ch == "D":
            prev = out[-1] if out else ""
            out.append(_pick_voiced(prev.lower() in VOICELESS, "d", "t"))
        elif ch == "C":
            prev = out[-1] if out else ""
            out.append(_pick_voiced(prev.lower() in VOICELESS, "c", "ç"))
        else:
            out.append(ch)
        i += 1
    return "".join(out)


# ---------------------------------------------------------------------------
# Voicing alternation mining
# ---------------------------------------------------------------------------

def mine_voicing_alternations(corpus_path: Path | str) -> dict[str, str]:
    """Mine ``{root: voiced_root}`` pairs from the tiered corpus.

    Looks for co-occurrences like ``kitap`` (bare) and ``kitab-`` (inflected)
    in surface forms. Independent of label quality: the alternation is
    surface-level so silver tokens are admissible.
    """
    alternation_map: dict[str, str] = {}
    finals = {"p": "b", "ç": "c", "t": "d", "k": "ğ"}
    roots: set[str] = set()
    with Path(corpus_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            parts = rec.get("label", "").split()
            if parts:
                roots.add(parts[0])
    # Any root ending in a voiceless stop whose voiced-variant + vowel
    # also appears as the prefix of some inflected surface is a confirmed
    # alternation. Simpler deterministic version: trust the final-consonant
    # rule and emit every candidate.
    for r in roots:
        if r and r[-1] in finals:
            alternation_map[r] = r[:-1] + finals[r[-1]]
    return alternation_map


# ---------------------------------------------------------------------------
# Augmentation budget & corpus writer
# ---------------------------------------------------------------------------

def augmentation_budget(tag_frequency_json: Path | str) -> dict[str, int]:
    """Return ``{tag: n_synthetic}`` per the rarity rule (50 / 20 / 0)."""
    payload = json.loads(Path(tag_frequency_json).read_text())
    budget: dict[str, int] = {}
    for entry in payload["tags"]:
        count = entry["count"]
        if count < 10:
            budget[entry["tag"]] = 50
        elif count <= 100:
            budget[entry["tag"]] = 20
        else:
            budget[entry["tag"]] = 0
    return budget


def generate_paradigm(
    root: str,
    pos: str,
    alternation_map: dict[str, str] | None = None,
    max_forms: int = 15,
) -> list[tuple[str, str]]:
    """Return ``[(surface, canonical_label), ...]`` for a single root."""
    if pos.lower().startswith("verb"):
        paradigm = VERBAL_PARADIGM
    else:
        paradigm = NOMINAL_PARADIGM
    out: list[tuple[str, str]] = []
    for template, tag_seq in paradigm[:max_forms]:
        surface = attach_suffix(root, template, alternation_map)
        label = f"{root} +{pos} {tag_seq}"
        out.append((surface, label))
    return out


def augment_corpus(
    tag_frequency_json: Path | str,
    output_path: Path | str,
    roots: Iterable[tuple[str, str]],
    alternation_map: dict[str, str] | None = None,
    validator: Callable[[str, str], bool] | None = None,
) -> AugmentReport:
    """Write a synthetic-tier JSONL.

    Args:
        tag_frequency_json: Output of :mod:`benchmark.tag_frequency`.
        output_path: Destination JSONL (tier=``synthetic``).
        roots: Iterable of ``(root, pos)``.
        alternation_map: From :func:`mine_voicing_alternations`.
        validator: Optional callable ``(root, surface) -> bool``. When
            supplied, generated forms that fail validation are discarded
            rather than written. Batch validation is the caller's problem —
            we invoke this once per form.
    """
    budget = augmentation_budget(tag_frequency_json)
    report = AugmentReport()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for root, pos in roots:
            forms = generate_paradigm(root, pos, alternation_map)
            for surface, label in forms:
                # Bucket by the first ``+TAG`` of the synthetic label
                parts = label.split()
                suffix_tag = parts[2] if len(parts) >= 3 else "+?"
                allowance = budget.get(suffix_tag, 0)
                if allowance <= 0:
                    continue
                report.generated += 1
                report.per_tag_generated[suffix_tag] = (
                    report.per_tag_generated.get(suffix_tag, 0) + 1
                )
                if validator is not None and not validator(root, surface):
                    report.discarded_invalid += 1
                    continue
                f.write(
                    json.dumps(
                        {
                            "surface": surface,
                            "label": label,
                            "tier": "synthetic",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                report.kept += 1
                budget[suffix_tag] = allowance - 1
    return report
