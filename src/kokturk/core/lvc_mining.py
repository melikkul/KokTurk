"""Discover fused Light Verb Constructions from a corpus.

The seed table in :mod:`kokturk.core.compound_lexicon` is hand-curated and
necessarily incomplete. This module discovers additional candidates by
scanning a corpus through Zeyrek and looking for verbs whose root ends in
``-et`` or ``-ol`` AND whose stripped stem is a valid Zeyrek noun AND whose
surface shows the morphophonological alternation that distinguishes a true
fusion (gemination, vowel elision, voicing) from a plain ``noun + light verb``
spelling.

The negative filter is critical: ordinary verbs like ``gelmek`` (root
``gel``) or ``bitmek`` (root ``bit``) coincidentally end in ``-et`` once a
suffix is appended but are NOT fusions. They must be rejected.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

from kokturk.core.phonology import VOICING_MAP

LIGHT_VERBS: tuple[str, ...] = ("et", "ol")


class _ZeyrekLike(Protocol):
    """Minimal interface this module needs from a Zeyrek-style analyzer."""

    def analyze(self, word: str) -> Any:  # pragma: no cover - protocol
        ...


def _is_noun(zeyrek_analyzer: _ZeyrekLike, candidate: str) -> bool:
    """Return True if ``candidate`` parses as a noun via Zeyrek."""
    try:
        result = zeyrek_analyzer.analyze(candidate)
    except Exception:
        return False
    for word_parses in result or []:
        for parse in word_parses or []:
            pos = getattr(parse, "pos", None) or ""
            if pos.lower().startswith("noun"):
                return True
    return False


def _has_morphophonological_alternation(
    nominal: str, fused_stem: str, light_verb: str
) -> bool:
    """Verify that ``fused_stem`` is a true fusion of ``nominal + light_verb``.

    Accepts the candidate only if at least one of the following holds:

    * **gemination** — the final consonant of ``nominal`` doubles in
      ``fused_stem`` (e.g. ``ret`` + ``et`` → ``redd-et``).
    * **vowel elision** — a medial vowel of ``nominal`` drops in
      ``fused_stem`` (e.g. ``kayıp`` + ``ol`` → ``kayb-ol``,
      ``emir`` + ``et`` → ``emr-et``).
    * **voicing alternation** — the final consonant of ``nominal`` is
      voiced in ``fused_stem`` per :data:`phonology.VOICING_MAP`
      (e.g. ``bahis`` → ``bahs`` with ``s`` → ``s`` plus elision; or any
      ``p/ç/t/k`` → ``b/c/d/ğ``).

    Plain concatenation (``noun + et``) without any of these is rejected.
    """
    if not nominal or not fused_stem:
        return False
    expected_plain = nominal + light_verb
    if fused_stem == expected_plain:
        # No morphophonology applied → not a fusion.
        return False

    # Strip the trailing light_verb to expose the modified nominal.
    if not fused_stem.endswith(light_verb):
        return False
    modified = fused_stem[: -len(light_verb)]

    # Gemination check: last char doubled.
    if len(modified) >= 2 and modified[-1] == modified[-2] and nominal[-1] == modified[-1]:
        return True

    # Vowel elision check: nominal length > modified length and modified
    # is a subsequence-by-deletion of nominal restricted to vowel removal.
    if len(modified) < len(nominal):
        # Try to align by removing one medial vowel from nominal.
        for i in range(1, len(nominal) - 1):
            if nominal[i] in "aeıioöuü" and nominal[:i] + nominal[i + 1:] == modified:
                return True

    # Voicing alternation: nominal final voiceless → voiced final in modified.
    if nominal and modified and nominal[-1] in VOICING_MAP:
        if modified[-1] == VOICING_MAP[nominal[-1]]:
            return True
        # Combined elision + voicing (bahis → bahs has neither, but
        # bahis → bahz would). Cover the elision-then-voicing case:
        if (
            len(modified) == len(nominal) - 1
            and modified[:-1] == nominal[:-2]
            and modified[-1] == VOICING_MAP[nominal[-1]]
        ):
            return True

    return False


def mine_fused_lvcs_from_corpus(
    tokens: Iterable[str],
    zeyrek_analyzer: _ZeyrekLike,
) -> dict[str, tuple[str, str]]:
    """Mine fused-LVC candidates from a token stream.

    Args:
        tokens: iterable of corpus tokens (raw surface forms).
        zeyrek_analyzer: a Zeyrek-style ``MorphAnalyzer`` (or any object
            exposing ``.analyze(word)`` returning the same shape).

    Returns:
        Mapping ``fused_stem -> (nominal, light_verb)`` of newly discovered
        candidates that pass the morphophonological filter. Entries already
        present in :data:`compound_lexicon.FUSED_LVC_TABLE` are not
        rediscovered here — callers can merge as desired.
    """
    discovered: dict[str, tuple[str, str]] = {}
    seen_roots: set[str] = set()

    for token in tokens:
        try:
            parses = zeyrek_analyzer.analyze(token)
        except Exception:
            continue
        for word_parses in parses or []:
            for parse in word_parses or []:
                pos = (getattr(parse, "pos", None) or "").lower()
                if not pos.startswith("verb"):
                    continue
                root = getattr(parse, "lemma", None) or ""
                if not root or root in seen_roots:
                    continue
                seen_roots.add(root)

                for lv in LIGHT_VERBS:
                    if not root.endswith(lv) or len(root) <= len(lv):
                        continue
                    stem_modified = root[: -len(lv)]
                    # Try a few candidate nominal restorations: doubled
                    # consonant collapse, medial-vowel reinsertion, voicing.
                    candidates = _restore_nominal_candidates(stem_modified)
                    for nominal in candidates:
                        if not _is_noun(zeyrek_analyzer, nominal):
                            continue
                        if _has_morphophonological_alternation(
                            nominal, root, lv
                        ):
                            discovered[root] = (nominal, lv)
                            break
    return discovered


def _restore_nominal_candidates(modified_stem: str) -> list[str]:
    """Generate plausible underlying-noun candidates for a fused stem.

    Reverses the three alternations checked in
    :func:`_has_morphophonological_alternation`:
    de-gemination, vowel reinsertion, and de-voicing.
    """
    candidates: list[str] = [modified_stem]

    # De-gemination: collapse a doubled final consonant.
    if len(modified_stem) >= 2 and modified_stem[-1] == modified_stem[-2]:
        candidates.append(modified_stem[:-1])

    # De-voicing: voiced final → voiceless original.
    inverse_voicing = {v: k for k, v in VOICING_MAP.items()}
    if modified_stem and modified_stem[-1] in inverse_voicing:
        candidates.append(modified_stem[:-1] + inverse_voicing[modified_stem[-1]])

    # Vowel reinsertion: insert each vowel between the last two chars.
    if len(modified_stem) >= 2:
        for v in "aeıioöuü":
            candidates.append(modified_stem[:-1] + v + modified_stem[-1])

    return candidates
