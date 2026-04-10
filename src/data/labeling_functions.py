"""Six labeling functions for weak supervision of morphological parsing.

Each LF takes a token context and returns either a predicted label index
or ABSTAIN (-1). Labels are indices into the candidate parse list for
that token.

LF1: lf_zeyrek_unambiguous — Zeyrek returns exactly 1 parse
LF2: lf_trmorph_unambiguous — TRMorph returns exactly 1 parse
LF3: lf_suffix_regex — High-precision suffix pattern matching
LF4: lf_pos_bigram — POS bigram transition constraints from BOUN
LF5: lf_gazetteer — Proper noun gazetteer from BOUN lemmas
LF6: lf_neural_draft — Stub (BERTurk, implemented in Phase 2)
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ABSTAIN = -1

# ============================================================
# Token context passed to each LF
# ============================================================


class TokenContext:
    """Context for a single token passed to labeling functions.

    Attributes:
        surface: The word form.
        analyses: List of candidate parse dicts with root, tags, source.
        prev_pos: POS tag of the previous token (or None).
        next_pos: POS tag of the next token (or None).
        sentence_id: ID of the containing sentence.
        token_idx: Position in the sentence.
    """

    __slots__ = ("surface", "analyses", "prev_pos", "next_pos", "sentence_id", "token_idx")

    def __init__(
        self,
        surface: str,
        analyses: list[dict[str, object]],
        prev_pos: str | None = None,
        next_pos: str | None = None,
        sentence_id: str = "",
        token_idx: int = 0,
    ) -> None:
        self.surface = surface
        self.analyses = analyses
        self.prev_pos = prev_pos
        self.next_pos = next_pos
        self.sentence_id = sentence_id
        self.token_idx = token_idx


# ============================================================
# LF1: Zeyrek unambiguous
# ============================================================


def lf_zeyrek_unambiguous(ctx: TokenContext) -> int:
    """If Zeyrek returns exactly 1 parse, return that parse index."""
    zeyrek_indices = [
        i for i, a in enumerate(ctx.analyses)
        if a.get("source") == "zeyrek"
    ]
    if len(zeyrek_indices) == 1:
        return zeyrek_indices[0]
    return ABSTAIN


lf_zeyrek_unambiguous.name = "zeyrek_unambiguous"  # type: ignore[attr-defined]


# ============================================================
# LF2: TRMorph unambiguous
# ============================================================


def lf_trmorph_unambiguous(ctx: TokenContext) -> int:
    """If TRMorph returns exactly 1 parse, return that parse index."""
    trmorph_indices = [
        i for i, a in enumerate(ctx.analyses)
        if a.get("source") == "trmorph"
    ]
    if len(trmorph_indices) == 1:
        return trmorph_indices[0]
    return ABSTAIN


lf_trmorph_unambiguous.name = "trmorph_unambiguous"  # type: ignore[attr-defined]


# ============================================================
# LF3: Suffix regex patterns
# ============================================================

# High-precision suffix patterns. Each pattern matches a surface suffix
# and predicts a specific canonical tag.
_SUFFIX_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Evidential past: -mIş
    (re.compile(r"(?:miş|mış|müş|muş)$", re.IGNORECASE), "+EVID"),
    # Ablative: -DAn
    (re.compile(r"(?:den|dan|ten|tan)$", re.IGNORECASE), "+ABL"),
    # Locative: -DA
    (re.compile(r"(?:de|da|te|ta)$", re.IGNORECASE), "+LOC"),
    # Dative: -yA / -A
    (re.compile(r"(?:ye|ya|ne|na|e|a)$", re.IGNORECASE), "+DAT"),
    # Instrumental: -ylA
    (re.compile(r"(?:yle|yla|le|la)$", re.IGNORECASE), "+INS"),
]


def lf_suffix_regex(ctx: TokenContext) -> int:
    """Match high-precision suffix patterns against candidate parses.

    Only votes if a suffix pattern matches AND exactly one candidate parse
    contains the corresponding tag. This avoids false positives from
    ambiguous suffix forms.
    """
    surface_lower = ctx.surface.lower()

    for pattern, expected_tag in _SUFFIX_PATTERNS:
        if not pattern.search(surface_lower):
            continue

        # Find candidates that have this tag
        matching_indices = [
            i for i, a in enumerate(ctx.analyses)
            if expected_tag in a.get("tags", [])
        ]
        if len(matching_indices) == 1:
            return matching_indices[0]

    return ABSTAIN


lf_suffix_regex.name = "suffix_regex"  # type: ignore[attr-defined]


# ============================================================
# LF4: POS bigram transitions
# ============================================================


class POSBigramModel:
    """POS bigram transition model from BOUN Treebank.

    Loads POS bigram counts and uses them to select the most likely
    parse based on the POS of the preceding token.
    """

    def __init__(self) -> None:
        self._bigrams: dict[str, Counter[str]] = defaultdict(Counter)
        self._loaded = False

    def load_from_boun(self, boun_dir: Path = Path("data/external/boun_treebank")) -> None:
        """Extract POS bigram counts from CoNLL-U files."""
        for split in ["train"]:  # Only use training data to avoid leakage
            path = boun_dir / f"tr_boun-ud-{split}.conllu"
            if not path.exists():
                continue

            prev_pos = "<S>"  # sentence start
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if line == "":
                        prev_pos = "<S>"
                        continue
                    if line.startswith("#"):
                        continue
                    fields = line.split("\t")
                    if len(fields) < 4:
                        continue
                    if "-" in fields[0] or "." in fields[0]:
                        continue
                    pos = fields[3]
                    self._bigrams[prev_pos][pos] += 1
                    prev_pos = pos

        self._loaded = True

    def most_likely_pos(self, prev_pos: str) -> str | None:
        """Return the most likely POS following prev_pos."""
        if not self._loaded or prev_pos not in self._bigrams:
            return None
        return self._bigrams[prev_pos].most_common(1)[0][0]


# Global POS bigram model (lazy-loaded)
_pos_bigram_model: POSBigramModel | None = None

# Map from Zeyrek POS tags to UPOS
_ZEYREK_POS_TO_UPOS: dict[str, str] = {
    "+Noun": "NOUN", "+Verb": "VERB", "+Adj": "ADJ", "+Adv": "ADV",
    "+Det": "DET", "+Pron": "PRON", "+Postp": "ADP", "+Conj": "CCONJ",
    "+Interj": "INTJ", "+Num": "NUM", "+Prop": "PROPN", "+Punc": "PUNCT",
}


def _get_pos_bigram_model() -> POSBigramModel:
    global _pos_bigram_model
    if _pos_bigram_model is None:
        _pos_bigram_model = POSBigramModel()
        _pos_bigram_model.load_from_boun()
    return _pos_bigram_model


def _analysis_upos(analysis: dict[str, object]) -> str | None:
    """Extract UPOS from a candidate analysis."""
    tags: list[str] = analysis.get("tags", [])  # type: ignore[assignment]
    for tag in tags:
        if tag in _ZEYREK_POS_TO_UPOS:
            return _ZEYREK_POS_TO_UPOS[tag]
    return None


def lf_pos_bigram(ctx: TokenContext) -> int:
    """Use POS bigram model to select parse consistent with context.

    Votes for a parse whose POS matches the most likely POS transition
    from the previous token. Only votes when exactly one candidate
    matches the expected POS.
    """
    if ctx.prev_pos is None:
        return ABSTAIN

    model = _get_pos_bigram_model()
    expected_pos = model.most_likely_pos(ctx.prev_pos)
    if expected_pos is None:
        return ABSTAIN

    matching_indices = [
        i for i, a in enumerate(ctx.analyses)
        if _analysis_upos(a) == expected_pos
    ]

    if len(matching_indices) == 1:
        return matching_indices[0]
    return ABSTAIN


lf_pos_bigram.name = "pos_bigram"  # type: ignore[attr-defined]


# ============================================================
# LF5: Gazetteer (proper nouns)
# ============================================================

_proper_noun_gazetteer: set[str] | None = None


def _get_gazetteer(boun_dir: Path = Path("data/external/boun_treebank")) -> set[str]:
    """Build a proper noun gazetteer from BOUN Treebank PROPN lemmas."""
    global _proper_noun_gazetteer
    if _proper_noun_gazetteer is not None:
        return _proper_noun_gazetteer

    propn_lemmas: set[str] = set()
    for split in ["train"]:
        path = boun_dir / f"tr_boun-ud-{split}.conllu"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#") or line.strip() == "":
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 4:
                    continue
                if "-" in fields[0] or "." in fields[0]:
                    continue
                if fields[3] == "PROPN":
                    propn_lemmas.add(fields[1])  # surface form
                    propn_lemmas.add(fields[2])  # lemma

    _proper_noun_gazetteer = propn_lemmas
    return _proper_noun_gazetteer


def lf_gazetteer(ctx: TokenContext) -> int:
    """If token matches a known proper noun, vote for the Noun parse."""
    gazetteer = _get_gazetteer()

    if ctx.surface not in gazetteer and ctx.surface.capitalize() not in gazetteer:
        return ABSTAIN

    # Find a Noun parse candidate
    noun_indices = [
        i for i, a in enumerate(ctx.analyses)
        if "+Noun" in a.get("tags", []) or "+Prop" in a.get("tags", [])  # type: ignore[operator]
    ]

    if len(noun_indices) >= 1:
        return noun_indices[0]
    return ABSTAIN


lf_gazetteer.name = "gazetteer"  # type: ignore[attr-defined]


# ============================================================
# LF6: Neural draft (stub — Phase 2)
# ============================================================


def lf_neural_draft(ctx: TokenContext) -> int:
    """Neural draft model using fine-tuned BERTurk.

    TODO: Implement in Phase 2 when BERTurk is fine-tuned on BOUN data.
    Currently always returns ABSTAIN.
    """
    return ABSTAIN


lf_neural_draft.name = "neural_draft"  # type: ignore[attr-defined]


# ============================================================
# All labeling functions registry
# ============================================================

ALL_LFS = [
    lf_zeyrek_unambiguous,
    lf_trmorph_unambiguous,
    lf_suffix_regex,
    lf_pos_bigram,
    lf_gazetteer,
    lf_neural_draft,
]


# ============================================================
# Label matrix generation
# ============================================================


def build_label_matrix(
    candidates_path: Path = Path("data/prelabeled/candidates.jsonl"),
    corpus_path: Path = Path("data/processed/corpus.jsonl"),
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Apply all labeling functions to every token and build label matrix.

    Args:
        candidates_path: Path to prelabeled candidates JSONL.
        corpus_path: Path to ingested corpus JSONL.

    Returns:
        Tuple of (label_matrix, records) where label_matrix has shape
        (n_tokens, n_lfs) and records is the list of candidate dicts.
    """
    # Load corpus for POS context
    corpus_by_id: dict[str, dict[str, object]] = {}
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            sent = json.loads(line)
            corpus_by_id[sent["sentence_id"]] = sent

    # Load candidates
    records: list[dict[str, object]] = []
    with open(candidates_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    n_tokens = len(records)
    n_lfs = len(ALL_LFS)
    label_matrix = np.full((n_tokens, n_lfs), ABSTAIN, dtype=np.int32)

    for i, record in enumerate(records):
        sent_id = str(record["sentence_id"])
        token_idx = int(record["token_idx"])  # type: ignore[arg-type]
        sent = corpus_by_id.get(sent_id, {})
        pos_tags: list[str] = sent.get("pos_tags", [])  # type: ignore[assignment]

        prev_pos = pos_tags[token_idx - 1] if token_idx > 0 and pos_tags else None
        next_pos = pos_tags[token_idx + 1] if token_idx + 1 < len(pos_tags) else None

        ctx = TokenContext(
            surface=str(record["surface"]),
            analyses=record["analyses"],  # type: ignore[arg-type]
            prev_pos=prev_pos,
            next_pos=next_pos,
            sentence_id=sent_id,
            token_idx=token_idx,
        )

        for j, lf in enumerate(ALL_LFS):
            label_matrix[i, j] = lf(ctx)

    return label_matrix, records


def compute_lf_stats(label_matrix: np.ndarray) -> dict[str, dict[str, float]]:
    """Compute per-LF coverage, overlap, and conflict statistics.

    Returns:
        Dict mapping LF name to {coverage, overlap_rate, conflict_rate}.
    """
    n_tokens, n_lfs = label_matrix.shape
    stats: dict[str, dict[str, float]] = {}

    for j, lf in enumerate(ALL_LFS):
        votes = label_matrix[:, j]
        voted = votes != ABSTAIN
        coverage = float(voted.sum()) / max(n_tokens, 1)

        # Overlap: fraction of tokens where this LF AND at least one other LF vote
        overlap_count = 0
        conflict_count = 0
        for i in range(n_tokens):
            if votes[i] == ABSTAIN:
                continue
            other_votes = [
                label_matrix[i, k]
                for k in range(n_lfs)
                if k != j and label_matrix[i, k] != ABSTAIN
            ]
            if other_votes:
                overlap_count += 1
                if any(v != votes[i] for v in other_votes):
                    conflict_count += 1

        voted_count = int(voted.sum())
        stats[lf.name] = {  # type: ignore[attr-defined]
            "coverage": coverage,
            "overlap_rate": overlap_count / max(voted_count, 1),
            "conflict_rate": conflict_count / max(voted_count, 1),
        }

    return stats
