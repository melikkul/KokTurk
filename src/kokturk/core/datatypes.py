"""Core data types for morphological analysis.

All public output types use frozen dataclasses with slots for immutability
and performance. Tag sequences are tuples (not lists) because Turkish
morphotactics are non-commutative — tag ORDER is semantically meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Morpheme:
    """A single morpheme with surface form and canonical representation.

    Attributes:
        surface: The actual allomorph as it appears in the word (e.g., "den", "ler").
        canonical: The normalized tag (e.g., "+ABL", "+PLU"). Turkish vowel harmony
            produces 8+ surface variants per suffix; the canonical form collapses them.
        category: Either "derivational" or "inflectional".
    """

    surface: str
    canonical: str
    category: str


@dataclass(frozen=True, slots=True)
class MorphologicalAnalysis:
    """A single morphological parse of a surface token.

    Attributes:
        surface: The original word form (e.g., "evlerinden").
        root: The lexical root (e.g., "ev").
        tags: Ordered tuple of canonical suffix tags (e.g., ("+PLU", "+POSS.3SG", "+ABL")).
            Order matters — Turkish morphotactics are non-commutative.
        morphemes: Detailed morpheme-level information.
        source: Which backend produced this analysis ("zeyrek", "trmorph", "neural", "gold").
        score: Confidence/probability score in [0.0, 1.0].
    """

    surface: str
    root: str
    tags: tuple[str, ...]
    morphemes: tuple[Morpheme, ...]
    source: str
    score: float

    @property
    def lemma(self) -> str:
        """The dictionary form (root) of the word."""
        return self.root

    def to_str(self) -> str:
        """Human-readable atom representation.

        Returns:
            String like 'ev +PLU +POSS.3SG +ABL'.
        """
        if self.tags:
            return f"{self.root} {' '.join(self.tags)}"
        return self.root

    def to_conllu(self) -> str:
        """Export to CoNLL-U morphological features format.

        Returns:
            CoNLL-U compatible feature string (e.g., 'Case=Abl|Number=Plur|...').
        """
        tag_to_conllu: dict[str, str] = {
            "+NOM": "Case=Nom",
            "+ACC": "Case=Acc",
            "+DAT": "Case=Dat",
            "+LOC": "Case=Loc",
            "+ABL": "Case=Abl",
            "+GEN": "Case=Gen",
            "+INS": "Case=Ins",
            "+PLU": "Number=Plur",
            "+POSS.1SG": "Number[psor]=Sing|Person[psor]=1",
            "+POSS.2SG": "Number[psor]=Sing|Person[psor]=2",
            "+POSS.3SG": "Number[psor]=Sing|Person[psor]=3",
            "+POSS.1PL": "Number[psor]=Plur|Person[psor]=1",
            "+POSS.2PL": "Number[psor]=Plur|Person[psor]=2",
            "+POSS.3PL": "Number[psor]=Plur|Person[psor]=3",
            "+PAST": "Tense=Past",
            "+EVID": "Evident=Nfh",
            "+AOR": "Aspect=Hab",
            "+PROG": "Aspect=Prog",
            "+FUT": "Tense=Fut",
        }
        features: list[str] = []
        for tag in self.tags:
            if tag in tag_to_conllu:
                features.append(tag_to_conllu[tag])
        return "|".join(sorted(features)) if features else "_"

    def parse_identity(self) -> tuple[str, tuple[str, ...]]:
        """Key for deduplication — two analyses with same root+tags are the same parse.

        Use this when deduplicating across backends (different scores/sources,
        same linguistic analysis).
        """
        return (self.root, self.tags)


@dataclass(frozen=True, slots=True)
class TokenAnalyses:
    """Container for all morphological analyses of a single token.

    This is what the acquisition function consumes. MAD (Morphological
    Ambiguity Density) computes mean(log(token.parse_count)) across tokens.

    Attributes:
        surface: The original word form.
        analyses: All candidate parses from all backends.
    """

    surface: str
    analyses: tuple[MorphologicalAnalysis, ...]

    @property
    def is_ambiguous(self) -> bool:
        """True when multiple valid analyses exist for this token."""
        return len(self.analyses) > 1

    @property
    def parse_count(self) -> int:
        """Number of candidate parses."""
        return len(self.analyses)

    @property
    def best(self) -> MorphologicalAnalysis | None:
        """Highest-scoring analysis, or None if no analyses exist."""
        if not self.analyses:
            return None
        return max(self.analyses, key=lambda a: a.score)
