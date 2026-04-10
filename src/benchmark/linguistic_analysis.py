"""Linguistic analysis of BPE vs morphological atomization.

Documents concrete failure cases where BPE fragments Turkish morphemes
but the atomizer preserves linguistic structure.
"""

from __future__ import annotations

# This module's analysis is integrated into run_all_benchmarks.py
# via the analyze_bpe_failures() function. This file provides
# the supporting data and utility functions.

# Known Turkish agglutination examples that challenge BPE
AGGLUTINATION_EXAMPLES: list[dict[str, str]] = [
    {
        "surface": "evlerinden",
        "morphemes": "ev + ler + i + nden",
        "canonical": "ev +PLU +POSS.3SG +ABL",
        "meaning": "from their houses",
        "bpe_issue": "BPE splits 'evlerinden' into arbitrary subwords "
                     "without morpheme boundaries",
    },
    {
        "surface": "Çekoslovakyalılaştıramadıklarımızdan",
        "morphemes": "Çekoslovakya + lı + laş + tır + ama + dık + lar + ımız + dan",
        "canonical": "Çekoslovakya +BECOME +CAUS +ABIL +NEG +PASTPART +PLU +POSS.1PL +ABL",
        "meaning": "from those whom we could not make into Czechoslovakians",
        "bpe_issue": "9 morphemes in one word — BPE either produces [UNK] "
                     "or fragments it into meaningless subwords",
    },
    {
        "surface": "gidiyordum",
        "morphemes": "git + iyor + du + m",
        "canonical": "git +PROG +PAST +1SG",
        "meaning": "I was going",
        "bpe_issue": "Root allomorphy: 'git' becomes 'gid-' before vowel. "
                     "BPE cannot recover the root 'git'",
    },
    {
        "surface": "güzelleştirmek",
        "morphemes": "güzel + leş + tir + mek",
        "canonical": "güzel +BECOME +CAUS +INF",
        "meaning": "to beautify",
        "bpe_issue": "4-stage derivation: adj→verb→verb→noun. "
                     "BPE subwords carry no derivation information",
    },
    {
        "surface": "kitapçılardan",
        "morphemes": "kitap + çı + lar + dan",
        "canonical": "kitap +AGT +PLU +ABL",
        "meaning": "from the booksellers",
        "bpe_issue": "Voicing: 'kitap' → 'kitapçı' (p stays). BPE fragments "
                     "may split across the voicing boundary",
    },
]


def get_failure_examples() -> list[dict[str, str]]:
    """Return the curated list of BPE failure examples."""
    return AGGLUTINATION_EXAMPLES
