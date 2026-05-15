"""Source corpus manifest. Every entry has license, sha256, version pin."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Source:
    name: str
    url: str
    license: str
    version: str
    sha256: str | None
    redistribute: bool  # False ⇒ internal eval only; NOT in published HF/Zenodo dataset


SOURCES: list[Source] = [
    Source(
        "oscar-tr",
        "oscar-corpus/OSCAR-2301",
        "CC0+CC-BY-4.0",
        "2301",
        None,
        True,
    ),
    Source(
        "mc4-tr",
        "allenai/c4",
        "ODC-BY",
        "noclean.tr",
        None,
        True,
    ),
    Source(
        "wiki-tr",
        "wikimedia/wikipedia",
        "CC-BY-SA-3.0",
        "20250401.tr",
        None,
        True,  # SA shard stays CC BY-SA; published shard is labeled CC BY-SA
    ),
    Source(
        "boun-ud",
        "UniversalDependencies/UD_Turkish-BOUN",
        "CC-BY-SA-4.0",
        "<pin>",  # replaced with actual SHA at fetch time
        None,
        True,  # SA shard stays CC BY-SA; published shard labeled CC BY-SA
    ),
    # IMST is CC-BY-NC-SA-3.0 — NC clause is incompatible with CC BY 4.0 main dataset
    # license and would block downstream commercial use. Use IMST only for internal eval.
    # D-Step 6 filter: redistribute=False ⇒ excluded from published HF/Zenodo slices.
    Source(
        "imst-ud",
        "UniversalDependencies/UD_Turkish-IMST",
        "CC-BY-NC-SA-3.0",
        "<pin>",
        None,
        False,
    ),
]
