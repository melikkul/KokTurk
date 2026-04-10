"""Turkish-specific robustness perturbation suite.

Applies deterministic noise recipes (deasciification, casing,
elongation, code-switch, hashtag fusion) to a clean test set and
measures accuracy drop per attack.

Casing uses :func:`ariturk.normalize.turkish_lower` /
:func:`ariturk.normalize.turkish_upper` so that Turkish ``i``/``İ``
and ``ı``/``I`` are handled correctly. We never fall back to Python's
``str.lower()`` — that is the exact bug the CasingAttack regression
test exists to catch.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from ariturk.normalize import turkish_lower, turkish_upper

_VOWELS = set("aeıioöuüAEIİOÖUÜ")


@dataclass(frozen=True, slots=True)
class AttackResult:
    name: str
    clean_em: float
    attacked_em: float
    delta: float


@dataclass
class RobustnessReport:
    results: list[AttackResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Attacks
# ---------------------------------------------------------------------------


_DEASCII_MAP = {"ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u",
                "Ç": "C", "Ğ": "G", "İ": "I", "Ö": "O", "Ş": "S", "Ü": "U"}


class DeasciificationAttack:
    def __init__(self, prob: float = 0.4, seed: int = 0) -> None:
        self.prob = prob
        self._rng = random.Random(seed)

    def perturb(self, text: str) -> str:
        out = []
        for ch in text:
            if ch in _DEASCII_MAP and self._rng.random() < self.prob:
                out.append(_DEASCII_MAP[ch])
            else:
                out.append(ch)
        return "".join(out)


class CasingAttack:
    """Apply Turkish-locale-correct casing transforms."""

    def perturb(self, text: str, mode: str = "all_caps") -> str:
        if mode == "all_caps":
            return turkish_upper(text)
        if mode == "all_lower":
            return turkish_lower(text)
        if mode == "random_mixed":
            rng = random.Random(hash(text) & 0xFFFF)
            out = []
            for ch in text:
                out.append(turkish_upper(ch) if rng.random() < 0.5 else turkish_lower(ch))
            return "".join(out)
        raise ValueError(f"unknown casing mode: {mode}")


class ElongationAttack:
    def perturb(self, text: str, vowel_repeat: int = 3) -> str:
        out = []
        prev_vowel = False
        for ch in text:
            if ch in _VOWELS and not prev_vowel:
                out.append(ch * vowel_repeat)
                prev_vowel = True
            else:
                out.append(ch)
                prev_vowel = ch in _VOWELS
        return "".join(out) + "!!!"


class CodeSwitchAttack:
    SWAP_DICT = {
        "beğen": "like'la", "ara": "search'le", "paylaş": "share'la",
        "takip": "follow'la", "indir": "download'la", "yükle": "upload'la",
        "gönder": "send'le", "al": "buy'la", "sat": "sell'le",
        "izle": "watch'le", "dinle": "listen'la", "oku": "read'le",
        "yaz": "write'la", "sev": "love'la", "seç": "pick'le",
        "başla": "start'la", "bitir": "finish'le", "çalış": "work'la",
        "kontrol": "check'le", "kullan": "use'la",
    }

    def perturb(self, text: str) -> str:
        out = text
        for tr, en in self.SWAP_DICT.items():
            out = re.sub(rf"\b{re.escape(tr)}", en, out)
        return out


class HashtagAttack:
    def perturb(self, words: list[str]) -> str:
        return "#" + "".join(w.capitalize() for w in words)


# ---------------------------------------------------------------------------
# Suite runner
# ---------------------------------------------------------------------------


def run_robustness_suite(
    score_fn,
    clean_texts: list[str],
    clean_labels: list[str],
    output_path: str | Path | None = None,
) -> RobustnessReport:
    """Run every attack and record EM deltas.

    Args:
        score_fn: callable ``(texts, labels) -> em_float``.
        clean_texts: surfaces to feed through each attack.
        clean_labels: gold labels matched to ``clean_texts`` (unchanged).
        output_path: optional markdown report destination.
    """
    clean_em = score_fn(clean_texts, clean_labels)
    report = RobustnessReport()
    attacks: list[tuple[str, callable]] = [
        ("deasciification", lambda ts: [DeasciificationAttack(seed=0).perturb(t) for t in ts]),
        ("casing_upper", lambda ts: [CasingAttack().perturb(t, "all_caps") for t in ts]),
        ("casing_lower", lambda ts: [CasingAttack().perturb(t, "all_lower") for t in ts]),
        ("elongation", lambda ts: [ElongationAttack().perturb(t) for t in ts]),
        ("code_switch", lambda ts: [CodeSwitchAttack().perturb(t) for t in ts]),
    ]
    for name, fn in attacks:
        attacked = fn(clean_texts)
        em = score_fn(attacked, clean_labels)
        report.results.append(
            AttackResult(name=name, clean_em=clean_em, attacked_em=em, delta=em - clean_em)
        )
    if output_path is not None:
        _write_report(report, Path(output_path))
    return report


def _write_report(report: RobustnessReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Robustness Report\n", "| Attack | Clean EM | Attacked EM | Delta |", "|---|---|---|---|"]
    for r in report.results:
        lines.append(f"| {r.name} | {r.clean_em:.4f} | {r.attacked_em:.4f} | {r.delta:+.4f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
