"""Heuristic label correction for the silver tiers.

Rules (Turkish-morphotactics driven):

1. **POSS-on-Verb → Noun**
   Verbs cannot take possessive suffixes. If a token is labeled as Verb but
   has a ``+POSS.*`` tag, the POS must be Noun (or Adj via zero derivation).

2. **Adjective gazetteer + next-word-is-Noun → Adj**
   If the root is in the ADJ gazetteer AND the next sentence token is a Noun,
   relabel the POS as Adj. The gazetteer is built programmatically from the
   BOUN UD treebank (``UPOS == 'ADJ'``) — NOT a hand-curated list.

3. **GEN + POSS proximity**
   A ``+GEN`` token must be followed by a ``+POSS.*`` token within
   ``max_distance=3`` positions in the same sentence (Turkish allows
   intervening adjectives, e.g. ``arabanın kırmızı kapısı``). Violations are
   flagged but NOT auto-corrected (we just record a warning count).

**Hard invariants (enforced by assertion):**
- Gold-tier rows are NEVER modified. Attempting to do so raises.
- A mandatory read-only sanity check runs the same rules against gold labels;
  agreement with gold must exceed 98% or a warning is logged.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclass
class CorrectionReport:
    rule_counts: dict[str, int] = field(default_factory=dict)
    before: Counter = field(default_factory=Counter)
    after: Counter = field(default_factory=Counter)
    total_silver: int = 0
    total_corrected: int = 0


# ---------------------------------------------------------------------------
# Gazetteer
# ---------------------------------------------------------------------------

def build_adj_gazetteer(
    boun_dir: Path | str | None = None,
    gold_fallback: Path | str | None = None,
) -> set[str]:
    """Build the adjective gazetteer.

    Tries (in order): CoNLL-U files under ``boun_dir`` (column 4 ``UPOS ==
    'ADJ'`` → column 3 ``LEMMA``); then silver/gold JSONL under
    ``gold_fallback`` where the label contains ``+Adj``.
    """
    adjs: set[str] = set()
    if boun_dir is not None:
        p = Path(boun_dir)
        if p.exists():
            for conllu in p.rglob("*.conllu"):
                with conllu.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("#") or not line.strip():
                            continue
                        cols = line.rstrip("\n").split("\t")
                        if len(cols) >= 4 and cols[3] == "ADJ":
                            adjs.add(cols[2])
    if not adjs and gold_fallback is not None:
        gp = Path(gold_fallback)
        if gp.exists():
            with gp.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    label = rec.get("label", "")
                    if "+Adj" in label:
                        parts = label.split()
                        if parts:
                            adjs.add(parts[0])
    logger.info("Adjective gazetteer: %d entries", len(adjs))
    return adjs


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def _has_poss(label: str) -> bool:
    return any(t.startswith("+POSS") for t in label.split())


def _has_gen(label: str) -> bool:
    return "+GEN" in label.split()


def _is_pos(label: str, pos: str) -> bool:
    return f"+{pos}" in label.split()


def _replace_pos(label: str, old: str, new: str) -> str:
    parts = label.split()
    return " ".join(new if p == f"+{old}" else p for p in parts)


def apply_rules_to_record(
    rec: dict,
    next_rec: dict | None,
    adj_gazetteer: set[str],
    window: list[dict] | None = None,
    max_gen_poss_distance: int = 3,
) -> tuple[str, str | None]:
    """Apply rules to a single record.

    Returns: ``(new_label, rule_fired)``. ``rule_fired`` is ``None`` when no
    rule applied.
    """
    label = rec.get("label", "")
    if not label:
        return label, None

    # Rule 1: POSS-on-Verb → Noun
    if _is_pos(label, "Verb") and _has_poss(label):
        return _replace_pos(label, "Verb", "+Noun"), "rule1_poss_on_verb"

    # Rule 2: root in ADJ gazetteer + next is Noun → Adj
    if _is_pos(label, "Noun"):
        parts = label.split()
        root = parts[0] if parts else ""
        if root in adj_gazetteer and next_rec is not None:
            next_label = next_rec.get("label", "")
            if _is_pos(next_label, "Noun"):
                return _replace_pos(label, "Noun", "+Adj"), "rule2_adj_gazetteer"

    # Rule 3: GEN proximity — only flag (informational)
    if _has_gen(label) and window is not None:
        saw_poss = False
        for neighbour in window[:max_gen_poss_distance]:
            if _has_poss(neighbour.get("label", "")):
                saw_poss = True
                break
        if not saw_poss:
            return label, "rule3_gen_without_poss_flag"

    return label, None


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def _iter_sentences(path: Path) -> Iterable[list[dict]]:
    current_sid: str | None = None
    buf: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = rec.get("sentence_id")
            if sid != current_sid:
                if buf:
                    yield buf
                buf = []
                current_sid = sid
            buf.append(rec)
        if buf:
            yield buf


def apply_heuristic_corrections(
    corpus_path: Path | str,
    output_path: Path | str,
    adj_gazetteer: set[str],
    max_gen_poss_distance: int = 3,
) -> CorrectionReport:
    """Apply NOUN_ADJ_RULES to silver tiers. Gold is NEVER modified.

    Output: new JSONL at ``output_path`` with corrected labels for silver
    rows; gold rows pass through unchanged.
    """
    report = CorrectionReport()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out:
        for sentence in _iter_sentences(Path(corpus_path)):
            for i, rec in enumerate(sentence):
                tier = rec.get("tier", "")
                if tier == "gold":
                    # HARD INVARIANT: never mutate gold rows.
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    continue

                report.total_silver += 1
                report.before[rec.get("label", "")] += 1

                next_rec = sentence[i + 1] if i + 1 < len(sentence) else None
                window = sentence[i + 1: i + 1 + max_gen_poss_distance]

                new_label, rule = apply_rules_to_record(
                    rec, next_rec, adj_gazetteer, window, max_gen_poss_distance,
                )

                if rule is not None and rule != "rule3_gen_without_poss_flag" \
                        and new_label != rec.get("label", ""):
                    # IMPORTANT: defensive assertion — refuse to write a
                    # corrected gold row (tier shouldn't be "gold" here).
                    assert rec.get("tier", "") != "gold", \
                        "Gold-tier correction attempted — invariant violated"
                    corrected_rec = dict(rec)
                    corrected_rec["label"] = new_label
                    corrected_rec["_corrected_by"] = rule
                    report.total_corrected += 1
                    report.after[new_label] += 1
                    report.rule_counts[rule] = report.rule_counts.get(rule, 0) + 1
                    out.write(json.dumps(corrected_rec, ensure_ascii=False) + "\n")
                else:
                    if rule == "rule3_gen_without_poss_flag":
                        report.rule_counts[rule] = report.rule_counts.get(rule, 0) + 1
                    report.after[rec.get("label", "")] += 1
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return report


def gold_sanity_check(
    corpus_path: Path | str,
    adj_gazetteer: set[str],
) -> float:
    """Run the same rules against gold rows (read-only) and return agreement.

    Agreement = fraction of gold rows where the rule-suggested label equals
    the existing gold label. Must be >98% to validate the ruleset; warn if
    <95% (plan gate).
    """
    agree = 0
    total = 0
    for sentence in _iter_sentences(Path(corpus_path)):
        for i, rec in enumerate(sentence):
            if rec.get("tier") != "gold":
                continue
            total += 1
            next_rec = sentence[i + 1] if i + 1 < len(sentence) else None
            window = sentence[i + 1: i + 4]
            new_label, _ = apply_rules_to_record(
                rec, next_rec, adj_gazetteer, window,
            )
            if new_label == rec.get("label", ""):
                agree += 1
    if total == 0:
        return 1.0
    agreement = agree / total
    if agreement < 0.95:
        logger.warning(
            "Heuristic rules disagree with gold on %.1f%% of rows — "
            "rules are likely too aggressive. Agreement: %.2f%%",
            100 * (1 - agreement), 100 * agreement,
        )
    elif agreement < 0.98:
        logger.info(
            "Heuristic rule / gold agreement: %.2f%% (below 98%% target, "
            "above 95%% warning threshold)", 100 * agreement,
        )
    return agreement
