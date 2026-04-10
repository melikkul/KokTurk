"""Batch annotation of remaining unlabeled tokens using linguistic heuristics.

Uses BOUN Treebank gold POS tags and sentence context to disambiguate
among Zeyrek candidate parses. This is acceptable for the seed dataset
since the active learning loop will catch and correct errors.

Usage:
    PYTHONPATH=src python src/annotation/batch_annotate.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SEED_PATH = Path("data/gold/seed/seed_200_partial.jsonl")
CORPUS_PATH = Path("data/processed/corpus.jsonl")
OUTPUT_PATH = Path("data/gold/seed/seed_200_annotated.jsonl")

# Map BOUN UPOS tags to our canonical POS tags used in Zeyrek output
UPOS_TO_ZEYREK_POS: dict[str, list[str]] = {
    "NOUN": ["+Noun"],
    "VERB": ["+Verb"],
    "ADJ": ["+Adj"],
    "ADV": ["+Adv"],
    "DET": ["+Det"],
    "PRON": ["+Pron"],
    "ADP": ["+Postp", "+Adv"],  # ADP often maps to postpositions or adverbs
    "CCONJ": ["+Conj"],
    "SCONJ": ["+Conj"],
    "INTJ": ["+Interj"],
    "NUM": ["+Num"],
    "PROPN": ["+Prop", "+Noun"],
    "PUNCT": ["+Punc"],
    "AUX": ["+Verb", "+Noun", "+Adj"],  # AUX in Turkish can be verb or adj (değil)
    "PART": ["+Conj", "+Adv", "+Noun"],  # particles like "de/da"
    "X": [],  # foreign words — fallback
}


def _parse_to_str(parse: dict[str, object]) -> str:
    """Convert a candidate parse dict to label string."""
    tags = " ".join(parse.get("tags", []))  # type: ignore[union-attr]
    return f"{parse['root']} {tags}".strip()


def _parse_has_pos(parse: dict[str, object], pos_tags: list[str]) -> bool:
    """Check if a parse contains any of the given POS tags."""
    tags: list[str] = parse.get("tags", [])  # type: ignore[assignment]
    return any(pos in tags for pos in pos_tags)


def _select_parse(
    token: dict[str, object],
    token_idx: int,
    all_tokens: list[dict[str, object]],
    pos_tags: list[str],
) -> tuple[str, str]:
    """Select the best parse for an unlabeled token.

    Args:
        token: Token dict with candidate_parses.
        token_idx: Position in sentence.
        all_tokens: All tokens in the sentence.
        pos_tags: BOUN gold POS tags for the sentence.

    Returns:
        (selected_label, decision_reason) tuple.
    """
    parses: list[dict[str, object]] = token.get(
        "candidate_parses", []
    )  # type: ignore[assignment]

    if not parses:
        return str(token.get("surface", "")), "no_candidates"

    if len(parses) == 1:
        return _parse_to_str(parses[0]), "single_candidate"

    surface = str(token.get("surface", ""))
    upos = pos_tags[token_idx] if token_idx < len(pos_tags) else ""

    # Rule 1: Use BOUN gold POS to filter candidates
    if upos and upos in UPOS_TO_ZEYREK_POS:
        target_pos = UPOS_TO_ZEYREK_POS[upos]
        if target_pos:
            pos_matches = [
                p for p in parses if _parse_has_pos(p, target_pos)
            ]
            if len(pos_matches) == 1:
                return _parse_to_str(pos_matches[0]), f"pos_unique:{upos}"
            if pos_matches:
                # Multiple POS matches — apply sub-rules
                parses = pos_matches

    # Rule 2: Proper noun — uppercase initial + has +Prop parse
    if surface and surface[0].isupper() and upos == "PROPN":
        prop_parses = [p for p in parses if _parse_has_pos(p, ["+Prop"])]
        if prop_parses:
            return _parse_to_str(prop_parses[0]), "proper_noun"

    # Rule 3: Context-based case disambiguation
    # After a verb, prefer accusative/dative objects
    if token_idx > 0 and token_idx < len(pos_tags):
        prev_pos = pos_tags[token_idx - 1] if token_idx > 0 else ""
        next_pos = pos_tags[token_idx + 1] if token_idx + 1 < len(pos_tags) else ""

        # Noun after DET → prefer nominative or accusative noun
        if prev_pos == "DET":
            noun_parses = [
                p for p in parses if _parse_has_pos(p, ["+Noun"])
            ]
            if noun_parses:
                return _parse_to_str(noun_parses[0]), "det_noun"

        # Adjective before noun → prefer adjective parse
        if next_pos == "NOUN" and upos == "ADJ":
            adj_parses = [
                p for p in parses if _parse_has_pos(p, ["+Adj"])
            ]
            if adj_parses:
                return _parse_to_str(adj_parses[0]), "adj_before_noun"

    # Rule 4: POSS.3SG vs ACC disambiguation
    # "eksikliği" — if followed by a postposition like "yüzünden",
    # it's possessive (POSS.3SG), not accusative
    if any("+ACC" in str(p.get("tags", [])) for p in parses):
        acc_parses = [
            p for p in parses
            if "+ACC" in p.get("tags", [])  # type: ignore[operator]
        ]
        poss_parses = [
            p for p in parses
            if "+POSS.3SG" in p.get("tags", [])  # type: ignore[operator]
        ]
        if acc_parses and poss_parses and token_idx + 1 < len(pos_tags):
            next_p = pos_tags[token_idx + 1]
            # Before ADP/VERB → possessive; before PUNCT/end → accusative
            if next_p in ("ADP", "NOUN"):
                return _parse_to_str(poss_parses[0]), "poss_before_adp"
            if next_p in ("PUNCT", "VERB"):
                return _parse_to_str(acc_parses[0]), "acc_before_verb"

    # Rule 5: 2SG possession (evin/okulun) vs genitive
    # Genitive is far more common in written text
    gen_parses = [
        p for p in parses
        if "+GEN" in p.get("tags", [])  # type: ignore[operator]
    ]
    poss2sg_parses = [
        p for p in parses
        if "+POSS.2SG" in p.get("tags", [])  # type: ignore[operator]
    ]
    if gen_parses and poss2sg_parses:
        return _parse_to_str(gen_parses[0]), "gen_over_poss2sg"

    # Rule 6: Prefer Noun+PLU+POSS.3PL over Noun+A3sg+POSS.3PL
    # (the plural marker reading is more common in written text)
    plu_parses = [
        p for p in parses
        if "+PLU" in p.get("tags", [])  # type: ignore[operator]
    ]
    if plu_parses and len(plu_parses) < len(parses):
        return _parse_to_str(plu_parses[0]), "prefer_plu"

    # Fallback: highest confidence candidate
    return _parse_to_str(parses[0]), "confidence_fallback"


def batch_annotate(
    seed_path: Path = SEED_PATH,
    corpus_path: Path = CORPUS_PATH,
    output_path: Path = OUTPUT_PATH,
) -> dict[str, int]:
    """Annotate all remaining unlabeled tokens programmatically.

    Returns:
        Dict with annotation statistics.
    """
    # Load corpus for POS context
    corpus: dict[str, dict[str, object]] = {}
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            corpus[str(s["sentence_id"])] = s

    # Load seed
    data: list[dict[str, object]] = []
    with open(seed_path, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    stats: dict[str, int] = {
        "total_tokens": 0,
        "already_labeled": 0,
        "newly_annotated": 0,
    }
    decision_counts: dict[str, int] = {}

    for sent in data:
        sid = str(sent["sentence_id"])
        csent = corpus.get(sid, {})
        pos_tags: list[str] = csent.get("pos_tags", [])  # type: ignore[assignment]
        tokens: list[dict[str, object]] = sent["tokens"]  # type: ignore[assignment]

        for i, token in enumerate(tokens):
            stats["total_tokens"] += 1

            if token.get("gold_label") is not None:
                stats["already_labeled"] += 1
                continue

            label, reason = _select_parse(token, i, tokens, pos_tags)
            token["gold_label"] = label
            stats["newly_annotated"] += 1
            decision_counts[reason] = decision_counts.get(reason, 0) + 1

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sent in data:
            f.write(json.dumps(sent, ensure_ascii=False) + "\n")

    return {**stats, **{f"decision_{k}": v for k, v in decision_counts.items()}}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s",
    )

    stats = batch_annotate()

    print(f"\n{'='*60}")
    print("BATCH ANNOTATION RESULTS")
    print(f"{'='*60}")
    print(f"Total tokens:       {stats['total_tokens']}")
    print(f"Already labeled:    {stats['already_labeled']}")
    print(f"Newly annotated:    {stats['newly_annotated']}")
    print("\nDecision breakdown:")
    for k, v in sorted(stats.items()):
        if k.startswith("decision_"):
            reason = k.replace("decision_", "")
            print(f"  {reason:25s} {v:5d}")
    fallback = stats.get("decision_confidence_fallback", 0)
    total_new = stats["newly_annotated"]
    heuristic = total_new - fallback
    print(f"\nHeuristic decisions:  {heuristic} ({heuristic/max(total_new,1):.1%})")
    print(f"Confidence fallback:  {fallback} ({fallback/max(total_new,1):.1%})")
    print(f"{'='*60}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
