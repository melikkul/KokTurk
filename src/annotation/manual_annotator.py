"""Terminal-based manual annotation tool for morphological parsing.

Displays candidate parses for each token and accepts annotator input.
Saves progress after every sentence. Can resume from where left off.

Usage:
    python src/annotation/manual_annotator.py [--input data/gold/seed/seed_200.jsonl]
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

DEFAULT_INPUT = Path("data/gold/seed/seed_200.jsonl")


def load_data(path: Path) -> list[dict[str, object]]:
    """Load annotation data from JSONL."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_data(data: list[dict[str, object]], path: Path) -> None:
    """Save annotation data, creating a backup first."""
    backup = path.with_suffix(".jsonl.bak")
    if path.exists():
        shutil.copy2(path, backup)
    with open(path, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def find_resume_point(data: list[dict[str, object]]) -> tuple[int, int]:
    """Find the first unannotated token to resume from.

    Returns:
        (sentence_idx, token_idx) of the first unannotated token.
    """
    for si, sent in enumerate(data):
        tokens = sent.get("tokens", [])
        for ti, tok in enumerate(tokens):  # type: ignore[union-attr]
            if tok.get("gold_label") is None:  # type: ignore[union-attr]
                return si, ti
    return len(data), 0  # All done


def format_context(
    sentence: dict[str, object], token_idx: int, context_window: int = 3,
) -> str:
    """Format sentence with the current token highlighted."""
    tokens = sentence.get("tokens", [])
    surfaces = [t["surface"] for t in tokens]  # type: ignore[union-attr]
    parts: list[str] = []
    for i, s in enumerate(surfaces):
        if i == token_idx:
            parts.append(f"[{s}]")
        else:
            parts.append(s)
    return " ".join(parts)


def annotate_token(
    sentence: dict[str, object],
    token_idx: int,
    sent_num: int,
    total_sents: int,
) -> str | None:
    """Display a token and get annotator choice.

    Returns:
        The chosen label string, "SKIP", or None to quit.
    """
    tokens = sentence.get("tokens", [])
    token = tokens[token_idx]  # type: ignore[index]
    surface = token["surface"]  # type: ignore[index]
    candidates = token.get("candidate_parses", [])  # type: ignore[union-attr]

    # Header
    print(f"\n{'━'*60}")
    print(f"  Sentence {sent_num}/{total_sents}  |  "
          f"Token {token_idx + 1}/{len(tokens)}")  # type: ignore[arg-type]
    print(f"{'━'*60}")
    print(f"  Context: {format_context(sentence, token_idx)}")
    print()

    if not candidates:
        print(f"  No candidate parses for \"{surface}\"")
        print("  [m] Type manually  [s] Skip  [q] Save & quit")
    else:
        print(f"  Candidate parses for \"{surface}\":")
        for i, c in enumerate(candidates):  # type: ignore[union-attr]
            tags = " ".join(c.get("tags", []))  # type: ignore[union-attr]
            conf = c.get("confidence", 0.0)  # type: ignore[union-attr]
            src = c.get("source", "?")  # type: ignore[union-attr]
            marker = " ← recommended" if i == 0 else ""
            label = f"{c['root']} {tags}".strip()  # type: ignore[index]
            print(f"    [{i + 1}] {label:40s} "
                  f"(conf={conf:.2f}, src={src}){marker}")
        print("    [m] Type manually  [s] Skip  [q] Save & quit")

    print()
    choice = input("  Your choice: ").strip().lower()

    if choice == "q":
        return None
    if choice == "s":
        return "SKIP"
    if choice == "m":
        manual = input("  Type label (root +TAG1 +TAG2 ...): ").strip()
        return manual if manual else "SKIP"

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(candidates):  # type: ignore[arg-type]
            c = candidates[idx]  # type: ignore[index]
            tags = " ".join(c.get("tags", []))  # type: ignore[union-attr]
            return f"{c['root']} {tags}".strip()  # type: ignore[index]
    except ValueError:
        pass

    print("  Invalid choice, skipping.")
    return "SKIP"


def run_annotator(input_path: Path = DEFAULT_INPUT) -> None:
    """Main annotation loop."""
    if not input_path.exists():
        print(f"Error: {input_path} not found. Run create_seed.py first.")
        sys.exit(1)

    data = load_data(input_path)
    total_sents = len(data)

    start_sent, start_tok = find_resume_point(data)
    if start_sent >= total_sents:
        print("All tokens already annotated!")
        return

    # Count progress
    total_tokens = sum(
        len(s.get("tokens", []))  # type: ignore[arg-type]
        for s in data
    )
    annotated = sum(
        1 for s in data
        for t in s.get("tokens", [])  # type: ignore[union-attr]
        if t.get("gold_label") is not None  # type: ignore[union-attr]
    )

    remaining = total_tokens - annotated

    print(f"\n{'='*60}")
    print("  MORPHOLOGICAL ANNOTATION TOOL")
    print(f"{'='*60}")
    print(f"  File: {input_path}")
    print(f"  Completed:  {annotated}/{total_tokens} tokens "
          f"({annotated * 100 // max(total_tokens, 1)}% done)")
    print(f"  Remaining:  {remaining} tokens to annotate")
    print("  Auto-skip:  completed tokens are skipped automatically")
    print("  Press 'q' at any time to save and quit")
    print(f"{'='*60}")

    session_start = time.time()
    session_count = 0

    for si in range(start_sent, total_sents):
        tokens = data[si].get("tokens", [])
        tok_start = start_tok if si == start_sent else 0

        for ti in range(tok_start, len(tokens)):  # type: ignore[arg-type]
            token = tokens[ti]  # type: ignore[index]
            if token.get("gold_label") is not None:  # type: ignore[union-attr]
                continue

            result = annotate_token(data[si], ti, si + 1, total_sents)

            if result is None:
                # Quit
                save_data(data, input_path)
                elapsed = time.time() - session_start
                rate = session_count / (elapsed / 60) if elapsed > 0 else 0
                print(f"\n  Saved. Annotated {session_count} tokens "
                      f"this session ({rate:.0f} tokens/min)")
                return

            if result != "SKIP":
                token["gold_label"] = result  # type: ignore[index]
                session_count += 1

        # Save after every sentence
        save_data(data, input_path)

    elapsed = time.time() - session_start
    rate = session_count / (elapsed / 60) if elapsed > 0 else 0
    print(f"\n  All done! Annotated {session_count} tokens "
          f"this session ({rate:.0f} tokens/min)")
    save_data(data, input_path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Terminal annotation tool for morphological parsing"
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help="Path to seed/batch JSONL file",
    )
    parser.add_argument(
        "--skip-completed", action="store_true",
        help="Skip tokens that already have a gold_label (e.g., auto-accepted).",
    )
    args = parser.parse_args()
    run_annotator(args.input)


if __name__ == "__main__":
    main()
