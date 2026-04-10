"""Command-line interface for morpho-tr.

Usage:
    morpho-tr analyze "evlerinden gidiyorum"
    morpho-tr analyze --file input.txt
"""

from __future__ import annotations


def analyze_command(text: str, backend: str = "zeyrek") -> None:
    """Analyze text and print morphological atoms."""
    from kokturk.core.analyzer import MorphoAnalyzer

    analyzer = MorphoAnalyzer(backends=[backend])
    for word in text.split():
        result = analyzer.analyze(word)
        best = result.best
        if best:
            print(f"{word:20s} → {best.to_str()}")
        else:
            print(f"{word:20s} → (no analysis)")
    analyzer.close()


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="morpho-tr",
        description="Turkish morphological atomization",
    )
    sub = parser.add_subparsers(dest="command")

    # analyze command
    p_analyze = sub.add_parser("analyze", help="Analyze Turkish text")
    p_analyze.add_argument("text", nargs="?", help="Text to analyze")
    p_analyze.add_argument("--file", type=str, help="Input file path")
    p_analyze.add_argument(
        "--backend", default="zeyrek",
        choices=["zeyrek", "neural"],
    )

    args = parser.parse_args()

    if args.command == "analyze":
        if args.file:
            with open(args.file, encoding="utf-8") as f:
                text = f.read()
        elif args.text:
            text = args.text
        else:
            parser.error("Provide text or --file")
            return
        analyze_command(text, args.backend)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
