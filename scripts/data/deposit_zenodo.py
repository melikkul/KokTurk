"""Deposit TR-Gold-Morph dataset to Zenodo and retrieve DOI.

Usage:
    ZENODO_TOKEN=<token> python scripts/data/deposit_zenodo.py \\
        --files data/gold/tr_gold_morph_v1.jsonl data/gold/tr_gold_morph_v1.tsv \\
        --sandbox   # use sandbox.zenodo.org for testing

After minting the DOI, add it to CITATION.cff and re-render README.md.

Pre-condition: set ZENODO_TOKEN env var (from https://zenodo.org/account/settings/applications/).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
ZENODO_BASE = "https://zenodo.org/api"
ZENODO_SANDBOX_BASE = "https://sandbox.zenodo.org/api"

METADATA = {
    "metadata": {
        "title": "TR-Gold-Morph v1: Turkish Morphological Analysis Gold Corpus",
        "upload_type": "dataset",
        "description": (
            "TR-Gold-Morph v1 — 80,537 Turkish tokens with gold/silver/bronze morphological "
            "analysis annotations (root, POS tags, suffix sequence). Part of the Aksu project: "
            "neural morphological atomization for Turkish."
        ),
        "creators": [{"name": "Kul, Melik", "affiliation": "Ostim Technical University"}],
        "license": "cc-by-4.0",
        "keywords": ["Turkish", "morphology", "NLP", "corpus", "annotation"],
        "language": "tur",
        "access_right": "open",
        "related_identifiers": [
            {
                "identifier": "https://github.com/melikkul/Aksu",
                "relation": "isSupplementTo",
                "resource_type": "software",
            }
        ],
    }
}


def _requests():
    try:
        import requests
        return requests
    except ImportError:
        print("ERROR: requests not installed. Run: pip install requests", file=sys.stderr)
        sys.exit(1)


def deposit(files: list[Path], sandbox: bool = False, token: str | None = None) -> dict:
    requests = _requests()
    token = token or os.environ.get("ZENODO_TOKEN")
    if not token:
        print("ERROR: ZENODO_TOKEN not set.", file=sys.stderr)
        sys.exit(1)

    base = ZENODO_SANDBOX_BASE if sandbox else ZENODO_BASE
    headers = {"Authorization": f"Bearer {token}"}
    env_label = "sandbox" if sandbox else "production"
    print(f"Depositing to Zenodo {env_label} ...")

    # Create empty deposit
    r = requests.post(f"{base}/deposit/depositions", json={}, headers=headers)
    r.raise_for_status()
    deposition = r.json()
    dep_id = deposition["id"]
    bucket_url = deposition["links"]["bucket"]
    print(f"  Created deposition ID {dep_id}")

    # Upload files
    for f in files:
        if not f.exists():
            print(f"  SKIP {f} — not found", file=sys.stderr)
            continue
        with f.open("rb") as fh:
            r2 = requests.put(f"{bucket_url}/{f.name}", data=fh, headers=headers)
            r2.raise_for_status()
            print(f"  Uploaded {f.name}")

    # Set metadata
    r3 = requests.put(
        f"{base}/deposit/depositions/{dep_id}",
        json=METADATA,
        headers={**headers, "Content-Type": "application/json"},
    )
    r3.raise_for_status()

    # Publish (in sandbox) — omit for production (requires DOI reservation first)
    if sandbox:
        r4 = requests.post(f"{base}/deposit/depositions/{dep_id}/actions/publish", headers=headers)
        r4.raise_for_status()
        result = r4.json()
        doi = result.get("doi", result.get("prereserve_doi", {}).get("doi", "pending"))
        print(f"  Published. DOI: {doi}")
    else:
        print(f"  Metadata set. Submit manually at: {base.replace('/api', '')}/deposit/{dep_id}")
        doi = "pending-manual-publish"

    return {"deposition_id": dep_id, "doi": doi, "sandbox": sandbox}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--files", nargs="+", default=[], help="Files to upload")
    ap.add_argument("--sandbox", action="store_true", help="Use Zenodo sandbox")
    ap.add_argument("--token", default=None, help="Zenodo API token (default: $ZENODO_TOKEN)")
    args = ap.parse_args()

    files = [Path(f) for f in args.files] if args.files else []
    if not files:
        # Default to v1 dataset files
        files = [
            ROOT / "data/gold/tr_gold_morph_v1.jsonl",
            ROOT / "data/gold/tr_gold_morph_v1.tsv",
            ROOT / "data/gold/tr_gold_morph_v1.conllu",
        ]

    result = deposit(files, sandbox=args.sandbox, token=args.token)
    out = ROOT / "audit/benchmark_results/zenodo_deposit.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Deposit info written to {out}")
    if result["doi"] not in ("pending-manual-publish", "pending"):
        print(f"\nNext: add doi: {result['doi']} to CITATION.cff and re-run scripts/build_readme.py")


if __name__ == "__main__":
    main()
