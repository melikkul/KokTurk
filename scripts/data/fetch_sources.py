"""Download and pin all source corpora listed in sources.py.

Updates data/external/manifest.json with sha256 and actual version pins.

Usage:
    python scripts/data/fetch_sources.py --sources oscar-tr boun-ud
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_boun(output_dir: Path) -> str:
    """Clone or update BOUN treebank; return pinned HEAD SHA."""
    repo_dir = output_dir / "boun_treebank"
    url = "https://github.com/UniversalDependencies/UD_Turkish-BOUN.git"
    if repo_dir.exists():
        subprocess.run(["git", "-C", str(repo_dir), "pull"], check=True)
    else:
        subprocess.run(["git", "clone", "--depth=1", url, str(repo_dir)], check=True)
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["boun-ud"])
    ap.add_argument("--output-dir", default="data/external")
    args = ap.parse_args()

    from aksu.data.build.sources import SOURCES
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.json"
    manifest: dict = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    for src in SOURCES:
        if src.name not in args.sources:
            continue
        logger.info("Fetching %s ...", src.name)

        if src.name == "boun-ud":
            sha = fetch_boun(output_dir)
            manifest[src.name] = {
                "name": src.name,
                "url": src.url,
                "license": src.license,
                "version_pin": sha,
                "redistribute": src.redistribute,
            }
            logger.info("  BOUN pinned at %s", sha)
        else:
            logger.info(
                "  %s: run 'python -m aksu.data.build.preprocess --shard %s' to download",
                src.name, src.name,
            )
            manifest[src.name] = {
                "name": src.name,
                "url": src.url,
                "license": src.license,
                "version_pin": src.version,
                "sha256": src.sha256,
                "redistribute": src.redistribute,
                "status": "pending_download",
            }

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Manifest saved to %s", manifest_path)


if __name__ == "__main__":
    main()
