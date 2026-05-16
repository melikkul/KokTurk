"""Publish TR-Gold-Morph dataset and Aksu model checkpoints to HuggingFace Hub.

Usage:
    huggingface-cli login
    python scripts/data/publish_huggingface.py --dataset-repo melikkul/tr-gold-morph
    python scripts/data/publish_huggingface.py --model-repo melikkul/aksu-disambiguator-v6
    python scripts/data/publish_huggingface.py --model-repo melikkul/aksu-dualhead-v2

Requirements:
    pip install huggingface_hub datasets
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def _hf():
    try:
        from huggingface_hub import HfApi, create_repo
        return HfApi, create_repo
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)


def publish_dataset(repo_id: str, data_dir: Path, private: bool = False) -> None:
    HfApi, create_repo = _hf()
    api = HfApi()

    print(f"Creating/verifying dataset repo {repo_id} ...")
    create_repo(repo_id, repo_type="dataset", exist_ok=True, private=private)

    gold_dir = ROOT / "data/gold"
    slices = list(gold_dir.glob("*.jsonl")) + list(gold_dir.glob("*.tsv")) + list(gold_dir.glob("*.conllu"))
    if not slices:
        print(f"No dataset slices found under {gold_dir}. Build the dataset first.", file=sys.stderr)
        sys.exit(1)

    dataset_card = _build_dataset_card(repo_id)
    card_path = ROOT / "data/gold/README.md"
    card_path.write_text(dataset_card, encoding="utf-8")

    print(f"Uploading {len(slices)} files to {repo_id} ...")
    for f in slices:
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=repo_id,
            repo_type="dataset",
        )
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Dataset published to https://huggingface.co/datasets/{repo_id}")


def publish_model(repo_id: str, model_dir: Path, private: bool = False) -> None:
    HfApi, create_repo = _hf()
    api = HfApi()

    if not model_dir.exists():
        print(f"ERROR: model dir {model_dir} does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Creating/verifying model repo {repo_id} ...")
    create_repo(repo_id, repo_type="model", exist_ok=True, private=private)

    checkpoints = list(model_dir.glob("*.pt"))
    sidecars = list(model_dir.glob("*_metadata.json")) + list(model_dir.glob("*.json"))
    files = checkpoints + sidecars

    model_card = _build_model_card(repo_id, model_dir)
    card_path = model_dir / "README.md"
    card_path.write_text(model_card, encoding="utf-8")
    files.append(card_path)

    print(f"Uploading {len(files)} files to {repo_id} ...")
    for f in files:
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=repo_id,
            repo_type="model",
        )
    print(f"Model published to https://huggingface.co/models/{repo_id}")


def _build_dataset_card(repo_id: str) -> str:
    manifest_path = ROOT / "data/external/manifest.json"
    manifest_note = ""
    if manifest_path.exists():
        manifest_note = f"\nSource corpus manifest: `data/external/manifest.json`\n"

    return f"""---
license: cc-by-4.0
language:
  - tr
tags:
  - morphology
  - turkish
  - nlp
  - annotated
pretty_name: TR-Gold-Morph
---

# TR-Gold-Morph

Largest Turkish morphological analysis dataset with gold/silver/bronze tiers.

- **v1**: 80,537 tokens, manually validated, 3 export formats
- **v2**: 2.5M token target (pipeline ready; harvest in progress)

## License

CC BY 4.0. BOUN/Wikipedia-derived shards carry CC BY-SA.
IMST-UD (CC-BY-NC-SA) is excluded from this dataset.
{manifest_note}
## Citation

```bibtex
@thesis{{kul2026aksu,
  title={{Aksu: Neural Morphological Atomization for Turkish}},
  author={{Kul, Melik}},
  year={{2026}},
  school={{Ostim Technical University}},
}}
```
"""


def _build_model_card(repo_id: str, model_dir: Path) -> str:
    sidecar = model_dir / "best_model_metadata.json"
    meta_note = ""
    if sidecar.exists():
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        meta_note = f"\n- Training commit: `{meta.get('training_commit_sha', 'backfilled')}`\n"
        meta_note += f"- Dataset sha256: `{meta.get('data_sha256', 'pending')}`\n"

    return f"""---
language:
  - tr
tags:
  - morphology
  - turkish
  - nlp
license: mit
---

# {repo_id.split('/')[-1]}

Aksu Turkish morphological disambiguator / generator checkpoint.
See the [Aksu GitHub repo](https://github.com/melikkul/Aksu) for usage.
{meta_note}
## Usage

```python
from aksu import MorphoAnalyzer
analyzer = MorphoAnalyzer(backends=["disambiguator"])
results = analyzer.analyze_sentence("Çocuklar evlerinden çıktı")
```

## License

MIT
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-repo", default=None, help="HF dataset repo ID (e.g. melikkul/tr-gold-morph)")
    ap.add_argument("--model-repo", default=None, help="HF model repo ID (e.g. melikkul/aksu-disambiguator-v6)")
    ap.add_argument("--model-dir", default=None, help="Local model directory (default: models/<repo-name>/)")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    if not args.dataset_repo and not args.model_repo:
        ap.error("Supply at least one of --dataset-repo or --model-repo")

    if args.dataset_repo:
        publish_dataset(args.dataset_repo, ROOT / "data/gold", private=args.private)

    if args.model_repo:
        model_name = args.model_repo.split("/")[-1]
        model_dir = Path(args.model_dir) if args.model_dir else ROOT / "models" / model_name
        publish_model(args.model_repo, model_dir, private=args.private)


if __name__ == "__main__":
    main()
