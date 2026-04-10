"""Download and save BERTurk model to models/berturk/.

Run once before training with BERTurkContext encoder:

    python scripts/download_berturk.py

Requires ~440 MB of disk space and internet access.
"""
from __future__ import annotations

import argparse
from pathlib import Path


MODEL_ID = "dbmdz/bert-base-turkish-cased"
DEFAULT_SAVE_PATH = "models/berturk"


def download(save_path: str = DEFAULT_SAVE_PATH) -> None:
    from transformers import AutoModel, AutoTokenizer  # type: ignore[import]

    out = Path(save_path)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading tokenizer for {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained(str(out))
    print(f"  Tokenizer saved to {out}/")

    print(f"Downloading model weights for {MODEL_ID} ...")
    model = AutoModel.from_pretrained(MODEL_ID)
    model.save_pretrained(str(out))
    print(f"  Model saved to {out}/")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nDone. BERTurk has {n_params / 1e6:.1f}M parameters.")
    print(f"Saved to: {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download BERTurk to local path")
    parser.add_argument(
        "--output", default=DEFAULT_SAVE_PATH,
        help=f"Directory to save model (default: {DEFAULT_SAVE_PATH})"
    )
    args = parser.parse_args()
    download(args.output)
