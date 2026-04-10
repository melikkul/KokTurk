"""Hyperparameter sensitivity analysis for TAAC vs fixed curriculum.

Compares the HP sensitivity of TAAC (2 key HPs: epsilon, patience) against
fixed curriculum (8+ HPs: 4 phase boundaries + 5 LR values).  If TAAC's
small HP space explains less EM variance, TAAC is more robust.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def generate_taac_configs(n_samples: int = 50, seed: int = 42) -> list[dict]:
    """Generate random TAAC hyperparameter configurations."""
    rng = np.random.RandomState(seed)
    configs: list[dict] = []
    for _ in range(n_samples):
        configs.append({
            "curriculum": "taac",
            "epsilon": float(rng.choice([0.005, 0.01, 0.02, 0.05])),
            "patience": int(rng.choice([1, 2, 3, 5])),
        })
    return configs


def generate_fixed_configs(n_samples: int = 50, seed: int = 42) -> list[dict]:
    """Generate random fixed curriculum configurations."""
    rng = np.random.RandomState(seed)
    configs: list[dict] = []
    for _ in range(n_samples):
        boundaries = sorted(rng.choice(range(2, 40), size=3, replace=False))
        configs.append({
            "curriculum": "fixed",
            "phase1_end": int(boundaries[0]),
            "phase2_end": int(boundaries[1]),
            "phase3_end": int(boundaries[2]),
            "lr_phase1": float(rng.choice([1e-4, 3e-4, 5e-4, 1e-3])),
            "lr_phase2": float(rng.choice([5e-5, 1e-4, 3e-4, 5e-4])),
            "lr_phase3": float(rng.choice([1e-5, 5e-5, 1e-4])),
            "lr_phase4": float(rng.choice([5e-6, 1e-5, 5e-5])),
            "lr_phase5": float(rng.choice([1e-6, 5e-6, 1e-5])),
        })
    return configs


def analyze_sensitivity(results_dir: str) -> dict[str, dict[str, float]]:
    """Load result JSONs and compute per-curriculum EM variance.

    Each JSON file must contain ``{"config": {...}, "final_em": float}``.

    Returns:
        ``{"taac": {"n": ..., "mean_em": ..., "std_em": ..., "spread": ...},
           "fixed": {...}}``
    """
    results: list[dict] = []
    for f in Path(results_dir).glob("*.json"):
        results.append(json.loads(f.read_text()))

    summary: dict[str, dict[str, float]] = {}
    for curriculum_type in ("taac", "fixed"):
        ems = [
            r["final_em"]
            for r in results
            if r.get("config", {}).get("curriculum") == curriculum_type
        ]
        if ems:
            summary[curriculum_type] = {
                "n": len(ems),
                "mean_em": float(np.mean(ems)),
                "std_em": float(np.std(ems)),
                "spread": float(max(ems) - min(ems)),
            }
    return summary
