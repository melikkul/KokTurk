"""Reproducibility utilities for deterministic training.

Ensures identical results across runs given the same seed, hardware, and code.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Pin all random number generators for reproducibility.

    Sets: Python random, numpy, torch CPU, torch CUDA (all devices),
    cuDNN deterministic mode, PYTHONHASHSEED.

    Note:
        cudnn.deterministic=True may slow training by ~10%.

    Args:
        seed: Random seed to use across all generators.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker initializer for deterministic multi-process loading.

    Each worker gets a unique but reproducible seed derived from
    torch.initial_seed() (which incorporates the global seed + epoch).
    Without this, workers with fork() inherit identical PRNG states.

    Args:
        worker_id: Worker index assigned by DataLoader.

    Usage::

        loader = DataLoader(dataset, num_workers=4,
                           worker_init_fn=worker_init_fn,
                           generator=torch.Generator().manual_seed(42))
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def capture_environment() -> dict[str, Any]:
    """Capture full execution environment for reproducibility logging.

    Returns:
        Dict with keys: git_commit, git_dirty, python_version,
        torch_version, cuda_version, slurm_job_id, slurm_node, pip_freeze.
    """
    env: dict[str, Any] = {}

    # Git state
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        env["git_commit"] = result.stdout.strip() if result.returncode == 0 else "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        env["git_commit"] = "unknown"

    try:
        result = subprocess.run(
            ["git", "diff", "--quiet"],
            capture_output=True, timeout=5,
        )
        env["git_dirty"] = result.returncode != 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        env["git_dirty"] = False

    # Python / PyTorch
    env["python_version"] = sys.version
    env["torch_version"] = torch.__version__
    env["cuda_version"] = torch.version.cuda or "none"

    # HPC job scheduler (if available)
    env["slurm_job_id"] = os.environ.get("SLURM_JOB_ID", "local")
    env["slurm_node"] = os.environ.get("SLURM_NODELIST", "local")

    # pip freeze
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--local"],
            capture_output=True, text=True, timeout=30,
        )
        env["pip_freeze"] = result.stdout.strip() if result.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        env["pip_freeze"] = ""

    return env


def log_environment_to_mlflow(env: dict[str, Any]) -> None:
    """Log captured environment to MLflow as params/tags/artifacts.

    Args:
        env: Dictionary from capture_environment().

    Silently returns if MLflow is not available or logging fails.
    """
    try:
        import mlflow  # noqa: PLC0415

        mlflow.log_param("git_commit", env.get("git_commit", "unknown"))
        mlflow.set_tag("slurm_job_id", env.get("slurm_job_id", "local"))
        mlflow.set_tag("slurm_node", env.get("slurm_node", "local"))
        mlflow.set_tag("torch_version", env.get("torch_version", "unknown"))
        mlflow.set_tag("cuda_version", env.get("cuda_version", "none"))

        # Save pip freeze as artifact
        pip_freeze = env.get("pip_freeze", "")
        if pip_freeze:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", prefix="requirements_snapshot_", delete=False,
            ) as f:
                f.write(pip_freeze)
                f.flush()
                mlflow.log_artifact(f.name, "environment")
            os.unlink(f.name)
    except Exception:  # noqa: BLE001
        pass
