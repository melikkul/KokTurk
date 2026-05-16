#!/bin/bash
# Build Singularity/Apptainer images for TRUBA
# Run on TRUBA login node (arf-ui1 or cuda-ui)

set -euo pipefail

echo "=== Building Apptainer images ==="

# Ensure apptainer is available
command -v apptainer > /dev/null 2>&1 || { echo "ERROR: apptainer not found"; exit 1; }

# TODO: Build from Docker images when containers are implemented
# apptainer build morpho-train.sif docker://registry/morpho-atomizer:train-latest
# apptainer build morpho-base.sif docker://registry/morpho-atomizer:base-latest

echo "WARNING: Container build not yet configured. Implement Docker images first."
echo "For now, use the Python venv directly on TRUBA."
