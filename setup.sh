#!/bin/bash
# KokTurk — First-time setup
set -euo pipefail

echo "=== Setting up KokTurk ==="

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Download required data (BOUN Treebank)
mkdir -p data/external
if [ ! -d "data/external/boun_treebank" ]; then
    echo "Downloading BOUN Treebank..."
    git clone --depth 1 https://github.com/UniversalDependencies/UD_Turkish-BOUN.git data/external/boun_treebank/
fi

# Run tests
echo "Running tests..."
PYTHONPATH=src pytest tests/ -x -q --ignore=tests/regression/ -m "not slow and not gpu"

echo "=== Setup complete ==="
echo "Activate: source .venv/bin/activate"
echo "Test:     PYTHONPATH=src pytest tests/ -q"
echo "Analyze:  PYTHONPATH=src python -m kokturk.cli.main analyze 'evlerinden'"
