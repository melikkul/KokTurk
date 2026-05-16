#!/bin/bash
#SBATCH -p akya-cuda
#SBATCH -A $SLURM_ACCOUNT
#SBATCH -J kokturkain
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --time=0-04:00:00
#SBATCH --mem=32G
#SBATCH --output=$SCRATCH_DIR/logs/train_%j.out
#SBATCH --error=$SCRATCH_DIR/logs/train_%j.err

set -euo pipefail

module load lib/cuda/12.4
module load comp/python/miniconda3

cd $PROJECT_DIR
source .venv/bin/activate

echo "=== Starting training job ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

export PYTHONPATH=src

python src/train/train_atomizer.py

echo "=== Training complete: $(date) ==="
