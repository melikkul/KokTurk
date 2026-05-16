#!/bin/bash
#SBATCH -p orfoz
#SBATCH -A $SLURM_ACCOUNT
#SBATCH -J kokturkain_cpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 56
#SBATCH --time=0-03:00:00
#SBATCH --mem=64G
#SBATCH --output=$SCRATCH_DIR/logs/train_cpu_%j.out
#SBATCH --error=$SCRATCH_DIR/logs/train_cpu_%j.err

set -euo pipefail

module load comp/python/miniconda3

cd $PROJECT_DIR
source .venv/bin/activate

echo "=== Starting CPU training job ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $(nproc)"

export PYTHONPATH=src
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

python src/train/train_atomizer.py

echo "=== Training complete: $(date) ==="
