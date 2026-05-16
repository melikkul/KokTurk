#!/bin/bash
#SBATCH -p orfoz
#SBATCH -A $SLURM_ACCOUNT
#SBATCH -J cat-d-std
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 56
#SBATCH --time=0-02:00:00
#SBATCH --mem=64G
#SBATCH --output=$SCRATCH_DIR/logs/standard_%j.out
#SBATCH --error=$SCRATCH_DIR/logs/standard_%j.err

set -euo pipefail
module load comp/python/miniconda3
cd $PROJECT_DIR
source .venv/bin/activate
export PYTHONPATH=src

python -m benchmark.standard_benchmarks
