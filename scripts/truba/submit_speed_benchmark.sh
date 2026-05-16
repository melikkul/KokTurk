#!/bin/bash
#SBATCH -p orfoz
#SBATCH -A $SLURM_ACCOUNT
#SBATCH -J cat-d-spd
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 56
#SBATCH --time=0-01:00:00
#SBATCH --mem=32G
#SBATCH --output=$SCRATCH_DIR/logs/speed_%j.out
#SBATCH --error=$SCRATCH_DIR/logs/speed_%j.err

# For GPU speed benchmarking, swap this header for:
#   #SBATCH -p akya-cuda
#   #SBATCH -c 10
#   #SBATCH --gres=gpu:1
# and set DEVICE=cuda below.

set -euo pipefail
module load comp/python/miniconda3
cd $PROJECT_DIR
source .venv/bin/activate
export PYTHONPATH=src

python -m benchmark.speed_benchmark
