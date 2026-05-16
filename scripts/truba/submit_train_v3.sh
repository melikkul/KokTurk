#!/bin/bash
#SBATCH -p orfoz
#SBATCH -A $SLURM_ACCOUNT
#SBATCH -J morpho_v3
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 56
#SBATCH --time=0-08:00:00
#SBATCH --mem=64G
#SBATCH --output=$SCRATCH_DIR/logs/train_v3_%j.out
#SBATCH --error=$SCRATCH_DIR/logs/train_v3_%j.err

set -euo pipefail

module load comp/python/miniconda3

cd $PROJECT_DIR
source .venv/bin/activate

echo "=== Starting v3 training (curriculum + tier weights + scheduled sampling + copy) ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $(nproc)"

export PYTHONPATH=src
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

python src/train/train_atomizer.py \
    --train-path data/splits/train.jsonl \
    --val-path data/splits/val.jsonl \
    --output-dir models/atomizer_v3/ \
    --embed-dim 64 \
    --hidden-dim 128 \
    --num-layers 2 \
    --batch-size 64 \
    --epochs 30 \
    --lr 0.001 \
    --dropout 0.3 \
    --use-curriculum \
    --use-tier-weights \
    --use-scheduled-sampling \
    --use-copy-mechanism \
    --seed 42 \
    --device cpu

echo "=== Training complete: $(date) ==="
