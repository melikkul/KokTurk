#!/bin/bash
#SBATCH -p orfoz
#SBATCH -A $SLURM_ACCOUNT
#SBATCH -J morpho_ensemble
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 56
#SBATCH --time=0-08:00:00
#SBATCH --mem=64G
#SBATCH --array=0-4
#SBATCH --output=$SCRATCH_DIR/logs/ensemble_%A_%a.out
#SBATCH --error=$SCRATCH_DIR/logs/ensemble_%A_%a.err

set -euo pipefail

module load comp/python/miniconda3

cd $PROJECT_DIR
source .venv/bin/activate

SEEDS=(42 123 456 789 1337)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== Ensemble training: seed=${SEED} task=${SLURM_ARRAY_TASK_ID} ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

export PYTHONPATH=src
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

python src/train/train_atomizer.py \
    --train-path data/splits/train.jsonl \
    --val-path data/splits/val.jsonl \
    --output-dir "models/ensemble/model_seed${SEED}/" \
    --embed-dim 64 \
    --hidden-dim 128 \
    --num-layers 2 \
    --batch-size 64 \
    --epochs 30 \
    --lr 0.001 \
    --dropout 0.3 \
    --seed "${SEED}" \
    --device cpu

echo "=== Training seed=${SEED} complete: $(date) ==="
