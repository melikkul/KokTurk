#!/bin/bash
#SBATCH -p orfoz
#SBATCH -A $SLURM_ACCOUNT
#SBATCH -J noise_sweep
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 56
#SBATCH --time=2-00:00:00
#SBATCH --array=0-7
#SBATCH --output=$SCRATCH_DIR/logs/noise_%a_%j.out
#SBATCH --error=$SCRATCH_DIR/logs/noise_%a_%j.err

# Noise robustness sweep: 4 noise levels x 2 curricula = 8 configs
# Demonstrates TAAC's superior noise tolerance vs fixed curriculum.

set -euo pipefail
module load comp/python/miniconda3
cd $PROJECT_DIR
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

NOISE_LEVELS=(0.0 0.05 0.10 0.20 0.0 0.05 0.10 0.20)
CURRICULA=(fixed fixed fixed fixed taac taac taac taac)

NOISE=${NOISE_LEVELS[$SLURM_ARRAY_TASK_ID]}
CURR=${CURRICULA[$SLURM_ARRAY_TASK_ID]}

echo "=== Noise sweep: noise=$NOISE curriculum=$CURR ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

python src/train/train_v4_master.py \
    --model dual_head \
    --context-type none \
    --curriculum $CURR \
    --training-data data/splits/train.jsonl \
    --eval-data data/splits/val.jsonl \
    --char-vocab models/vocabs/char_vocab.json \
    --tag-vocab models/vocabs/tag_vocab.json \
    --root-vocab models/vocabs/root_vocab.json \
    --base-lr 5e-4 \
    --max-epochs 30 \
    --batch-size 256 \
    --embed-dim 64 \
    --hidden-dim 128 \
    --num-layers 2 \
    --dropout 0.3 \
    --gold-noise-rate $NOISE \
    --output-dir models/noise_sweep/noise_${NOISE}_${CURR} \
    --seed 42 \
    --device cpu

echo "=== Done: noise=$NOISE curriculum=$CURR $(date) ==="
