#!/bin/bash
#SBATCH -p orfoz
#SBATCH -A $SLURM_ACCOUNT
#SBATCH -J v6_disamb
#SBATCH -N 1 -n 1 -c 56
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --output=$SCRATCH_DIR/logs/v6_disamb_%j.out
#SBATCH --error=$SCRATCH_DIR/logs/v6_disamb_%j.err

set -euo pipefail
module load comp/python/miniconda3
cd $PROJECT_DIR
source .venv/bin/activate
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "=== v6 Disambiguation Training ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Date: $(date)"

python -m train.train_disambiguator \
    --train-data data/splits/train.jsonl \
    --val-data data/splits/val.jsonl \
    --tag-vocab models/vocabs/tag_vocab.json \
    --berturk-path models/berturk \
    --cache-dir $SCRATCH_DIR/bert_cache \
    --lr 1e-3 --epochs 30 --batch-size 128 \
    --patience 7 \
    --output-dir models/v6/disambiguator --seed 42 --device cpu

echo "=== Done: $(date) ==="
