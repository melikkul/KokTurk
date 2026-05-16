#!/bin/bash
#SBATCH -p orfoz
#SBATCH -A $SLURM_ACCOUNT
#SBATCH -J morpho_preprocess
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 112
#SBATCH --time=0-12:00:00
#SBATCH --output=$SCRATCH_DIR/logs/preprocess_%j.out
#SBATCH --error=$SCRATCH_DIR/logs/preprocess_%j.err

set -euo pipefail

module load comp/python/miniconda3

cd $PROJECT_DIR
source .venv/bin/activate

echo "=== Starting preprocessing job ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $(nproc)"

python src/data/prelabel.py \
    --input data/raw/corpus.jsonl \
    --output data/prelabeled/ \
    --workers 112

echo "=== Preprocessing complete: $(date) ==="
