#!/bin/bash
#SBATCH -p orfoz
#SBATCH -A $SLURM_ACCOUNT
#SBATCH -J v5_s2s_545K
#SBATCH -N 1 -n 1 -c 56
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G
#SBATCH --output=$SCRATCH_DIR/logs/v5_s2s_%j.out
#SBATCH --error=$SCRATCH_DIR/logs/v5_s2s_%j.err

set -euo pipefail
module load comp/python/miniconda3
cd $PROJECT_DIR
source .venv/bin/activate
export PYTHONPATH=src

echo "=== v5 seq2seq on 545K data: $(date) ==="
echo "Node: $(hostname)"

python src/train/train_v4_master.py \
    --model single_seq2seq \
    --curriculum fixed \
    --training-data data/resource/training_export_2.5M.jsonl \
    --eval-data data/splits/val.jsonl \
    --char-vocab models/vocabs/char_vocab.json \
    --tag-vocab models/vocabs/tag_vocab.json \
    --base-lr 1e-3 --max-epochs 30 --batch-size 256 \
    --output-dir models/v5_s2s_545K \
    --seed 42 --device cpu

echo "=== Done: $(date) ==="
