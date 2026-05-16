#!/bin/bash
#SBATCH -p orfoz
#SBATCH -A $SLURM_ACCOUNT
#SBATCH -J v5.2_dh
#SBATCH -N 1 -n 1 -c 56
#SBATCH --time=1-00:00:00
#SBATCH --output=$SCRATCH_DIR/logs/v5.2_dh_%j.out
#SBATCH --error=$SCRATCH_DIR/logs/v5.2_dh_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
module load comp/python/miniconda3
cd $PROJECT_DIR
source .venv/bin/activate
export PYTHONPATH=src

echo "=== v5.2 dual_head (106K augmented, original vocabs) ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

python src/train/train_v4_master.py \
    --model dual_head \
    --context-type none \
    --curriculum fixed \
    --training-data data/resource/training_augmented_80K+resource.jsonl \
    --eval-data data/splits/val.jsonl \
    --char-vocab models/vocabs/char_vocab.json \
    --tag-vocab models/vocabs/tag_vocab.json \
    --root-vocab models/vocabs/root_vocab.json \
    --base-lr 5e-4 \
    --max-epochs 50 \
    --batch-size 256 \
    --early-stop-patience 10 \
    --output-dir models/v5.2/dh \
    --seed 42 \
    --device cpu

echo "=== Done: $(date) ==="
