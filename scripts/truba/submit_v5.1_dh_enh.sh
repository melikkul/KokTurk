#!/bin/bash
#SBATCH -p orfoz
#SBATCH -A $SLURM_ACCOUNT
#SBATCH -J v5.1_dh_enh
#SBATCH -N 1 -n 1 -c 56
#SBATCH --time=1-12:00:00
#SBATCH --output=$SCRATCH_DIR/logs/v5.1_dh_enh_%j.out
#SBATCH --error=$SCRATCH_DIR/logs/v5.1_dh_enh_%j.err

set -euo pipefail
module load comp/python/miniconda3
cd $PROJECT_DIR
source .venv/bin/activate
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "=== v5.1 Job B: Dual-Head Enhanced (Focal + R-Drop + VarDrop + EMA) ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Date: $(date)"

python src/train/train_v4_master.py \
    --model dual_head \
    --context-type none \
    --curriculum fixed \
    --training-data data/resource/training_export_v5.1.jsonl \
    --eval-data data/splits/val.jsonl \
    --char-vocab models/vocabs/char_vocab.json \
    --tag-vocab models/vocabs/tag_vocab.json \
    --root-vocab models/vocabs/root_vocab_15K.json \
    --base-lr 5e-4 --max-epochs 50 --batch-size 256 \
    --loss-fn focal --focal-gamma 2.0 \
    --rdrop-alpha 5.0 \
    --optimizer adamw --weight-decay 0.01 \
    --variational-dropout 0.2 \
    --ema-decay 0.999 \
    --early-stop-patience 999 \
    --output-dir models/v5.1/dh_enhanced --seed 42 --device cpu

echo "=== Done: $(date) ==="
