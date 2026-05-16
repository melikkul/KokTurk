#!/bin/bash
#SBATCH -p orfoz
#SBATCH -A $SLURM_ACCOUNT
#SBATCH -J morpho_v4
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 56
#SBATCH --time=1-00:00:00
#SBATCH --output=$SCRATCH_DIR/logs/train_v4_%j.out
#SBATCH --error=$SCRATCH_DIR/logs/train_v4_%j.err

set -euo pipefail

module load comp/python/miniconda3

cd $PROJECT_DIR
source .venv/bin/activate

echo "=== Starting v4 training (CPU) ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $(nproc)"

export PYTHONPATH=src
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

python src/train/train_v4_master.py \
    --model contextual_dual_head \
    --context-type word2vec \
    --w2v-path models/word2vec/tr_word2vec_128.bin \
    --curriculum taac \
    --training-data data/splits/train.jsonl \
    --eval-data data/splits/val.jsonl \
    --char-vocab models/vocabs/char_vocab.json \
    --tag-vocab models/vocabs/tag_vocab.json \
    --root-vocab models/vocabs/root_vocab.json \
    --word-vocab models/vocabs/word_vocab.json \
    --base-lr 5e-4 \
    --max-epochs 50 \
    --batch-size 256 \
    --embed-dim 64 \
    --hidden-dim 128 \
    --num-layers 2 \
    --dropout 0.3 \
    --context-dropout 0.3 \
    --output-dir models/v4 \
    --seed 42 \
    --device cpu

echo "=== Training complete: $(date) ==="
