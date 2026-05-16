#!/bin/bash
#SBATCH -p orfoz
#SBATCH -A $SLURM_ACCOUNT
#SBATCH -J v5_dhctx_545K
#SBATCH -N 1 -n 1 -c 56
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G
#SBATCH --output=$SCRATCH_DIR/logs/v5_dhctx_%j.out
#SBATCH --error=$SCRATCH_DIR/logs/v5_dhctx_%j.err

set -euo pipefail
module load comp/python/miniconda3
cd $PROJECT_DIR
source .venv/bin/activate
export PYTHONPATH=src

echo "=== v5 dual-head+context on 545K data + expanded root vocab: $(date) ==="
echo "Node: $(hostname)"

python src/train/train_v4_master.py \
    --model contextual_dual_head \
    --context-type word2vec \
    --w2v-path models/word2vec/tr_word2vec_128.bin \
    --curriculum fixed \
    --training-data data/resource/training_export_2.5M.jsonl \
    --eval-data data/splits/val.jsonl \
    --char-vocab models/vocabs/char_vocab.json \
    --tag-vocab models/vocabs/tag_vocab.json \
    --root-vocab models/vocabs/root_vocab_2.5M.json \
    --word-vocab models/vocabs/word_vocab_2.5M.json \
    --base-lr 5e-4 --max-epochs 30 --batch-size 256 \
    --output-dir models/v5_dhctx_545K \
    --seed 42 --device cpu

echo "=== Done: $(date) ==="
