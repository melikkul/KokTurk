#!/bin/bash
#SBATCH -p orfoz
#SBATCH -A $SLURM_ACCOUNT
#SBATCH -J taac_retest
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 56
#SBATCH --time=1-00:00:00
#SBATCH --array=0-1
#SBATCH --output=$SCRATCH_DIR/logs/taac_retest_%a_%j.out
#SBATCH --error=$SCRATCH_DIR/logs/taac_retest_%a_%j.err

# TAAC retest with fixed LR multipliers [1.0, 1.0, 0.8, 0.5, 0.3]
# ID0 = TAAC-only (dual_head, no context)
# ID1 = full_v4 (contextual_dual_head + word2vec + TAAC)

set -euo pipefail
module load comp/python/miniconda3
cd $PROJECT_DIR
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

COMMON="
    --training-data data/splits/train.jsonl
    --eval-data data/splits/val.jsonl
    --char-vocab models/vocabs/char_vocab.json
    --tag-vocab models/vocabs/tag_vocab.json
    --root-vocab models/vocabs/root_vocab.json
    --base-lr 5e-4 --max-epochs 50 --batch-size 256
    --seed 42 --device cpu
"

if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    echo "=== TAAC retest: dual_head + TAAC (component) ==="
    python src/train/train_v4_master.py \
        --model dual_head --context-type none --curriculum taac \
        --transition-mode component \
        $COMMON \
        --output-dir models/taac_retest/id3
elif [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    echo "=== TAAC retest: full_v4 + TAAC (component) ==="
    python src/train/train_v4_master.py \
        --model contextual_dual_head --context-type word2vec \
        --w2v-path models/word2vec/tr_word2vec_128.bin \
        --word-vocab models/vocabs/word_vocab.json \
        --curriculum taac --transition-mode component \
        --context-dropout 0.3 \
        $COMMON \
        --output-dir models/taac_retest/id4
fi

echo "=== Done: $(date) ==="
