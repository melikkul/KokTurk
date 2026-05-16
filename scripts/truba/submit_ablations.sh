#!/bin/bash
#SBATCH -p orfoz
#SBATCH -A $SLURM_ACCOUNT
#SBATCH -J morpho_ablation
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 56
#SBATCH --time=1-00:00:00
#SBATCH --array=0-4
#SBATCH --output=$SCRATCH_DIR/logs/ablation_%x_%A_%a.out
#SBATCH --error=$SCRATCH_DIR/logs/ablation_%x_%A_%a.err

# Ablation study: isolate contributions of dual-head, context, and TAAC.
#
# Ablation design — each row adds exactly one component over the previous:
#   ID 0  baseline_seq2seq    MorphAtomizer (original seq2seq)   — true baseline
#   ID 1  dual_head           DualHeadAtomizer, no context        — +dual-head
#   ID 2  dual_head_context   ContextualDualHead, word2vec, fixed — +context
#   ID 3  dual_head_taac      DualHeadAtomizer, no ctx, taac      — +TAAC (isolated)
#   ID 4  full_v4             ContextualDualHead, word2vec, taac  — full model
#
# Delta analysis:
#   Δ(dual-head) = ID1 - ID0
#   Δ(context)   = ID2 - ID1
#   Δ(TAAC)      = ID4 - ID2
#
# Usage:
#   sbatch scripts/truba/submit_ablations.sh

set -euo pipefail

module load comp/python/miniconda3

cd $PROJECT_DIR
source .venv/bin/activate

echo "=== Ablation job ${SLURM_ARRAY_TASK_ID} ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $(nproc)"

export PYTHONPATH=src
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Common flags shared by all ablations
COMMON="
    --training-data data/splits/train.jsonl
    --eval-data data/splits/val.jsonl
    --char-vocab models/vocabs/char_vocab.json
    --tag-vocab models/vocabs/tag_vocab.json
    --root-vocab models/vocabs/root_vocab.json
    --word-vocab models/vocabs/word_vocab.json
    --base-lr 5e-4
    --max-epochs 50
    --batch-size 256
    --embed-dim 64
    --hidden-dim 128
    --num-layers 2
    --dropout 0.3
    --seed 42
    --device cpu
"

case ${SLURM_ARRAY_TASK_ID} in
  0)
    NAME="baseline_seq2seq"
    EXTRA="--model single_seq2seq --context-type none --curriculum fixed"
    ;;
  1)
    NAME="dual_head"
    EXTRA="--model dual_head --context-type none --curriculum fixed"
    ;;
  2)
    NAME="dual_head_context"
    EXTRA="--model contextual_dual_head --context-type word2vec --curriculum fixed
           --w2v-path models/word2vec/tr_word2vec_128.bin --context-dropout 0.3"
    ;;
  3)
    NAME="dual_head_taac"
    EXTRA="--model dual_head --context-type none --curriculum taac"
    ;;
  4)
    NAME="full_v4"
    EXTRA="--model contextual_dual_head --context-type word2vec --curriculum taac
           --w2v-path models/word2vec/tr_word2vec_128.bin --context-dropout 0.3"
    ;;
  *)
    echo "ERROR: Unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

echo "Running ablation: ${NAME}"
echo "Extra flags: ${EXTRA}"

python src/train/train_v4_master.py \
    ${COMMON} \
    ${EXTRA} \
    --output-dir models/ablations/${NAME}

echo "=== Ablation ${NAME} complete: $(date) ==="
