#!/bin/bash
#SBATCH --job-name=kokturk_rdrop
#SBATCH --partition=akya-cuda
#SBATCH --account=$SLURM_ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --time=20:00:00
#SBATCH --output=logs/rdrop_%j.out
#SBATCH --error=logs/rdrop_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USER

set -euo pipefail

PROJECT=$PROJECT_DIR
SCRATCH=$SCRATCH_DIR
cd "$SCRATCH"

mkdir -p "$PROJECT/jobs"
echo "$SLURM_JOB_ID rdrop_train $(date -Iseconds)" >> "$PROJECT/jobs/submitted_jobs.log"

module load comp/python/miniconda3
source "$PROJECT/.venv/bin/activate"

export PYTHONPATH="$PROJECT/src"

# Cat C v6_rdrop: R-Drop + variational dropout + AdamW + EMA
python -m train.train_v4_master \
    --model contextual_dual_head \
    --context-type word2vec \
    --w2v-path "$PROJECT/models/word2vec/tr_word2vec_128.bin" \
    --training-data "$PROJECT/data/splits/train.jsonl" \
    --eval-data "$PROJECT/data/splits/val.jsonl" \
    --char-vocab "$PROJECT/models/vocabs/char_vocab.json" \
    --tag-vocab "$PROJECT/models/vocabs/tag_vocab.json" \
    --root-vocab "$PROJECT/models/vocabs/root_vocab.json" \
    --curriculum taac \
    --loss-fn focal --focal-gamma 2.0 --label-smoothing 0.01 \
    --rdrop-alpha 5.0 \
    --variational-dropout 0.3 \
    --optimizer adamw --weight-decay 0.01 \
    --ema-decay 0.999 \
    --early-stop-metric val_loss --early-stop-patience 5 \
    --base-lr 5e-4 --max-epochs 40 --batch-size 256 \
    --output-dir "$PROJECT/models/v6_rdrop" \
    --seed 42
