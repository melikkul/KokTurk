#!/bin/bash
#SBATCH --job-name=kokturk_focal
#SBATCH --partition=akya-cuda
#SBATCH --account=$SLURM_ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=logs/focal_%j.out
#SBATCH --error=logs/focal_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USER

set -euo pipefail

PROJECT=$PROJECT_DIR
SCRATCH=$SCRATCH_DIR
cd "$SCRATCH"

mkdir -p "$PROJECT/jobs"
echo "$SLURM_JOB_ID focal $(date -Iseconds)" >> "$PROJECT/jobs/submitted_jobs.log"

module load comp/python/miniconda3
source "$PROJECT/.venv/bin/activate"

export PYTHONPATH="$PROJECT/src"

python -m train.train_v4_master \
    --model single_seq2seq \
    --curriculum fixed \
    --loss-fn focal \
    --focal-gamma 2.0 \
    --label-smoothing 0.1 \
    --base-lr 5e-4 \
    --max-epochs 30 \
    --batch-size 256 \
    --seed 42 \
    --char-vocab "$PROJECT/models/vocabs/char_vocab.json" \
    --tag-vocab "$PROJECT/models/vocabs/tag_vocab.json" \
    --training-data "$PROJECT/data/gold/tr_gold_morph_v1.jsonl" \
    --output-dir "$PROJECT/models/v5_focal"
