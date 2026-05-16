#!/bin/bash
#SBATCH --job-name=kokturk_noise_audit
#SBATCH --partition=orfoz
#SBATCH --account=$SLURM_ACCOUNT
#SBATCH --cpus-per-task=56
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/noise_audit_%j.out
#SBATCH --error=logs/noise_audit_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USER

set -euo pipefail

PROJECT=$PROJECT_DIR
SCRATCH=$SCRATCH_DIR
cd "$SCRATCH"

mkdir -p "$PROJECT/jobs"
echo "$SLURM_JOB_ID noise_audit $(date -Iseconds)" >> "$PROJECT/jobs/submitted_jobs.log"

module load comp/python/miniconda3
source "$PROJECT/.venv/bin/activate"

export PYTHONPATH="$PROJECT/src"

python -m data.noise_audit \
    --model-path "$PROJECT/models/atomizer_v2/best_model.pt" \
    --corpus-path "$PROJECT/data/gold/tr_gold_morph_v1.jsonl" \
    --vocab-dir "$PROJECT/models/vocabs/" \
    --output-dir "$PROJECT/data/noise_audit" \
    --device cpu
