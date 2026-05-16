#!/bin/bash
#SBATCH --job-name=kokturk_hpo
#SBATCH --partition=orfoz
#SBATCH --account=$SLURM_ACCOUNT
#SBATCH --cpus-per-task=56
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/hpo_%j.out
#SBATCH --error=logs/hpo_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USER

set -euo pipefail

PROJECT=$PROJECT_DIR
SCRATCH=$SCRATCH_DIR
cd "$SCRATCH"

mkdir -p "$PROJECT/jobs" "$PROJECT/models/hpo"
echo "$SLURM_JOB_ID hpo $(date -Iseconds)" >> "$PROJECT/jobs/submitted_jobs.log"

module load comp/python/miniconda3
source "$PROJECT/.venv/bin/activate"

# Optuna is not in base requirements — ensure it's available.
python -c "import optuna" 2>/dev/null || pip install optuna

export PYTHONPATH="$PROJECT/src"

python -m train.hpo \
    --n-trials 50 \
    --train-path "$PROJECT/data/splits/train.jsonl" \
    --val-path   "$PROJECT/data/splits/val.jsonl" \
    --max-epochs 15 \
    --device cpu \
    --study-name morpho_hpo \
    --storage "sqlite:///$PROJECT/models/hpo/optuna.db"
