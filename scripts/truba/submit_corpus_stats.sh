#!/bin/bash
#SBATCH --job-name=kokturk_corpus_stats
#SBATCH --partition=orfoz
#SBATCH --account=$SLURM_ACCOUNT
#SBATCH --cpus-per-task=56
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/corpus_stats_%j.out
#SBATCH --error=logs/corpus_stats_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USER

set -euo pipefail

PROJECT=$PROJECT_DIR
SCRATCH=$SCRATCH_DIR

cd "$SCRATCH"
mkdir -p "$PROJECT/jobs"
echo "$SLURM_JOB_ID corpus_stats $(date -Iseconds)" >> "$PROJECT/jobs/submitted_jobs.log"

module load comp/python/miniconda3
source "$PROJECT/.venv/bin/activate"
export PYTHONPATH="$PROJECT/src"

echo "=== Corpus Statistics ==="

# Run on training data
python -m benchmark.corpus_stats --corpus "$PROJECT/data/splits/train.jsonl"

echo "=== Done ==="
