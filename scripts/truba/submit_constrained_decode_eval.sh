#!/bin/bash
#SBATCH --job-name=kokturk_constrained_decode
#SBATCH --partition=orfoz
#SBATCH --account=$SLURM_ACCOUNT
#SBATCH --cpus-per-task=56
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/constrained_decode_%j.out
#SBATCH --error=logs/constrained_decode_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USER

# Cat B Task 5 — evaluate the v2 atomizer with vs. without the
# morphotactic FSA constraint mask. Reports illegal-sequence counts on
# both runs so the FSA's effect can be quantified directly.

set -euo pipefail

PROJECT=$PROJECT_DIR
SCRATCH=$SCRATCH_DIR
cd "$SCRATCH"

mkdir -p "$PROJECT/jobs"
echo "$SLURM_JOB_ID constrained_decode $(date -Iseconds)" \
    >> "$PROJECT/jobs/submitted_jobs.log"

module load comp/python/miniconda3
source "$PROJECT/.venv/bin/activate"

export PYTHONPATH="$PROJECT/src"

OUT="$PROJECT/models/benchmark/constrained_decode"
mkdir -p "$OUT"

python -m benchmark.constrained_decode_eval \
    --model-path "$PROJECT/models/atomizer_v2/best_model.pt" \
    --vocab-dir "$PROJECT/models/vocabs/" \
    --test-path "$PROJECT/data/gold/test.jsonl" \
    --output-dir "$OUT" \
    --device cpu
