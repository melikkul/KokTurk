#!/bin/bash
#SBATCH --job-name=kokturk_strat_eval
#SBATCH --partition=orfoz
#SBATCH --account=$SLURM_ACCOUNT
#SBATCH --cpus-per-task=56
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/strat_eval_%j.out
#SBATCH --error=logs/strat_eval_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USER

# Supports dependency chaining:
#     sbatch --dependency=afterok:$TRAIN_JOB_ID submit_stratified_eval.sh

set -euo pipefail

PROJECT=$PROJECT_DIR
SCRATCH=$SCRATCH_DIR
cd "$SCRATCH"

mkdir -p "$PROJECT/jobs"
echo "$SLURM_JOB_ID strat_eval $(date -Iseconds)" >> "$PROJECT/jobs/submitted_jobs.log"

module load comp/python/miniconda3
source "$PROJECT/.venv/bin/activate"

export PYTHONPATH="$PROJECT/src"

# Ensure frequency JSON is present on scratch.
FREQ="$PROJECT/models/benchmark/tag_frequency.json"
if [ ! -f "$FREQ" ]; then
    python -m benchmark.tag_frequency
fi

python -m benchmark.tag_frequency
# Stratified eval is driven by a small Python entrypoint that loops over
# available checkpoints. The module is importable via CLI:
python -c "
from pathlib import Path
from aksu.benchmark.stratified_eval import inspect_checkpoint
for p in Path('$PROJECT/models').rglob('best_model.pt'):
    print(p, inspect_checkpoint(p))
"
