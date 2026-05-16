#!/bin/bash
#SBATCH --job-name=kokturk_cat_f_comp
#SBATCH --partition=orfoz
#SBATCH --account=$SLURM_ACCOUNT
#SBATCH --cpus-per-task=56
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/competitor_bench_%j.out
#SBATCH --error=logs/competitor_bench_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USER

# Category F: Competitor accuracy benchmarking
# Installs optional deps (stanza, spacy, spacy-udpipe) then runs
# competitor_accuracy and llm_baseline modules.

set -euo pipefail

PROJECT=$PROJECT_DIR
SCRATCH=$SCRATCH_DIR
cd "$SCRATCH"

mkdir -p "$PROJECT/jobs"
echo "$SLURM_JOB_ID cat_f_competitor_bench $(date -Iseconds)" >> "$PROJECT/jobs/submitted_jobs.log"

module load comp/python/miniconda3
source "$PROJECT/.venv/bin/activate"

export PYTHONPATH="$PROJECT/src"

# Install optional competitor dependencies (skip if already present).
pip install --quiet stanza spacy spacy-udpipe 2>/dev/null || true

# Download models (skip if already cached).
python -c "import stanza; stanza.download('tr')" 2>/dev/null || true
python -c "import spacy_udpipe; spacy_udpipe.download('tr')" 2>/dev/null || true

# Run competitor accuracy benchmark.
python -m benchmark.competitor_accuracy

# Run LLM baseline (generates prompt templates if no API keys).
python -m benchmark.llm_baseline
