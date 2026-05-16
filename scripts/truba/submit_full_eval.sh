#!/bin/bash
#SBATCH -p orfoz
#SBATCH -A $SLURM_ACCOUNT
#SBATCH -J full_eval_25
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 56
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=$SCRATCH_DIR/logs/full_eval_%j.out
#SBATCH --error=$SCRATCH_DIR/logs/full_eval_%j.err

set -euo pipefail

echo "============================================================"
echo "Full Test-Set Evaluation: 25 models x 8140 tokens"
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "Date:     $(date)"
echo "============================================================"
echo ""

# --- Environment ---
module load comp/python/miniconda3
cd $PROJECT_DIR
source .venv/bin/activate

export PYTHONPATH=src
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
# Ensure PyTorch uses only CPU (orfoz has no GPU)
export CUDA_VISIBLE_DEVICES=""

# --- Ensure output directories exist ---
mkdir -p models/benchmark
mkdir -p $SCRATCH_DIR/logs

# --- Run evaluation ---
echo "Starting evaluation at $(date)"
echo ""

python scripts/eval_full_test.py

echo ""
echo "Finished at $(date)"
echo "Results written to: models/benchmark/full_test_eval.tsv"
echo ""

# Print the TSV nicely with column alignment
echo "--- Results Table ---"
if [ -f models/benchmark/full_test_eval.tsv ]; then
    column -t -s $'\t' models/benchmark/full_test_eval.tsv
fi
echo "--- End ---"
