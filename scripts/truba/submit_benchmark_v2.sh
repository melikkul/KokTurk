#!/bin/bash
#SBATCH -p orfoz
#SBATCH -A $SLURM_ACCOUNT
#SBATCH -J morpho_bench_v2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 56
#SBATCH --time=0-02:00:00
#SBATCH --mem=64G
#SBATCH --output=$SCRATCH_DIR/logs/bench_v2_%j.out
#SBATCH --error=$SCRATCH_DIR/logs/bench_v2_%j.err

set -euo pipefail

module load comp/python/miniconda3

cd $PROJECT_DIR
source .venv/bin/activate

echo "=== Running full benchmark with new methods ==="
echo "Date: $(date)"

export PYTHONPATH=src
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

python src/benchmark/run_all_benchmarks.py

echo "=== Benchmark complete: $(date) ==="
