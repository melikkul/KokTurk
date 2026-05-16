#!/bin/bash
#SBATCH -p orfoz
#SBATCH -A $SLURM_ACCOUNT
#SBATCH -J bert_ft_bench
#SBATCH -N 1 -n 1 -c 56
#SBATCH --time=0-04:00:00
#SBATCH --output=$SCRATCH_DIR/logs/bert_ft_bench_%j.out
#SBATCH --error=$SCRATCH_DIR/logs/bert_ft_bench_%j.err

set -euo pipefail
module load comp/python/miniconda3
cd $PROJECT_DIR
source .venv/bin/activate
export PYTHONPATH=src

echo "Starting BERT/FastText benchmark on TTC-3600"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Python: $(which python)"
echo ""

python src/benchmark/run_bert_fasttext_benchmark.py

echo ""
echo "Completed: $(date)"
