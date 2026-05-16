#!/bin/bash
#SBATCH -p orfoz
#SBATCH -A $SLURM_ACCOUNT
#SBATCH -J v5.1_catd
#SBATCH -N 1 -n 1 -c 56
#SBATCH --time=0-01:00:00
#SBATCH --output=$SCRATCH_DIR/logs/v5.1_catd_%j.out
#SBATCH --error=$SCRATCH_DIR/logs/v5.1_catd_%j.err

set -euo pipefail
module load comp/python/miniconda3
cd $PROJECT_DIR
source .venv/bin/activate
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

BEST="models/ablation/dual_head/best_model.pt"
TEST="data/splits/test.jsonl"
VOCABS="models/vocabs/"

echo "=== Cat D Eval (fixed imports): $(date) ==="

echo ""
echo "--- Error Analysis ---"
python -c "
from aksu.benchmark.error_analysis import classify_errors, generate_error_report
print('error_analysis module loaded OK')
print('Available: classify_errors, generate_error_report')
" 2>&1

echo ""
echo "--- Weighted EM ---"
python -c "
from aksu.benchmark.weighted_em import corpus_weighted_em, score_pair
# Quick test with sample pairs
pairs = [
    ('ev +PLU +POSS.3SG +ABL', 'ev +PLU +POSS.3SG +ABL'),  # perfect
    ('ev +PLU +POSS.3SG +ABL', 'ev +PLU +ABL'),              # missing tag
    ('ev +PLU +POSS.3SG +ABL', 'evler +POSS.3SG +ABL'),      # wrong root
]
for gold, pred in pairs:
    s = score_pair(gold, pred)
    print(f'  {gold:35s} vs {pred:35s} => {s:.3f}')

# Corpus level
golds = [g for g, _ in pairs]
preds = [p for _, p in pairs]
print(f'  Corpus weighted EM: {corpus_weighted_em(golds, preds):.3f}')
" 2>&1

echo ""
echo "--- Speed Benchmark ---"
python -c "
from aksu.benchmark.speed_benchmark import benchmark_inference, generate_speed_report
print('speed_benchmark module loaded OK')
print('Available: benchmark_inference, generate_speed_report')
" 2>&1

echo ""
echo "--- Minimal Pairs ---"
python -c "
from aksu.benchmark.minimal_pairs import evaluate_minimal_pairs, load_pairs
pairs = load_pairs('configs/eval/minimal_pairs.yaml')
print(f'Loaded {len(pairs)} minimal pairs')
for p in pairs[:5]:
    print(f'  {p}')
" 2>&1

echo ""
echo "--- Robustness ---"
python -c "
from aksu.benchmark.robustness import run_robustness_suite
print('robustness module loaded OK')
print('Available: run_robustness_suite')
" 2>&1

echo ""
echo "--- Significance ---"
python -c "
from aksu.benchmark.significance import paired_bootstrap_test, multi_system_significance_report
print('significance module loaded OK')
" 2>&1

echo ""
echo "--- Checklist ---"
python -c "
from aksu.benchmark.checklist_morpho import generate_mft_tests
tests = generate_mft_tests()
print(f'Generated {len(tests)} MFT test cases')
" 2>&1

echo ""
echo "=== All Cat D modules verified: $(date) ==="
