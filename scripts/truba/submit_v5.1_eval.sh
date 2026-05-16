#!/bin/bash
#SBATCH -p orfoz
#SBATCH -A $SLURM_ACCOUNT
#SBATCH -J v5.1_eval
#SBATCH -N 1 -n 1 -c 56
#SBATCH --time=0-01:00:00
#SBATCH --output=$SCRATCH_DIR/logs/v5.1_eval_%j.out
#SBATCH --error=$SCRATCH_DIR/logs/v5.1_eval_%j.err

set -euo pipefail
module load comp/python/miniconda3
cd $PROJECT_DIR
source .venv/bin/activate
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "=== v5.1 Evaluation: $(date) ==="

python -c "
from aksu.benchmark.intrinsic_eval import compute_all_metrics
import json, os

models = {
    'prev_best (80K dh)': 'models/ablation/dual_head/best_model.pt',
    'v5.1 dh (95K)': 'models/v5.1/dh/best_model.pt',
    'v5.1 dh+enh (95K)': 'models/v5.1/dh_enhanced/best_model.pt',
    'v5.1 dhctx (95K)': 'models/v5.1/dhctx/best_model.pt',
}

print(f'{\"Model\":30s} {\"Test EM\":>10s} {\"Root\":>10s} {\"Tag F1\":>10s}')
print('=' * 65)

results = {}
for name, path in models.items():
    if os.path.exists(path):
        try:
            m = compute_all_metrics(path, 'data/splits/test.jsonl', 'models/vocabs/')
            em = m.get('exact_match', 0)
            root = m.get('root_accuracy', 0)
            f1 = m.get('tag_f1', 0)
            print(f'{name:30s} {em:>9.1%} {root:>9.1%} {f1:>9.1%}')
            results[name] = m
        except Exception as e:
            print(f'{name:30s} ERROR: {e}')
    else:
        print(f'{name:30s} NOT FOUND')

if results:
    best = max(results.items(), key=lambda x: x[1].get('exact_match', 0))
    prev_em = 0.8445
    best_em = best[1].get('exact_match', 0)
    print(f'\nBest: {best[0]} = {best_em:.1%} (delta vs 84.45%: {best_em - prev_em:+.1%})')
    os.makedirs('models/v5.1', exist_ok=True)
    json.dump(results, open('models/v5.1/eval_results.json', 'w'), indent=2)
"

echo "=== Done: $(date) ==="
