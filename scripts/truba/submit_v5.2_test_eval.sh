#!/bin/bash
#SBATCH -p orfoz
#SBATCH -A $SLURM_ACCOUNT
#SBATCH -J v5.2_test
#SBATCH -N 1 -n 1 -c 56
#SBATCH --time=0-01:00:00
#SBATCH --output=$SCRATCH_DIR/logs/v5.2_test_%j.out
#SBATCH --error=$SCRATCH_DIR/logs/v5.2_test_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
module load comp/python/miniconda3
cd $PROJECT_DIR
source .venv/bin/activate
export PYTHONPATH=src

echo "=== v5.2 Test-Set Evaluation: $(date) ==="

python scripts/eval_v4_models.py models/v5.2/dh/best_model.pt "v5.2_dh (106K)"
python scripts/eval_v4_models.py models/v5.2/dh_rdrop/best_model.pt "v5.2_rdrop (106K)"

# Also eval the previous best for comparison (if exists)
for m in models/ablation/dual_head/best_model.pt models/ablation/dual_head_context/best_model.pt; do
    if [ -f "$m" ]; then
        python scripts/eval_v4_models.py "$m" "prev: $m"
    else
        echo "NOT FOUND: $m"
    fi
done

echo "=== Done: $(date) ==="
