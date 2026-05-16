#!/bin/bash
#SBATCH -p orfoz
#SBATCH -A $SLURM_ACCOUNT
#SBATCH -J eval_all
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 56
#SBATCH --time=0-06:00:00
#SBATCH --output=$SCRATCH_DIR/logs/eval_all_%j.out
#SBATCH --error=$SCRATCH_DIR/logs/eval_all_%j.err

set -euo pipefail
module load comp/python/miniconda3
cd $PROJECT_DIR
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "=== Evaluating all v4 checkpoints on test split ==="
echo "Date: $(date)"
echo ""

run_eval() {
    local ckpt=$1
    local label=$2
    if [ -f "$ckpt" ]; then
        echo ""
        python scripts/eval_v4_models.py "$ckpt" "$label" 2>&1 | grep -E "===|EM|Root|Tag F1|val_loss|n  " || true
    else
        echo "SKIP (not found): $ckpt"
    fi
}

run_eval "models/ablations/baseline_seq2seq/best_model.pt" "ID0 seq2seq baseline"
run_eval "models/ablations/dual_head/best_model.pt" "ID1 dual_head (fixed)"
run_eval "models/ablations/dual_head_context/best_model.pt" "ID2 dual_head+context (fixed)"
run_eval "models/ablations/dual_head_taac/best_model.pt" "ID3 TAAC only (old LR)"
run_eval "models/ablations/full_v4/best_model.pt" "ID4 full_v4 (old LR)"
run_eval "models/taac_retest/id3/best_model.pt" "ID3 TAAC only (fixed LR)"
run_eval "models/taac_retest/id4/best_model.pt" "ID4 full_v4 (fixed LR)"
run_eval "models/v4_berturk/best_model.pt" "v4 BERTurk (old LR)"

echo ""
echo "=== Done: $(date) ==="
