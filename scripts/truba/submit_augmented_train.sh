#!/bin/bash
#SBATCH --job-name=kokturk_aug
#SBATCH --partition=akya-cuda
#SBATCH --account=$SLURM_ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --time=14:00:00
#SBATCH --output=logs/aug_%j.out
#SBATCH --error=logs/aug_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USER

set -euo pipefail

PROJECT=$PROJECT_DIR
SCRATCH=$SCRATCH_DIR
cd "$SCRATCH"

mkdir -p "$PROJECT/jobs"
echo "$SLURM_JOB_ID aug_train $(date -Iseconds)" >> "$PROJECT/jobs/submitted_jobs.log"

module load comp/python/miniconda3
source "$PROJECT/.venv/bin/activate"

export PYTHONPATH="$PROJECT/src"

# Preflight: targeted augmentation needs tag_frequency.json.
FREQ="$PROJECT/models/benchmark/tag_frequency.json"
if [ ! -f "$FREQ" ]; then
    echo "[preflight] Generating missing $FREQ"
    python -m benchmark.tag_frequency
fi

# Generate synthetic corpus.
python -c "
from src.resource.importers.synthetic_inflections import import_synthetic_inflections
import_synthetic_inflections(
    '$PROJECT/models/benchmark/tag_frequency.json',
    '$PROJECT/data/gold/synthetic_inflections.jsonl',
)
" || python - <<'PY'
from aksu.resource.importers.synthetic_inflections import import_synthetic_inflections
import_synthetic_inflections(
    'models/benchmark/tag_frequency.json',
    'data/gold/synthetic_inflections.jsonl',
)
PY

# Train on the augmented corpus.
python -m train.train_v4_master \
    --model single_seq2seq \
    --curriculum taac \
    --loss-fn focal \
    --focal-gamma 2.0 \
    --max-epochs 30 \
    --batch-size 256 \
    --seed 42 \
    --char-vocab "$PROJECT/models/vocabs/char_vocab.json" \
    --tag-vocab "$PROJECT/models/vocabs/tag_vocab.json" \
    --training-data "$PROJECT/data/gold/tr_gold_morph_v1.jsonl" \
    --output-dir "$PROJECT/models/v5_augmented"
