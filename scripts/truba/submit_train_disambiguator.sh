#!/bin/bash
# Re-time disambiguation training on Orfoz (CPU-only) to produce an authoritative
# training_wall_clock_min measurement for metrics.json.
#
# This is NOT the original training run; it is a single-seed re-timing job
# to verify the "14 min" claim from STATUS.md under SLURM accounting.
#
# Usage:
#   sbatch /arf/home/scolakoglu/NLP_Project/scripts/truba/submit_train_disambiguator.sh

#SBATCH --job-name=aksu-disamb-retime
#SBATCH --partition=orfoz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 56
#SBATCH --time=02:00:00
#SBATCH --output=/arf/scratch/scolakoglu/logs/disamb_retime_%j.out
#SBATCH --error=/arf/scratch/scolakoglu/logs/disamb_retime_%j.err

set -euo pipefail

PROJECT=/arf/home/scolakoglu/NLP_Project
module load comp/python/miniconda3
source "$PROJECT/.venv/bin/activate"

mkdir -p /arf/scratch/scolakoglu/logs

cd "$PROJECT"

START=$(date +%s)

/usr/bin/time -v python -m aksu.train.train_disambiguator \
    --config configs/train/v6_full.yaml \
    --seed 42 \
    --output-dir models/v6_retimed \
    2>&1 | tee /arf/scratch/scolakoglu/logs/disamb_retime_${SLURM_JOB_ID}_stdout.txt

END=$(date +%s)
WALL_CLOCK_SEC=$(( END - START ))
WALL_CLOCK_MIN=$(echo "scale=2; $WALL_CLOCK_SEC / 60" | bc)

# Write training log for metrics ingestion
mkdir -p "$PROJECT/models/v6_retimed"
python -c "
import json, sys
log = {
    'wall_clock_seconds': ${WALL_CLOCK_SEC},
    'wall_clock_minutes': ${WALL_CLOCK_MIN},
    'slurm_job_id': '${SLURM_JOB_ID}',
    'partition': '${SLURM_JOB_PARTITION}',
    'hostname': '$(hostname)',
    'seed': 42,
}
path = 'models/v6_retimed/training_log.json'
with open(path, 'w') as f:
    json.dump(log, f, indent=2)
print(f'Training log written to {path}')
print(f'Wall clock: ${WALL_CLOCK_MIN} min (${WALL_CLOCK_SEC} sec)')
"

echo "=== Re-timing complete. To ingest:"
echo "python scripts/ingest_metrics.py \\"
echo "    --source models/v6_retimed/training_log.json \\"
echo "    --keys wall_clock_minutes:training_wall_clock_min \\"
echo "    --target audit/benchmark_results/metrics.json"
