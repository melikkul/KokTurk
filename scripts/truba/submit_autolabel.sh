#!/bin/bash
# Submit autolabeling job to Orfoz (CPU-only, 112 cores/node).
# Parallelizes over token shards; each node handles 50K tokens.
# Must be submitted from /arf/scratch/scolakoglu/
#
# Usage:
#   sbatch /arf/home/scolakoglu/NLP_Project/scripts/truba/submit_autolabel.sh

#SBATCH --job-name=aksu-autolabel
#SBATCH --partition=orfoz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --time=3-00:00:00
#SBATCH --output=/arf/scratch/scolakoglu/logs/autolabel_%j.out
#SBATCH --error=/arf/scratch/scolakoglu/logs/autolabel_%j.err
#SBATCH --array=0-49%8

set -euo pipefail

PROJECT=/arf/home/scolakoglu/NLP_Project
module load comp/python/miniconda3
source "$PROJECT/.venv/bin/activate"

mkdir -p /arf/scratch/scolakoglu/logs

SHARD_SIZE=50000
START=$(( SLURM_ARRAY_TASK_ID * SHARD_SIZE ))
END=$(( START + SHARD_SIZE ))

cd "$PROJECT"
python -m aksu.data.build.autolabel \
    --unique-tokens data/intermediate/unique_tokens.jsonl \
    --token-sentences data/intermediate/token_sentences.jsonl \
    --sentences data/intermediate/sentences.jsonl \
    --output "data/intermediate/autolabeled_shard_${SLURM_ARRAY_TASK_ID}.jsonl" \
    --shard-start "$START" \
    --shard-end "$END" \
    --seed $(( 42 + SLURM_ARRAY_TASK_ID ))

echo "Shard ${SLURM_ARRAY_TASK_ID} done (tokens ${START}-${END})"
