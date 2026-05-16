#!/bin/bash
# Pilot autolabel run: 100K tokens from shard 0 to estimate full-run yield.
# Must complete before submit_autolabel.sh (full array) is submitted.
# Outputs data/intermediate/autolabel_pilot.jsonl for yield extrapolation.
#
# Usage:
#   sbatch /arf/home/scolakoglu/NLP_Project/scripts/truba/submit_autolabel_pilot.sh

#SBATCH --job-name=aksu-autolabel-pilot
#SBATCH --partition=orfoz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 56
#SBATCH --time=02:00:00
#SBATCH --output=/arf/scratch/scolakoglu/logs/autolabel_pilot_%j.out
#SBATCH --error=/arf/scratch/scolakoglu/logs/autolabel_pilot_%j.err

set -euo pipefail

PROJECT=/arf/home/scolakoglu/NLP_Project
module load comp/python/miniconda3
source "$PROJECT/.venv/bin/activate"

mkdir -p /arf/scratch/scolakoglu/logs
mkdir -p "$PROJECT/data/intermediate"

cd "$PROJECT"

python -m aksu.data.build.autolabel \
    --input  data/intermediate/unique_tokens.jsonl \
    --output data/intermediate/autolabel_pilot.jsonl \
    --shard-index 0 \
    --shard-size 100000 \
    --ensemble-ckpts \
        models/v6/disambiguator/best_model.pt \
        models/v6/disambiguator_s123/best_model.pt \
        models/v6/disambiguator_s456/best_model.pt \
        models/v6/disambiguator_s789/best_model.pt \
        models/v6/disambiguator_s1337/best_model.pt \
    --sentence-index data/intermediate/token_sentences.jsonl \
    --sentences-db  data/intermediate/sentences.jsonl

echo "Pilot complete. Extrapolate yield with:"
echo "python scripts/data/extrapolate_yield.py \\"
echo "    --pilot-output data/intermediate/autolabel_pilot.jsonl \\"
echo "    --target 2500000 \\"
echo "    --output audit/benchmark_results/yield_extrapolation.json"
