#!/bin/bash
# Preprocess OSCAR-tr shard → tokens.jsonl + sentences.jsonl + token_sentences.jsonl
# Then deduplicate → unique_tokens.jsonl (input for autolabel pilot).
#
# TRUBA compute nodes do NOT have internet access. Pre-stage data on the login node:
#   python scripts/data/download_oscar_pilot.py \
#       --out /arf/scratch/scolakoglu/oscar-tr-pilot.jsonl \
#       --max-sentences 500000
#
# Then submit with the pre-staged path:
#   sbatch scripts/truba/submit_preprocess_aksu.sh \
#       --local-jsonl /arf/scratch/scolakoglu/oscar-tr-pilot.jsonl
#
# Without --local-jsonl the job attempts to stream from HuggingFace,
# which will fail if the compute node has no outbound internet.
#
# Must complete before submit_autolabel_pilot.sh.

#SBATCH --job-name=aksu-preprocess
#SBATCH --partition=orfoz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 56
#SBATCH --time=12:00:00
#SBATCH --output=/arf/scratch/scolakoglu/logs/preprocess_%j.out
#SBATCH --error=/arf/scratch/scolakoglu/logs/preprocess_%j.err

set -euo pipefail

PROJECT=/arf/home/scolakoglu/NLP_Project
module load comp/python/miniconda3
source "$PROJECT/.venv/bin/activate"

mkdir -p /arf/scratch/scolakoglu/logs
mkdir -p "$PROJECT/data/intermediate"

cd "$PROJECT"

echo "=== Preprocessing OSCAR-tr shard ==="
echo "Date: $(date) | Node: $(hostname) | CPUs: $(nproc)"

# Accept optional --local-jsonl from sbatch args (passed after script name).
LOCAL_JSONL_ARG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --local-jsonl) LOCAL_JSONL_ARG="--local-jsonl $2"; shift 2 ;;
        *) shift ;;
    esac
done

python -m aksu.data.build.preprocess \
    --shard oscar-tr \
    --max-tokens 12000000 \
    --output-dir data/intermediate \
    $LOCAL_JSONL_ARG

echo "Preprocessing done: $(date)"

echo "=== Deduplicating tokens ==="
python scripts/data/dedup_tokens.py \
    --input  data/intermediate/tokens.jsonl \
    --output data/intermediate/unique_tokens.jsonl

echo "Dedup done: $(date)"
echo "unique_tokens.jsonl ready for autolabel pilot"
echo ""
echo "Next step:"
echo "  sbatch scripts/truba/submit_autolabel_pilot.sh"
