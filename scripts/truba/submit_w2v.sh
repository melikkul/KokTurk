#!/bin/bash
#SBATCH --job-name=tr_w2v
#SBATCH --account=$SLURM_ACCOUNT
#SBATCH --partition=orfoz
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=56
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=$SCRATCH_DIR/logs/w2v_%j.out
#SBATCH --error=$SCRATCH_DIR/logs/w2v_%j.err

set -euo pipefail

PROJECT_DIR=$PROJECT_DIR
SCRATCH_DIR=$SCRATCH_DIR/nlp_w2v

echo "=== TR Word2Vec Training ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Started: $(date)"

module load comp/python/miniconda3
source "$PROJECT_DIR/.venv/bin/activate"

mkdir -p "$SCRATCH_DIR"
mkdir -p $SCRATCH_DIR/logs
mkdir -p "$PROJECT_DIR/models/word2vec"

cd "$PROJECT_DIR"

PYTHONPATH="$PROJECT_DIR/src" python src/resource/train_word2vec.py \
    --jsonl_dirs data/splits data/resource \
    --conllu_dirs data/external/boun_treebank data/external/imst_treebank \
    --output models/word2vec/tr_word2vec_128.bin \
    --dim 128 \
    --window 5 \
    --min_count 5 \
    --workers 56 \
    --epochs 10

echo "Finished: $(date)"
echo "Output: $PROJECT_DIR/models/word2vec/tr_word2vec_128.bin"
