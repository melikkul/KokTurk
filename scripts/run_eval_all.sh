#!/bin/bash
# Run eval for all 25 models with adaptive sample counts (CPU time limit constraint)
cd $PROJECT_DIR

OUTFILE="eval_results.tsv"
echo -e "Model\tClass\tEM\tRoot_Acc\tTag_F1\tStored_EM\tParams\tN_samples" > "$OUTFILE"

eval_model() {
  local model="$1"
  local samples="$2"
  local short=$(echo "$model" | sed 's|models/||;s|/best_model.pt||')
  echo -n "[$3/25] $short (n=$samples) ... " >&2
  result=$(PYTHONPATH=src .venv/bin/python scripts/eval_one_model.py "$model" "$samples" 2>/dev/null)
  if [ $? -eq 0 ]; then
    echo "$result"$'\t'"$samples" >> "$OUTFILE"
    echo "done" >&2
  else
    echo "FAILED" >&2
    echo -e "${short}\tERROR\tERROR\tERROR\tERROR\tN/A\tN/A\t${samples}" >> "$OUTFILE"
  fi
}

# seq2seq models: 100 samples
eval_model "models/draft_v1/best_model.pt" 100 1
eval_model "models/atomizer_v2/best_model.pt" 100 2
eval_model "models/atomizer_v3/best_model.pt" 50 3
eval_model "models/ensemble/model_seed42/best_model.pt" 100 4
eval_model "models/ensemble/model_seed123/best_model.pt" 100 5
eval_model "models/ensemble/model_seed456/best_model.pt" 100 6
eval_model "models/ensemble/model_seed789/best_model.pt" 100 7
eval_model "models/ensemble/model_seed1337/best_model.pt" 100 8

# dual_head models: 50 samples (safer)
eval_model "models/noise_sweep/noise_0.0_fixed/best_model.pt" 50 9
eval_model "models/noise_sweep/noise_0.0_taac/best_model.pt" 50 10
eval_model "models/noise_sweep/noise_0.05_fixed/best_model.pt" 50 11
eval_model "models/noise_sweep/noise_0.05_taac/best_model.pt" 50 12
eval_model "models/noise_sweep/noise_0.10_fixed/best_model.pt" 50 13
eval_model "models/noise_sweep/noise_0.10_taac/best_model.pt" 50 14
eval_model "models/noise_sweep/noise_0.20_fixed/best_model.pt" 50 15
eval_model "models/noise_sweep/noise_0.20_taac/best_model.pt" 50 16
eval_model "models/ablations/baseline_seq2seq/best_model.pt" 50 17
eval_model "models/ablations/dual_head/best_model.pt" 50 18
eval_model "models/ablations/dual_head_taac/best_model.pt" 50 19

# contextual models: 25 samples
eval_model "models/ablations/dual_head_context/best_model.pt" 25 20
eval_model "models/ablations/full_v4/best_model.pt" 25 21
eval_model "models/v4/best_model.pt" 25 22
eval_model "models/taac_retest/id3/best_model.pt" 50 23
eval_model "models/taac_retest/id4/best_model.pt" 25 24

# BERTurk model: 10 samples (very heavy)
eval_model "models/v4_berturk/best_model.pt" 10 25

echo "" >&2
echo "=== Results (sorted by EM, descending) ===" >&2
# Print sorted results
head -1 "$OUTFILE"
tail -n +2 "$OUTFILE" | sort -t$'\t' -k3 -rn
