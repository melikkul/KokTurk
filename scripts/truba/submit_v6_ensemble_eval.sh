#!/bin/bash
#SBATCH -p orfoz
#SBATCH -A $SLURM_ACCOUNT
#SBATCH -J v6_ens_eval
#SBATCH -N 1 -n 1 -c 56
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=$SCRATCH_DIR/logs/v6_ens_eval_%j.out
#SBATCH --error=$SCRATCH_DIR/logs/v6_ens_eval_%j.err

set -euo pipefail
module load comp/python/miniconda3
cd $PROJECT_DIR
source .venv/bin/activate
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "=== v6 Ensemble Evaluation ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Date: $(date)"

python -c "
import json, logging, os, torch
from collections import Counter
from pathlib import Path
from torch.utils.data import DataLoader

from aksu.train.datasets import Vocab
from aksu.train.disambiguation_dataset import DisambiguationDataset, disambiguation_collate
from aksu.train.train_disambiguator import evaluate, pre_cache_bert_embeddings
from aksu.aksu.kokturk.models.disambiguator import BERTurkDisambiguator
from aksu.aksu.kokturk.models.morphotactic_mask import (
    TAG_TO_CATEGORY, TRANSITIONS, CATEGORY_TO_NEXT_STATE, MorphState,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# ---- Load tag vocab ----
tag_vocab = Vocab.load(Path('models/vocabs/tag_vocab.json'))

# ---- Load test dataset + cache ----
test_ds = DisambiguationDataset('data/splits/test.jsonl', tag_vocab)

from transformers import AutoModel, AutoTokenizer
bert = AutoModel.from_pretrained('models/berturk')
bert.eval()
for p in bert.parameters():
    p.requires_grad = False
tok = AutoTokenizer.from_pretrained('models/berturk')

test_cache = pre_cache_bert_embeddings(
    test_ds, 'models/berturk',
    cache_path=Path(os.environ.get('SCRATCH_DIR', '/tmp')) / 'bert_cache' / 'test_bert_cache.pt',
    shared_bert=bert, shared_tokenizer=tok,
)
del bert, tok
import gc; gc.collect()

test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=disambiguation_collate)

# ---- Load all seed models ----
seed_dirs = {
    42: 'models/v6/disambiguator',
    123: 'models/v6/disambiguator_s123',
    456: 'models/v6/disambiguator_s456',
    789: 'models/v6/disambiguator_s789',
    1337: 'models/v6/disambiguator_s1337',
}

models = {}
for seed, model_dir in seed_dirs.items():
    ckpt_path = Path(model_dir) / 'best_model.pt'
    if not ckpt_path.exists():
        print(f'  SKIP seed={seed}: {ckpt_path} not found')
        continue
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model = BERTurkDisambiguator(
        tag_vocab_size=ckpt.get('tag_vocab_size', len(tag_vocab)),
        skip_bert_loading=True,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    models[seed] = model
    print(f'  Loaded seed={seed}: val EM={ckpt[\"val_em\"]*100:.1f}%, val ambig={ckpt[\"val_ambig_em\"]*100:.1f}%')

print(f'\\nLoaded {len(models)} models')

# ---- Step 1: Evaluate each seed individually ----
print('\\n' + '='*60)
print('INDIVIDUAL SEED RESULTS (test set)')
print('='*60)

seed_results = {}
for seed, model in sorted(models.items()):
    m = evaluate(model, test_loader, 'cpu', bert_cache=test_cache)
    seed_results[seed] = m
    print(f'  seed={seed:>5d}: EM={m[\"overall_em\"]*100:.1f}%  ambig={m[\"ambiguous_em\"]*100:.1f}%')

# ---- Step 2: Majority-vote ensemble ----
print('\\n' + '='*60)
print('MAJORITY-VOTE ENSEMBLE')
print('='*60)

# Collect per-sample predictions from each model
all_preds = {seed: [] for seed in models}

for batch in test_loader:
    for seed, model in models.items():
        from aksu.train.train_disambiguator import _get_cached_embeds
        cached_embeds = _get_cached_embeds(test_cache, batch['sample_indices'])
        with torch.no_grad():
            logits, _ = model(
                sentence_texts=batch['sentence_texts'],
                target_positions=batch['target_positions'],
                candidate_ids=batch['candidate_ids'],
                candidate_mask=batch['candidate_mask'],
                cached_bert_embeds=cached_embeds,
            )
        preds = logits.argmax(dim=-1)
        all_preds[seed].extend(preds.tolist())

N = len(test_ds)
seeds = sorted(models.keys())

# Majority vote
ensemble_correct = 0
ensemble_ambig_correct = 0
ambig_total = 0

for i in range(N):
    sample = test_ds.samples[i]
    gold_idx = sample['gold_idx']
    num_cand = sample['num_candidates']

    votes = [all_preds[s][i] for s in seeds]
    counter = Counter(votes)
    majority_pred = counter.most_common(1)[0][0]

    if majority_pred == gold_idx:
        ensemble_correct += 1

    if num_cand > 1:
        ambig_total += 1
        if majority_pred == gold_idx:
            ensemble_ambig_correct += 1

ens_em = ensemble_correct / N
ens_ambig_em = ensemble_ambig_correct / ambig_total if ambig_total > 0 else 0
print(f'  Ensemble EM:       {ens_em*100:.1f}%')
print(f'  Ensemble ambig EM: {ens_ambig_em*100:.1f}%')

# ---- Step 3: Ensemble + morphotactic mask ----
print('\\n' + '='*60)
print('ENSEMBLE + MORPHOTACTIC MASK')
print('='*60)

def is_morphotactically_valid(candidate_str):
    \"\"\"Check if a candidate parse follows morphotactic ordering.\"\"\"
    tags = candidate_str.split()
    if not tags:
        return True
    # First token is root — skip it
    suffix_tags = [t for t in tags[1:] if t.startswith('+')]
    state = MorphState.START
    for tag in suffix_tags:
        cat = TAG_TO_CATEGORY.get(tag, 'DERIV')
        allowed = TRANSITIONS.get(state, set())
        if cat not in allowed:
            return False
        state = CATEGORY_TO_NEXT_STATE.get(cat, state)
    return True

# For each sample: ensemble selects candidate, if invalid pick next-best valid
mask_correct = 0
mask_ambig_correct = 0
mask_fixes = 0

for batch_start in range(0, N, 128):
    batch_end = min(batch_start + 128, N)
    batch_indices = list(range(batch_start, batch_end))

    # Get logits from all models for this batch
    # We need to reconstruct the batch
    batch_items = [test_ds[i] for i in batch_indices]
    batch_data = disambiguation_collate(batch_items)

    # Average logits across models (soft ensemble)
    avg_logits = None
    for seed, model in models.items():
        cached_embeds = _get_cached_embeds(test_cache, batch_data['sample_indices'])
        with torch.no_grad():
            logits, _ = model(
                sentence_texts=batch_data['sentence_texts'],
                target_positions=batch_data['target_positions'],
                candidate_ids=batch_data['candidate_ids'],
                candidate_mask=batch_data['candidate_mask'],
                cached_bert_embeds=cached_embeds,
            )
        if avg_logits is None:
            avg_logits = logits.clone()
        else:
            avg_logits += logits
    avg_logits /= len(models)

    for j, i in enumerate(batch_indices):
        sample = test_ds.samples[i]
        gold_idx = sample['gold_idx']
        num_cand = sample['num_candidates']
        candidates = sample['candidates']

        # Sort candidates by averaged score (descending)
        scores = avg_logits[j, :num_cand]
        sorted_indices = scores.argsort(descending=True).tolist()

        # Pick highest-scoring morphotactically valid candidate
        pred_idx = sorted_indices[0]  # default: best score
        for idx in sorted_indices:
            if is_morphotactically_valid(candidates[idx]):
                if idx != sorted_indices[0]:
                    mask_fixes += 1
                pred_idx = idx
                break

        if pred_idx == gold_idx:
            mask_correct += 1

        if num_cand > 1:
            if pred_idx == gold_idx:
                mask_ambig_correct += 1

mask_em = mask_correct / N
mask_ambig_em = mask_ambig_correct / ambig_total if ambig_total > 0 else 0
print(f'  Ensemble+Mask EM:       {mask_em*100:.1f}%')
print(f'  Ensemble+Mask ambig EM: {mask_ambig_em*100:.1f}%')
print(f'  Morphotactic fixes:     {mask_fixes}')

# ---- Step 4: Final comparison table ----
print('\\n' + '='*60)
print('FINAL COMPARISON')
print('='*60)
print(f'{\"Model\":30s} {\"Test EM\":>10s} {\"Ambig EM\":>10s}')
print('-' * 52)
for seed in seeds:
    m = seed_results[seed]
    print(f'{\"v6 seed=\"+str(seed):30s} {m[\"overall_em\"]*100:>9.1f}% {m[\"ambiguous_em\"]*100:>9.1f}%')
print(f'{\"Ensemble (\" + str(len(models)) + \" seeds)\":30s} {ens_em*100:>9.1f}% {ens_ambig_em*100:>9.1f}%')
print(f'{\"Ensemble + mask\":30s} {mask_em*100:>9.1f}% {mask_ambig_em*100:>9.1f}%')
print(f'{\"MorseDisamb (SOTA)\":30s} {\"98.6%\":>10s} {\"—\":>10s}')
print('=' * 52)

# Save results
results = {
    'individual': {str(s): {
        'overall_em': round(seed_results[s]['overall_em'], 4),
        'ambiguous_em': round(seed_results[s]['ambiguous_em'], 4),
    } for s in seeds},
    'ensemble': {
        'overall_em': round(ens_em, 4),
        'ambiguous_em': round(ens_ambig_em, 4),
        'num_models': len(models),
    },
    'ensemble_mask': {
        'overall_em': round(mask_em, 4),
        'ambiguous_em': round(mask_ambig_em, 4),
        'morphotactic_fixes': mask_fixes,
    },
}
with open('models/v6/ensemble_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\\nSaved to models/v6/ensemble_results.json')
"

echo "=== Done: $(date) ==="
