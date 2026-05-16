#!/bin/bash
#SBATCH -p orfoz
#SBATCH -A $SLURM_ACCOUNT
#SBATCH -J v6_eval
#SBATCH -N 1 -n 1 -c 56
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=$SCRATCH_DIR/logs/v6_eval_%j.out
#SBATCH --error=$SCRATCH_DIR/logs/v6_eval_%j.err

set -euo pipefail
module load comp/python/miniconda3
cd $PROJECT_DIR
source .venv/bin/activate
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "=== v6 Disambiguation Evaluation ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Date: $(date)"

python -c "
import json, logging, os, torch
from pathlib import Path
from torch.utils.data import DataLoader

from aksu.train.datasets import Vocab
from aksu.train.disambiguation_dataset import DisambiguationDataset, disambiguation_collate
from aksu.train.train_disambiguator import evaluate, pre_cache_bert_embeddings
from aksu.aksu.kokturk.models.disambiguator import BERTurkDisambiguator

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

tag_vocab = Vocab.load(Path('models/vocabs/tag_vocab.json'))

# Load model with skip_bert_loading (checkpoint has no BERT weights)
ckpt = torch.load('models/v6/disambiguator/best_model.pt', map_location='cpu')
model = BERTurkDisambiguator(
    tag_vocab_size=ckpt.get('tag_vocab_size', len(tag_vocab)),
    skip_bert_loading=True,
)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

print(f'Model: epoch {ckpt[\"epoch\"]}, val EM {ckpt[\"val_em\"]*100:.1f}%, val ambig EM {ckpt[\"val_ambig_em\"]*100:.1f}%')

# Load shared BERT once for all caching
from transformers import AutoModel, AutoTokenizer
bert = AutoModel.from_pretrained('models/berturk')
bert.eval()
for p in bert.parameters():
    p.requires_grad = False
tok = AutoTokenizer.from_pretrained('models/berturk')

cache_dir = Path(os.environ.get('SCRATCH_DIR', '/tmp')) / 'bert_cache'

results = {}

for split_name, split_path in [('test', 'data/splits/test.jsonl'), ('val', 'data/splits/val.jsonl')]:
    print(f'\\n=== {split_name.upper()} SET ===')
    ds = DisambiguationDataset(split_path, tag_vocab)
    cache = pre_cache_bert_embeddings(
        ds, 'models/berturk',
        cache_path=cache_dir / f'{split_name}_bert_cache.pt',
        shared_bert=bert, shared_tokenizer=tok,
    )
    loader = DataLoader(ds, batch_size=128, shuffle=False, collate_fn=disambiguation_collate)
    m = evaluate(model, loader, 'cpu', bert_cache=cache)

    print(f'Overall EM:       {m[\"overall_em\"]*100:.1f}%')
    print(f'Ambiguous EM:     {m[\"ambiguous_em\"]*100:.1f}%')
    print(f'Total tokens:     {m[\"total\"]}')
    print(f'Ambiguous tokens: {m[\"ambiguous_total\"]}')

    results[split_name] = {
        'overall_em': round(m['overall_em'], 4),
        'ambiguous_em': round(m['ambiguous_em'], 4),
        'total_tokens': m['total'],
        'ambiguous_tokens': m['ambiguous_total'],
    }

results['model'] = 'v6_disambiguator'
results['best_epoch'] = ckpt['epoch']
results['trainable_params'] = sum(p.numel() for p in model.parameters())

with open('models/v6/disambiguator/test_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\\nResults saved to models/v6/disambiguator/test_results.json')
print(f'\\n=== SUMMARY ===')
print(f'Val EM:  {results[\"val\"][\"overall_em\"]*100:.1f}% (ambig: {results[\"val\"][\"ambiguous_em\"]*100:.1f}%)')
print(f'Test EM: {results[\"test\"][\"overall_em\"]*100:.1f}% (ambig: {results[\"test\"][\"ambiguous_em\"]*100:.1f}%)')
"

echo "=== Done: $(date) ==="
