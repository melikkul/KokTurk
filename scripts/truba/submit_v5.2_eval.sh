#!/bin/bash
#SBATCH -p orfoz
#SBATCH -A $SLURM_ACCOUNT
#SBATCH -J v5.2_eval
#SBATCH -N 1 -n 1 -c 56
#SBATCH --time=0-01:00:00
#SBATCH --output=$SCRATCH_DIR/logs/v5.2_eval_%j.out
#SBATCH --error=$SCRATCH_DIR/logs/v5.2_eval_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
module load comp/python/miniconda3
cd $PROJECT_DIR
source .venv/bin/activate
export PYTHONPATH=src

echo "=== v5.2 Evaluation: $(date) ==="

python scripts/eval_v4_models.py \
    --test-data data/splits/test.jsonl \
    --char-vocab models/vocabs/char_vocab.json \
    --tag-vocab models/vocabs/tag_vocab.json \
    --root-vocab models/vocabs/root_vocab.json \
    --models \
        "prev_best_dh_80K=models/ablation/dual_head/best_model.pt" \
        "v5.2_dh=models/v5.2/dh/best_model.pt" \
        "v5.2_rdrop=models/v5.2/dh_rdrop/best_model.pt" \
    2>&1 || true

# Fallback: manual eval if the above script doesn't exist or fails
echo ""
echo "=== Manual eval ==="
python -c "
import torch, json, sys, os
sys.path.insert(0, 'src')

from aksu.train.datasets import TieredCorpusDataset
from aksu.aksu.kokturk.models.dual_head import DualHeadAtomizer

char_vocab = json.load(open('models/vocabs/char_vocab.json'))
tag_vocab = json.load(open('models/vocabs/tag_vocab.json'))
root_vocab = json.load(open('models/vocabs/root_vocab.json'))

# Build reverse mappings
if isinstance(tag_vocab, dict):
    idx2tag = {v: k for k, v in tag_vocab.items()}
elif isinstance(tag_vocab, list):
    idx2tag = {i: t for i, t in enumerate(tag_vocab)}

if isinstance(root_vocab, dict):
    idx2root = {v: k for k, v in root_vocab.items()}
elif isinstance(root_vocab, list):
    idx2root = {i: r for i, r in enumerate(root_vocab)}

# Load test data
test_ds = TieredCorpusDataset('data/splits/test.jsonl', char_vocab, tag_vocab, root_vocab=root_vocab)
print(f'Test set: {len(test_ds)} samples')

models = {
    'prev_best_dh_80K': 'models/ablation/dual_head/best_model.pt',
    'v5.2_dh':          'models/v5.2/dh/best_model.pt',
    'v5.2_rdrop':       'models/v5.2/dh_rdrop/best_model.pt',
}

print(f'{\"Model\":30s} {\"Test EM\":>10s} {\"Root Acc\":>10s} {\"Tag F1\":>10s}')
print('=' * 65)

for name, path in models.items():
    if not os.path.exists(path):
        print(f'{name:30s} NOT FOUND')
        continue
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        state = ckpt.get('model', ckpt.get('model_state_dict', ckpt))

        # Determine root vocab size from checkpoint
        root_size = None
        tag_size = None
        for k, v in state.items():
            if 'root_head.fc2.bias' in k:
                root_size = v.shape[0]
            if 'tag_decoder.output_proj.bias' in k:
                tag_size = v.shape[0]

        if root_size != len(root_vocab if isinstance(root_vocab, list) else root_vocab):
            print(f'{name:30s} VOCAB MISMATCH (model root={root_size}, vocab={len(root_vocab)})')
            continue

        model = DualHeadAtomizer(
            char_vocab_size=len(char_vocab) if isinstance(char_vocab, dict) else len(char_vocab),
            tag_vocab_size=tag_size or (len(tag_vocab) if isinstance(tag_vocab, dict) else len(tag_vocab)),
            root_vocab_size=root_size,
        )
        model.load_state_dict(state)
        model.eval()

        # Evaluate
        correct = 0
        root_correct = 0
        total = 0

        with torch.no_grad():
            for i in range(len(test_ds)):
                chars, tags, tier, root_idx = test_ds[i][:4]
                chars = chars.unsqueeze(0)
                pred_label = model.greedy_decode(chars, idx2tag, idx2root)

                # Reconstruct gold label
                gold_tags = [idx2tag.get(t.item(), '<UNK>') for t in tags if t.item() not in [0, 1, 2, 3]]  # skip special
                gold_root = idx2root.get(root_idx.item() if hasattr(root_idx, 'item') else root_idx, '<UNK>')
                gold_label = gold_root + ' ' + ' '.join(gold_tags)

                if pred_label.strip() == gold_label.strip():
                    correct += 1
                if pred_label.split()[0] == gold_label.split()[0]:
                    root_correct += 1
                total += 1

        em = correct / total
        root_acc = root_correct / total
        print(f'{name:30s} {em:>9.1%} {root_acc:>9.1%}       —')
    except Exception as e:
        print(f'{name:30s} ERROR: {e}')
" 2>&1 || true

echo "=== Done: $(date) ==="
