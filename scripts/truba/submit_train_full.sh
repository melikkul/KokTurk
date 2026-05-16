#!/bin/bash
#SBATCH -p orfoz
#SBATCH -A $SLURM_ACCOUNT
#SBATCH -J morpho_full
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 56
#SBATCH --time=0-06:00:00
#SBATCH --mem=64G
#SBATCH --output=$SCRATCH_DIR/logs/train_full_%j.out
#SBATCH --error=$SCRATCH_DIR/logs/train_full_%j.err

set -euo pipefail

module load comp/python/miniconda3

cd $PROJECT_DIR
source .venv/bin/activate

echo "=== Starting full corpus training ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $(nproc)"

export PYTHONPATH=src
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

python -c "
import logging, sys, time, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from aksu.aksu.kokturk.models.char_gru import MorphAtomizer
from aksu.train.datasets import TieredCorpusDataset, Vocab

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', stream=sys.stderr)
logger = logging.getLogger(__name__)
torch.manual_seed(42)

char_vocab = Vocab.load(Path('models/vocabs/char_vocab.json'))
tag_vocab = Vocab.load(Path('models/vocabs/tag_vocab.json'))
logger.info('Vocabs: %d chars, %d tags', len(char_vocab), len(tag_vocab))

train_ds = TieredCorpusDataset(Path('data/splits/train.jsonl'), char_vocab, tag_vocab, gold_weight=2.0)
val_ds = TieredCorpusDataset(Path('data/splits/val.jsonl'), char_vocab, tag_vocab)
logger.info('Train: %d samples %s', len(train_ds), train_ds.tier_counts)
logger.info('Val: %d samples %s', len(val_ds), val_ds.tier_counts)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

model = MorphAtomizer(len(char_vocab), len(tag_vocab), embed_dim=64, hidden_dim=128, num_layers=2, dropout=0.3)
logger.info('Params: %d', model.count_parameters())

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
criterion = nn.CrossEntropyLoss(ignore_index=0)
output_dir = Path('models/atomizer_v2')
output_dir.mkdir(parents=True, exist_ok=True)
best_em = 0.0

for epoch in range(30):
    tf = 0.5 - 0.5 * epoch / 29
    model.train()
    total_loss, n_batch = 0.0, 0
    t0 = time.time()
    for chars, tags, *_ in train_loader:
        optimizer.zero_grad()
        logits = model(chars, tags, teacher_forcing_ratio=tf)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tags.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batch += 1

    # Validate
    model.eval()
    em, root_ok, total = 0, 0, 0
    with torch.no_grad():
        for chars, tags, *_ in val_loader:
            preds = model.greedy_decode(chars)
            for i in range(chars.size(0)):
                pred_seq = [idx for idx in preds[i].tolist() if idx > 3]
                gold_seq = [idx for idx in tags[i].tolist() if idx > 3 and idx != 2]
                # Trim at EOS
                pred_trim = []
                for idx in preds[i].tolist():
                    if idx == 2: break
                    if idx > 3: pred_trim.append(idx)
                gold_trim = []
                for idx in tags[i].tolist():
                    if idx == 2: break
                    if idx > 3: gold_trim.append(idx)
                if pred_trim == gold_trim: em += 1
                if pred_trim and gold_trim and pred_trim[0] == gold_trim[0]: root_ok += 1
                total += 1

    em_rate = em / max(total, 1)
    root_rate = root_ok / max(total, 1)
    elapsed = time.time() - t0
    logger.info('Epoch %2d/30  loss=%.4f  EM=%.3f  root=%.3f  tf=%.2f  (%.1fs)',
                epoch+1, total_loss/max(n_batch,1), em_rate, root_rate, tf, elapsed)

    if em_rate > best_em:
        best_em = em_rate
        torch.save({'model_state_dict': model.state_dict(),
                    'char_vocab_size': len(char_vocab), 'tag_vocab_size': len(tag_vocab),
                    'embed_dim': 64, 'hidden_dim': 128, 'num_layers': 2,
                    'epoch': epoch+1, 'best_em': best_em,
                    'metrics': {'exact_match': em_rate, 'root_accuracy': root_rate}},
                   output_dir / 'best_model.pt')
        logger.info('  -> New best (EM=%.3f)', best_em)

logger.info('Training complete. Best EM: %.3f', best_em)
print(f'Best model: {output_dir}/best_model.pt')
"

echo "=== Training complete: $(date) ==="
