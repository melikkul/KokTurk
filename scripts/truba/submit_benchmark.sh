#!/bin/bash
#SBATCH -p orfoz
#SBATCH -A $SLURM_ACCOUNT
#SBATCH -J morpho_bench
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 56
#SBATCH --time=0-06:00:00
#SBATCH --mem=64G
#SBATCH --output=$SCRATCH_DIR/logs/benchmark_%j.out
#SBATCH --error=$SCRATCH_DIR/logs/benchmark_%j.err

set -euo pipefail

module load comp/python/miniconda3

cd $PROJECT_DIR
source .venv/bin/activate
export PYTHONPATH=src

echo "=== TTC-3600 Full Benchmark ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

python3 -c "
import json, os, sys, time, logging
import numpy as np
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', stream=sys.stderr)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
# Step 1: Load TTC-3600 raw text
# ══════════════════════════════════════════════════════════════

logger.info('Loading TTC-3600 raw text...')
BASE = Path('data/external/ttc3600_raw/TTC-3600_Orj')
CATS = ['ekonomi', 'kultursanat', 'saglik', 'siyaset', 'spor', 'teknoloji']
docs = []
for cat in CATS:
    cat_dir = BASE / cat
    for fpath in sorted(cat_dir.glob('*.txt')):
        text = fpath.read_text(encoding='utf-8', errors='replace').strip()
        if text:
            docs.append({'text': text, 'label': cat, 'doc_id': f'{cat}_{fpath.stem}'})

logger.info('Loaded %d documents', len(docs))
dist = Counter(d['label'] for d in docs)
for cat in CATS:
    logger.info('  %s: %d', cat, dist[cat])

# Save real JSONL
real_path = Path('data/external/ttc3600/ttc3600_real.jsonl')
real_path.parent.mkdir(parents=True, exist_ok=True)
with open(real_path, 'w', encoding='utf-8') as f:
    for d in docs:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')

# ══════════════════════════════════════════════════════════════
# Step 2: Atomize all documents
# ══════════════════════════════════════════════════════════════

logger.info('Atomizing documents...')
from aksu.aksu.kokturk.core.analyzer import MorphoAnalyzer
analyzer = MorphoAnalyzer()

atomized_docs = []
total_tokens = 0
atomized_tokens = 0

for i, doc in enumerate(docs):
    words = doc['text'].split()
    parts = []
    for w in words:
        total_tokens += 1
        result = analyzer.analyze(w)
        best = result.best
        if best is not None:
            parts.append(best.to_str())
            atomized_tokens += 1
        else:
            parts.append(w)
    atomized_docs.append({
        'text': doc['text'],
        'text_atomized': ' '.join(parts),
        'label': doc['label'],
        'doc_id': doc['doc_id'],
    })
    if (i + 1) % 500 == 0:
        logger.info('  Atomized %d/%d docs', i + 1, len(docs))

analyzer.close()
coverage = atomized_tokens / max(total_tokens, 1)
avg_tokens = total_tokens / max(len(docs), 1)
logger.info('Atomization: %d/%d tokens (%.1f%% coverage), avg %.0f tok/doc',
            atomized_tokens, total_tokens, 100*coverage, avg_tokens)

# Save atomized
atom_path = Path('data/external/ttc3600/ttc3600_atomized_real.jsonl')
with open(atom_path, 'w', encoding='utf-8') as f:
    for d in atomized_docs:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')

# ══════════════════════════════════════════════════════════════
# Step 3: Classification baselines (5-fold CV)
# ══════════════════════════════════════════════════════════════

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.info('XGBoost not available, using LogReg only')

texts_raw = [d['text'] for d in atomized_docs]
texts_atom = [d['text_atomized'] for d in atomized_docs]
labels = [d['label'] for d in atomized_docs]
label_names = sorted(set(labels))
label2idx = {name: i for i, name in enumerate(label_names)}
y = np.array([label2idx[lb] for lb in labels])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}
all_fold_preds = {}  # for significance testing

def run_tfidf_clf(name, texts, clf_class, clf_kwargs):
    logger.info('Running: %s...', name)
    f1s = []
    fold_preds = np.zeros(len(y), dtype=int)
    for train_idx, test_idx in skf.split(texts, y):
        tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), sublinear_tf=True)
        x_tr = tfidf.fit_transform([texts[i] for i in train_idx])
        x_te = tfidf.transform([texts[i] for i in test_idx])
        clf = clf_class(**clf_kwargs)
        clf.fit(x_tr, y[train_idx])
        preds = clf.predict(x_te)
        fold_preds[test_idx] = preds
        f1s.append(f1_score(y[test_idx], preds, average='macro'))
    results[name] = {'macro_f1_mean': float(np.mean(f1s)), 'macro_f1_std': float(np.std(f1s))}
    all_fold_preds[name] = fold_preds.tolist()
    logger.info('  %s: F1=%.3f ± %.3f', name, np.mean(f1s), np.std(f1s))

# Baseline 1: LogReg on raw text
run_tfidf_clf('LogReg_raw', texts_raw, LogisticRegression, {'max_iter': 1000, 'C': 1.0, 'random_state': 42})

# Baseline 2: LogReg on atomized text (OUR METHOD)
run_tfidf_clf('LogReg_atomized', texts_atom, LogisticRegression, {'max_iter': 1000, 'C': 1.0, 'random_state': 42})

# Baseline 3: XGBoost on raw text
if HAS_XGBOOST:
    run_tfidf_clf('XGBoost_raw', texts_raw, XGBClassifier,
                  {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.1,
                   'random_state': 42, 'eval_metric': 'mlogloss', 'verbosity': 0})
    run_tfidf_clf('XGBoost_atomized', texts_atom, XGBClassifier,
                  {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.1,
                   'random_state': 42, 'eval_metric': 'mlogloss', 'verbosity': 0})

# ══════════════════════════════════════════════════════════════
# Step 4: BERTurk (if downloadable)
# ══════════════════════════════════════════════════════════════

try:
    logger.info('Trying BERTurk download...')
    from transformers import AutoModel, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')
    bert_model = AutoModel.from_pretrained('dbmdz/bert-base-turkish-cased')
    bert_model.eval()

    logger.info('BERTurk loaded! Extracting [CLS] embeddings...')
    embeddings = []
    with torch.no_grad():
        for i, doc in enumerate(docs):
            enc = tokenizer(doc['text'][:512], return_tensors='pt', truncation=True, padding=True, max_length=512)
            out = bert_model(**enc)
            cls_emb = out.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(cls_emb)
            if (i + 1) % 500 == 0:
                logger.info('  BERTurk embeddings: %d/%d', i + 1, len(docs))

    x_bert = np.array(embeddings)
    logger.info('BERTurk embeddings: shape=%s', x_bert.shape)

    f1s_bert = []
    bert_preds = np.zeros(len(y), dtype=int)
    for train_idx, test_idx in skf.split(x_bert, y):
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        clf.fit(x_bert[train_idx], y[train_idx])
        preds = clf.predict(x_bert[test_idx])
        bert_preds[test_idx] = preds
        f1s_bert.append(f1_score(y[test_idx], preds, average='macro'))
    results['BERTurk_frozen'] = {'macro_f1_mean': float(np.mean(f1s_bert)), 'macro_f1_std': float(np.std(f1s_bert))}
    all_fold_preds['BERTurk_frozen'] = bert_preds.tolist()
    logger.info('  BERTurk frozen: F1=%.3f ± %.3f', np.mean(f1s_bert), np.std(f1s_bert))
except Exception as e:
    logger.warning('BERTurk unavailable: %s', e)

# ══════════════════════════════════════════════════════════════
# Step 5: Statistical significance
# ══════════════════════════════════════════════════════════════

logger.info('Running significance tests...')
from aksu.benchmark.significance import paired_bootstrap_test, holm_bonferroni_correction

our_method = 'LogReg_atomized'
if our_method not in all_fold_preds:
    logger.warning('Our method not in predictions')
else:
    our_preds = all_fold_preds[our_method]
    comparisons = {}
    p_values = []
    baseline_names = []

    for name, preds in all_fold_preds.items():
        if name == our_method:
            continue
        result = paired_bootstrap_test(our_preds, preds, y.tolist())
        comparisons[name] = result
        p_values.append(result['p_value'])
        baseline_names.append(name)
        logger.info('  vs %s: p=%.4f, diff=%.4f, d=%.3f',
                    name, result['p_value'], result['mean_diff'], result['cohens_d'])

    if p_values:
        corrected = holm_bonferroni_correction(p_values)
        for name, p_corr in zip(baseline_names, corrected):
            comparisons[name]['p_corrected'] = p_corr
            logger.info('  vs %s (corrected): p=%.4f', name, p_corr)

# ══════════════════════════════════════════════════════════════
# Step 6: Intrinsic eval
# ══════════════════════════════════════════════════════════════

logger.info('Running intrinsic evaluation...')
import torch
from aksu.aksu.kokturk.models.char_gru import MorphAtomizer
from aksu.train.datasets import EOS_IDX, TieredCorpusDataset, Vocab
from torch.utils.data import DataLoader

char_vocab = Vocab.load(Path('models/vocabs/char_vocab.json'))
tag_vocab = Vocab.load(Path('models/vocabs/tag_vocab.json'))
ckpt = torch.load('models/atomizer_v2/best_model.pt', weights_only=True, map_location='cpu')
model = MorphAtomizer(ckpt['char_vocab_size'], ckpt['tag_vocab_size'],
                      ckpt['embed_dim'], ckpt['hidden_dim'], ckpt['num_layers'])
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

test_ds = TieredCorpusDataset(Path('data/splits/test.jsonl'), char_vocab, tag_vocab)
loader = DataLoader(test_ds, batch_size=64, shuffle=False)

all_p, all_g = [], []
root_err, tag_err = 0, 0
with torch.no_grad():
    for chars, tags, *_ in loader:
        preds = model.greedy_decode(chars)
        for i in range(chars.size(0)):
            ps = [idx for idx in preds[i].tolist() if idx == EOS_IDX or idx > 3]
            gs = [idx for idx in tags[i].tolist() if idx == EOS_IDX or idx > 3]
            pt = [idx for idx in preds[i].tolist() if idx > 3]
            gt = [idx for idx in tags[i].tolist() if idx > 3 and idx != 2]
            # Trim at EOS
            ptrim, gtrim = [], []
            for idx in preds[i].tolist():
                if idx == 2: break
                if idx > 3: ptrim.append(idx)
            for idx in tags[i].tolist():
                if idx == 2: break
                if idx > 3: gtrim.append(idx)
            all_p.append(ptrim)
            all_g.append(gtrim)
            if ptrim != gtrim:
                if (ptrim[0] if ptrim else -1) != (gtrim[0] if gtrim else -2):
                    root_err += 1
                else:
                    tag_err += 1

from aksu.benchmark.intrinsic_eval import compute_all_metrics
intrinsic = compute_all_metrics(all_p, all_g)
intrinsic['root_errors'] = root_err
intrinsic['tag_errors'] = tag_err
intrinsic['total_test_tokens'] = len(all_p)
logger.info('Test EM=%.3f Root=%.3f F1=%.3f', intrinsic['exact_match'], intrinsic['root_accuracy'], intrinsic['f1'])

# ══════════════════════════════════════════════════════════════
# Step 7: Efficiency
# ══════════════════════════════════════════════════════════════

logger.info('Measuring efficiency...')
analyzer2 = MorphoAnalyzer()
test_words = ['evlerinden', 'gidiyorum', 'kitabı', 'güzel', 'Ankara'] * 200
t0 = time.time()
for w in test_words:
    analyzer2.analyze(w)
zeyrek_speed = len(test_words) / (time.time() - t0)
analyzer2.close()
logger.info('  Zeyrek: %.0f tok/s', zeyrek_speed)

efficiency = {
    'zeyrek': {'tokens_per_sec': zeyrek_speed, 'model_size_mb': 0},
    'gru_atomizer': {'model_size_mb': os.path.getsize('models/atomizer_v2/best_model.pt')/(1024*1024), 'params': 2255487},
}

# ══════════════════════════════════════════════════════════════
# Step 8: BPE analysis
# ══════════════════════════════════════════════════════════════

from aksu.benchmark.linguistic_analysis import get_failure_examples
bpe_examples = get_failure_examples()

# ══════════════════════════════════════════════════════════════
# Step 9: Generate report
# ══════════════════════════════════════════════════════════════

logger.info('Generating report...')
output_dir = Path('models/benchmark')
output_dir.mkdir(parents=True, exist_ok=True)

# Save full results JSON
with open(output_dir / 'classification_results.json', 'w') as f:
    json.dump({
        'intrinsic': {k: float(v) for k, v in intrinsic.items()},
        'classification': results,
        'efficiency': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in efficiency.items()},
        'significance': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in comparisons.items()} if 'comparisons' in dir() else {},
        'bpe_failure_count': len(bpe_examples),
        'atomization_coverage': coverage,
        'total_documents': len(docs),
    }, f, indent=2)

# Generate markdown report
lines = ['# Benchmark Report: Neural Morphological Atomization for Turkish\n']

lines.append('## 1. Morphological Analysis (Intrinsic)\n')
lines.append('### GRU Seq2Seq (2.25M params, 8.6 MB)\n')
lines.append('| Metric | Score |')
lines.append('|--------|-------|')
lines.append(f'| Exact Match | {intrinsic[\"exact_match\"]:.1%} |')
lines.append(f'| Root Accuracy | {intrinsic[\"root_accuracy\"]:.1%} |')
lines.append(f'| Tag F1 | {intrinsic[\"f1\"]:.1%} |')
lines.append(f'| Tag Precision | {intrinsic[\"precision\"]:.1%} |')
lines.append(f'| Tag Recall | {intrinsic[\"recall\"]:.1%} |')
lines.append(f'\nTest: {intrinsic[\"total_test_tokens\"]} tokens. Errors: {root_err} root, {tag_err} tag.\n')

lines.append('## 2. TTC-3600 Classification (5-fold CV, 3,600 docs)\n')
lines.append('| Method | Macro-F1 | vs Atomized (p) |')
lines.append('|--------|----------|-----------------|')
for name in ['LogReg_raw', 'LogReg_atomized', 'XGBoost_raw', 'XGBoost_atomized', 'BERTurk_frozen']:
    r = results.get(name, {})
    if not r:
        continue
    f1_str = f'{r[\"macro_f1_mean\"]:.3f} ± {r[\"macro_f1_std\"]:.3f}'
    sig = comparisons.get(name, {}) if 'comparisons' in dir() else {}
    p_str = f'{sig[\"p_corrected\"]:.4f}' if 'p_corrected' in sig else '—'
    marker = ' **← ours**' if 'atomized' in name.lower() and 'xgb' not in name.lower() else ''
    lines.append(f'| {name}{marker} | {f1_str} | {p_str} |')

lines.append(f'\nAtomization coverage: {coverage:.1%} ({total_tokens} tokens, {len(docs)} documents)\n')

lines.append('## 3. Efficiency\n')
lines.append('| System | Speed | Size |')
lines.append('|--------|-------|------|')
lines.append(f'| Zeyrek | {zeyrek_speed:.0f} tok/s | rule-based |')
lines.append(f'| GRU Atomizer | — | {efficiency[\"gru_atomizer\"][\"model_size_mb\"]:.1f} MB |')
lines.append('')

lines.append('## 4. BPE Failure Analysis\n')
lines.append(f'{len(bpe_examples)} cases where BPE fragments Turkish morphemes.\n')

with open(output_dir / 'BENCHMARK_REPORT.md', 'w') as f:
    f.write('\n'.join(lines))

logger.info('Report: models/benchmark/BENCHMARK_REPORT.md')
logger.info('Results: models/benchmark/classification_results.json')

# Print summary
print()
print('='*65)
print('BENCHMARK COMPLETE')
print('='*65)
print(f'Test EM: {intrinsic[\"exact_match\"]:.1%}')
print(f'Test Root: {intrinsic[\"root_accuracy\"]:.1%}')
print(f'Test Tag F1: {intrinsic[\"f1\"]:.1%}')
print()
for name, r in sorted(results.items()):
    print(f'{name:25s} F1={r[\"macro_f1_mean\"]:.3f} ± {r[\"macro_f1_std\"]:.3f}')
print(f'Atomization coverage: {coverage:.1%}')
print(f'BPE failures: {len(bpe_examples)}')
print('='*65)
"

echo "=== Benchmark complete: $(date) ==="
