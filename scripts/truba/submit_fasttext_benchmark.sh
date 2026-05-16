#!/bin/bash
#SBATCH -p orfoz
#SBATCH -A $SLURM_ACCOUNT
#SBATCH -J fasttext_bench
#SBATCH -N 1 -n 1 -c 56
#SBATCH --time=0-01:00:00
#SBATCH --output=$SCRATCH_DIR/logs/fasttext_bench_%j.out
#SBATCH --error=$SCRATCH_DIR/logs/fasttext_bench_%j.err

set -euo pipefail
module load comp/python/miniconda3
cd $PROJECT_DIR
source .venv/bin/activate
export PYTHONPATH=src

echo "Installing fasttext..."
pip install fasttext 2>&1 | tail -3

echo "Starting FastText benchmark on TTC-3600"
echo "Date: $(date)"

python -u -c "
import json, os, numpy as np, tempfile, shutil
from pathlib import Path
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

# Monkey-patch numpy for fasttext compatibility (NumPy 2.0)
_original_array = np.array
def _patched_array(*args, **kwargs):
    if 'copy' in kwargs and kwargs['copy'] is False:
        kwargs['copy'] = None  # NumPy 2.0 compatible
    return _original_array(*args, **kwargs)
np.array = _patched_array

import fasttext
fasttext.FastText.eprint = lambda x: None

# Load TTC-3600
CATEGORY_MAP = {
    'ekonomi': 'economy', 'kultursanat': 'culture_art',
    'saglik': 'health', 'siyaset': 'politics',
    'spor': 'sports', 'teknoloji': 'technology',
}
TTC_PATH = Path('data/external/ttc3600_raw/TTC-3600_Orj')
docs, labels, label_names = [], [], []
for cat_dir in sorted(TTC_PATH.iterdir()):
    if not cat_dir.is_dir(): continue
    cat_name = CATEGORY_MAP.get(cat_dir.name, cat_dir.name)
    if cat_name not in label_names: label_names.append(cat_name)
    cat_idx = label_names.index(cat_name)
    for f in sorted(cat_dir.glob('*.txt')):
        text = f.read_text(encoding='utf-8', errors='ignore').strip()
        if len(text) > 20:
            docs.append(text)
            labels.append(cat_idx)
labels = np.array(labels)
print(f'TTC-3600: {len(docs)} docs, {len(label_names)} classes')

# Restore original np.array for sklearn etc.
np.array = _original_array

# FastText supervised 5-fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
tmp = tempfile.mkdtemp()
ft_sup_scores, ft_sup_acc = [], []

for fold, (tr, te) in enumerate(skf.split(docs, labels)):
    ft_train = os.path.join(tmp, f'train_{fold}.txt')
    with open(ft_train, 'w', encoding='utf-8') as f:
        for i in tr:
            clean = docs[i].replace(chr(10), ' ').replace(chr(13), ' ')
            f.write(f'__label__{labels[i]} {clean}\n')

    # Patch np.array during fasttext calls
    np.array = _patched_array
    model = fasttext.train_supervised(
        ft_train, epoch=25, lr=0.5, wordNgrams=2,
        dim=100, loss='softmax', verbose=0,
    )
    np.array = _original_array

    preds = []
    for i in te:
        clean = docs[i].replace(chr(10), ' ').replace(chr(13), ' ')
        np.array = _patched_array
        pred_label = model.predict(clean)[0][0].replace('__label__', '')
        np.array = _original_array
        preds.append(int(pred_label))

    gold = labels[te]
    f1 = f1_score(gold, preds, average='macro')
    acc = accuracy_score(gold, preds)
    ft_sup_scores.append(f1)
    ft_sup_acc.append(acc)
    print(f'  FastText supervised fold {fold+1}: F1={f1:.4f} Acc={acc:.4f}')

mean_f1_sup = float(np.mean(ft_sup_scores))
std_f1_sup = float(np.std(ft_sup_scores))
mean_acc_sup = float(np.mean(ft_sup_acc))
print(f'FastText supervised: F1={mean_f1_sup:.4f}+/-{std_f1_sup:.4f} Acc={mean_acc_sup:.4f}')

# FastText embeddings + LogReg
corpus_file = os.path.join(tmp, 'corpus.txt')
with open(corpus_file, 'w', encoding='utf-8') as f:
    for doc in docs:
        f.write(doc.replace(chr(10), ' ') + '\n')

np.array = _patched_array
ft_emb = fasttext.train_unsupervised(corpus_file, model='skipgram', dim=100, epoch=10, verbose=0)
np.array = _original_array

def doc_vec(text):
    words = text.split()
    vecs = [ft_emb.get_word_vector(w) for w in words if w.strip()]
    return np.mean(vecs, axis=0) if vecs else np.zeros(100)

X_ft = np.array([doc_vec(doc) for doc in docs])
ft_emb_scores, ft_emb_acc = [], []

for fold, (tr, te) in enumerate(skf.split(X_ft, labels)):
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_ft[tr], labels[tr])
    preds = clf.predict(X_ft[te])
    f1 = f1_score(labels[te], preds, average='macro')
    acc = accuracy_score(labels[te], preds)
    ft_emb_scores.append(f1)
    ft_emb_acc.append(acc)
    print(f'  FastText emb+LogReg fold {fold+1}: F1={f1:.4f} Acc={acc:.4f}')

mean_f1_emb = float(np.mean(ft_emb_scores))
std_f1_emb = float(np.std(ft_emb_scores))
mean_acc_emb = float(np.mean(ft_emb_acc))
print(f'FastText emb+LogReg: F1={mean_f1_emb:.4f}+/-{std_f1_emb:.4f} Acc={mean_acc_emb:.4f}')

shutil.rmtree(tmp)

# Load existing results and merge
results = json.loads(open('models/benchmark/bert_fasttext_results.json').read())
results['fasttext_supervised'] = {'f1': mean_f1_sup, 'std': std_f1_sup, 'accuracy': mean_acc_sup}
results['fasttext_emb_logreg'] = {'f1': mean_f1_emb, 'std': std_f1_emb, 'accuracy': mean_acc_emb}

# Hybrid: atomized + fasttext
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack
from aksu.aksu.kokturk.core.analyzer import MorphoAnalyzer
import logging
logging.getLogger().setLevel(logging.ERROR)  # suppress zeyrek warnings

print('Atomizing docs for hybrid...')
analyzer = MorphoAnalyzer()
atomized = []
for i, doc in enumerate(docs):
    words = doc.split()[:300]
    parts = []
    for w in words:
        result = analyzer.analyze(w)
        best = result.best
        parts.append(best.to_str() if best is not None else w)
    atomized.append(' '.join(parts))
    if (i + 1) % 500 == 0: print(f'  Atomized: {i + 1}/{len(docs)}')
analyzer.close()

tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)
X_atom = tfidf.fit_transform(atomized)
X_hybrid_ft = hstack([X_atom, csr_matrix(X_ft)])

ft_hybrid_scores, ft_hybrid_acc = [], []
for fold, (tr, te) in enumerate(skf.split(X_hybrid_ft, labels)):
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_hybrid_ft[tr], labels[tr])
    preds = clf.predict(X_hybrid_ft[te])
    f1 = f1_score(labels[te], preds, average='macro')
    acc = accuracy_score(labels[te], preds)
    ft_hybrid_scores.append(f1)
    ft_hybrid_acc.append(acc)
    print(f'  HYBRID Atom+FT fold {fold+1}: F1={f1:.4f} Acc={acc:.4f}')

mean_f1_hyb = float(np.mean(ft_hybrid_scores))
std_f1_hyb = float(np.std(ft_hybrid_scores))
mean_acc_hyb = float(np.mean(ft_hybrid_acc))
print(f'HYBRID Atom+FT: F1={mean_f1_hyb:.4f}+/-{std_f1_hyb:.4f} Acc={mean_acc_hyb:.4f}')
results['hybrid_atom_ft'] = {'f1': mean_f1_hyb, 'std': std_f1_hyb, 'accuracy': mean_acc_hyb}

json.dump(results, open('models/benchmark/bert_fasttext_results.json', 'w'), indent=2)

# Final table
print()
print('=' * 70)
for k in ['hybrid_atom_bert', 'hybrid_atom_ft', 'berturk_cls_svm', 'berturk_cls_logreg',
          'atomized_tfidf_logreg', 'fasttext_supervised', 'raw_tfidf_logreg', 'fasttext_emb_logreg']:
    if k in results:
        r = results[k]
        print(f'  {k:43s} {r[\"f1\"]:.4f}+/-{r[\"std\"]:.4f}  {r[\"accuracy\"]:.4f}')
print('=' * 70)
print(f'Saved to models/benchmark/bert_fasttext_results.json')
"

echo "Completed: $(date)"
