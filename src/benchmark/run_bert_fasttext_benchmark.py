"""Direct BERT/FastText/Hybrid comparison on TTC-3600.

Thesis gap closure: measures FastText, BERTurk, and hybrid (atomized + BERTurk)
classifiers on the same TTC-3600 splits for direct comparison.

Methods evaluated:
    1. Raw TF-IDF + LogReg (baseline)
    2. Atomized TF-IDF + LogReg (ours, existing)
    3. FastText supervised (train_supervised)
    4. FastText embeddings + LogReg (skipgram → mean pool → LogReg)
    5. BERTurk [CLS] + LogReg (frozen features)
    6. BERTurk [CLS] + SVM (frozen features)
    7. HYBRID: Atomized TF-IDF + BERTurk [CLS] (feature concat)
    8. HYBRID: Atomized TF-IDF + FastText embeddings (feature concat)

Usage:
    PYTHONPATH=src python src/benchmark/run_bert_fasttext_benchmark.py
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)

# TTC-3600 Turkish category names → English labels
CATEGORY_MAP = {
    "ekonomi": "economy",
    "kultursanat": "culture_art",
    "saglik": "health",
    "siyaset": "politics",
    "spor": "sports",
    "teknoloji": "technology",
}

TTC_PATH = Path("data/external/ttc3600_raw/TTC-3600_Orj")
RESULTS_PATH = Path("models/benchmark/bert_fasttext_results.json")


def load_ttc3600() -> tuple[list[str], list[int], list[str]]:
    """Load TTC-3600 from raw text files.

    Returns:
        Tuple of (documents, label_indices, label_names).
    """
    docs: list[str] = []
    labels: list[int] = []
    label_names: list[str] = []

    for cat_dir in sorted(TTC_PATH.iterdir()):
        if not cat_dir.is_dir():
            continue
        cat_name = CATEGORY_MAP.get(cat_dir.name, cat_dir.name)
        if cat_name not in label_names:
            label_names.append(cat_name)
        cat_idx = label_names.index(cat_name)

        for f in sorted(cat_dir.glob("*.txt")):
            text = f.read_text(encoding="utf-8", errors="ignore").strip()
            if len(text) > 20:
                docs.append(text)
                labels.append(cat_idx)

    return docs, labels, label_names


def eval_5fold(
    X: np.ndarray,
    labels: np.ndarray,
    clf_factory,
    name: str,
    *,
    return_per_doc: bool = False,
) -> dict:
    """5-fold CV evaluation. Returns dict with f1, std, and optionally per-doc predictions."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores: list[float] = []
    acc_scores: list[float] = []
    all_preds = np.zeros(len(labels), dtype=int)

    for fold, (tr, te) in enumerate(skf.split(X, labels)):
        clf = clf_factory()
        y_tr = labels[tr]
        y_te = labels[te]
        clf.fit(X[tr], y_tr)
        preds = clf.predict(X[te])
        all_preds[te] = preds
        f1 = f1_score(y_te, preds, average="macro")
        acc = accuracy_score(y_te, preds)
        scores.append(f1)
        acc_scores.append(acc)
        logger.info("  %s fold %d: F1=%.3f Acc=%.3f", name, fold + 1, f1, acc)

    mean_f1 = float(np.mean(scores))
    std_f1 = float(np.std(scores))
    mean_acc = float(np.mean(acc_scores))
    print(f"  {name:45s} F1={mean_f1:.4f}±{std_f1:.4f}  Acc={mean_acc:.4f}")

    result = {"f1": mean_f1, "std": std_f1, "accuracy": mean_acc}
    if return_per_doc:
        result["predictions"] = all_preds.tolist()
    return result


def eval_fasttext_supervised(
    docs: list[str], labels: np.ndarray,
) -> dict | None:
    """FastText supervised classification with 5-fold CV."""
    try:
        import fasttext
    except ImportError:
        logger.warning("fasttext not installed — skipping")
        return None

    # Suppress fasttext warnings
    fasttext.FastText.eprint = lambda x: None

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores: list[float] = []
    acc_scores: list[float] = []
    all_preds = np.zeros(len(labels), dtype=int)
    tmp = tempfile.mkdtemp()

    try:
        for fold, (tr, te) in enumerate(skf.split(docs, labels)):
            ft_train = os.path.join(tmp, f"train_{fold}.txt")
            with open(ft_train, "w", encoding="utf-8") as f:
                for i in tr:
                    clean = docs[i].replace("\n", " ").replace("\r", " ")
                    f.write(f"__label__{labels[i]} {clean}\n")

            model = fasttext.train_supervised(
                ft_train, epoch=25, lr=0.5, wordNgrams=2,
                dim=100, loss="softmax", verbose=0,
            )

            preds = []
            for i in te:
                clean = docs[i].replace("\n", " ").replace("\r", " ")
                pred_label = model.predict(clean)[0][0].replace("__label__", "")
                preds.append(int(pred_label))

            gold = labels[te]
            all_preds[te] = preds
            f1 = f1_score(gold, preds, average="macro")
            acc = accuracy_score(gold, preds)
            scores.append(f1)
            acc_scores.append(acc)
            logger.info("  FastText supervised fold %d: F1=%.3f", fold + 1, f1)
    finally:
        shutil.rmtree(tmp)

    mean_f1 = float(np.mean(scores))
    std_f1 = float(np.std(scores))
    mean_acc = float(np.mean(acc_scores))
    print(f"  {'FastText supervised':45s} F1={mean_f1:.4f}±{std_f1:.4f}  Acc={mean_acc:.4f}")
    return {"f1": mean_f1, "std": std_f1, "accuracy": mean_acc, "predictions": all_preds.tolist()}


def get_fasttext_embeddings(docs: list[str]) -> np.ndarray | None:
    """Train skipgram on TTC-3600 corpus, return doc vectors (mean pool)."""
    try:
        import fasttext
    except ImportError:
        logger.warning("fasttext not installed — skipping embeddings")
        return None

    fasttext.FastText.eprint = lambda x: None
    tmp = tempfile.mkdtemp()
    try:
        corpus_file = os.path.join(tmp, "corpus.txt")
        with open(corpus_file, "w", encoding="utf-8") as f:
            for doc in docs:
                f.write(doc.replace("\n", " ") + "\n")

        ft_model = fasttext.train_unsupervised(
            corpus_file, model="skipgram", dim=100, epoch=10, verbose=0,
        )

        def doc_vec(text: str) -> np.ndarray:
            words = text.split()
            vecs = [ft_model.get_word_vector(w) for w in words if w.strip()]
            return np.mean(vecs, axis=0) if vecs else np.zeros(100)

        X = np.array([doc_vec(doc) for doc in docs])
    finally:
        shutil.rmtree(tmp)

    return X


def get_berturk_embeddings(docs: list[str]) -> np.ndarray | None:
    """Extract frozen BERTurk [CLS] embeddings."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        logger.warning("transformers not installed — skipping BERTurk")
        return None

    berturk_path = Path("models/berturk")
    if not berturk_path.exists():
        logger.warning("BERTurk not found at %s — skipping", berturk_path)
        return None

    tokenizer = AutoTokenizer.from_pretrained(str(berturk_path))
    model = AutoModel.from_pretrained(str(berturk_path))
    model.eval()

    embeddings: list[np.ndarray] = []
    batch_size = 32

    for i in range(0, len(docs), batch_size):
        batch = [d[:512] for d in docs[i : i + batch_size]]
        enc = tokenizer(
            batch, padding=True, truncation=True, max_length=128,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = model(**enc)
            cls = out.last_hidden_state[:, 0, :].numpy()
            embeddings.append(cls)
        if (i // batch_size) % 25 == 0:
            print(f"    BERTurk: {i}/{len(docs)} docs embedded")

    return np.vstack(embeddings)


def atomize_docs(docs: list[str]) -> list[str]:
    """Atomize documents using Zeyrek analyzer."""
    from kokturk.core.analyzer import MorphoAnalyzer

    analyzer = MorphoAnalyzer()
    atomized: list[str] = []

    for i, doc in enumerate(docs):
        words = doc.split()[:300]  # cap for speed
        parts: list[str] = []
        for w in words:
            result = analyzer.analyze(w)
            best = result.best
            parts.append(best.to_str() if best is not None else w)
        atomized.append(" ".join(parts))
        if (i + 1) % 500 == 0:
            print(f"    Atomized: {i + 1}/{len(docs)}")

    analyzer.close()
    return atomized


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    docs, label_list, label_names = load_ttc3600()
    labels = np.array(label_list)
    print(f"TTC-3600: {len(docs)} docs, {len(label_names)} classes: {label_names}")
    from collections import Counter
    dist = Counter(labels)
    for idx, name in enumerate(label_names):
        print(f"  {name}: {dist[idx]}")
    print()

    results: dict[str, dict] = {}
    results["_meta"] = {
        "dataset": "TTC-3600",
        "n_docs": len(docs),
        "n_classes": len(label_names),
        "label_names": label_names,
        "cv_folds": 5,
        "random_state": 42,
    }

    # ─── 1. Raw TF-IDF + LogReg ───────────────────────────────────────
    print("=" * 60)
    print("1. Raw TF-IDF + LogReg")
    tfidf_raw = TfidfVectorizer(
        max_features=50000, ngram_range=(1, 2), sublinear_tf=True,
    )
    X_raw = tfidf_raw.fit_transform(docs)
    results["raw_tfidf_logreg"] = eval_5fold(
        X_raw, labels,
        lambda: LogisticRegression(max_iter=1000, C=1.0),
        "Raw TF-IDF + LogReg",
        return_per_doc=True,
    )
    print()

    # ─── 2. Atomized TF-IDF + LogReg ──────────────────────────────────
    print("2. Atomized TF-IDF + LogReg (ours)")
    print("    Atomizing documents...")
    atomized_docs = atomize_docs(docs)
    tfidf_atom = TfidfVectorizer(
        max_features=50000, ngram_range=(1, 2), sublinear_tf=True,
    )
    X_atom = tfidf_atom.fit_transform(atomized_docs)
    results["atomized_tfidf_logreg"] = eval_5fold(
        X_atom, labels,
        lambda: LogisticRegression(max_iter=1000, C=1.0),
        "Atomized TF-IDF + LogReg",
        return_per_doc=True,
    )
    print()

    # ─── 3. FastText supervised ────────────────────────────────────────
    print("3. FastText supervised")
    ft_sup = eval_fasttext_supervised(docs, labels)
    if ft_sup is not None:
        results["fasttext_supervised"] = ft_sup
    print()

    # ─── 4. FastText embeddings + LogReg ───────────────────────────────
    print("4. FastText embeddings + LogReg")
    X_ft = get_fasttext_embeddings(docs)
    if X_ft is not None:
        results["fasttext_emb_logreg"] = eval_5fold(
            X_ft, labels,
            lambda: LogisticRegression(max_iter=1000, C=1.0),
            "FastText emb + LogReg",
            return_per_doc=True,
        )
    print()

    # ─── 5. BERTurk [CLS] + LogReg ────────────────────────────────────
    print("5. BERTurk [CLS] + LogReg")
    X_bert = get_berturk_embeddings(docs)
    if X_bert is not None:
        results["berturk_cls_logreg"] = eval_5fold(
            X_bert, labels,
            lambda: LogisticRegression(max_iter=1000, C=1.0),
            "BERTurk [CLS] + LogReg",
            return_per_doc=True,
        )
        print()

        # ─── 6. BERTurk [CLS] + SVM ───────────────────────────────────
        print("6. BERTurk [CLS] + SVM")
        results["berturk_cls_svm"] = eval_5fold(
            X_bert, labels,
            lambda: CalibratedClassifierCV(LinearSVC(max_iter=5000, C=0.5)),
            "BERTurk [CLS] + SVM",
            return_per_doc=True,
        )
        print()

        # ─── 7. HYBRID: Atomized TF-IDF + BERTurk [CLS] ──────────────
        print("7. HYBRID: Atomized TF-IDF + BERTurk [CLS]")
        from scipy.sparse import csr_matrix, hstack
        X_hybrid_bert = hstack([X_atom, csr_matrix(X_bert)])
        results["hybrid_atom_bert"] = eval_5fold(
            X_hybrid_bert, labels,
            lambda: LogisticRegression(max_iter=1000, C=1.0),
            "HYBRID: Atomized TF-IDF + BERTurk",
            return_per_doc=True,
        )
        print()

    # ─── 8. HYBRID: Atomized TF-IDF + FastText ────────────────────────
    if X_ft is not None:
        print("8. HYBRID: Atomized TF-IDF + FastText embeddings")
        from scipy.sparse import csr_matrix, hstack
        X_hybrid_ft = hstack([X_atom, csr_matrix(X_ft)])
        results["hybrid_atom_ft"] = eval_5fold(
            X_hybrid_ft, labels,
            lambda: LogisticRegression(max_iter=1000, C=1.0),
            "HYBRID: Atomized TF-IDF + FastText",
            return_per_doc=True,
        )
        print()

    # ─── Summary table ─────────────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"{'Method':45s} {'Macro-F1':>10s}  {'Accuracy':>10s}")
    print("=" * 70)
    for key in [
        "hybrid_atom_bert", "hybrid_atom_ft",
        "berturk_cls_svm", "berturk_cls_logreg",
        "atomized_tfidf_logreg", "fasttext_supervised",
        "raw_tfidf_logreg", "fasttext_emb_logreg",
    ]:
        if key in results:
            r = results[key]
            print(
                f"  {key:43s} {r['f1']:.4f}±{r['std']:.4f}  {r['accuracy']:.4f}"
            )
    print("=" * 70)

    # ─── Significance tests ────────────────────────────────────────────
    print()
    print("Paired bootstrap significance tests vs Atomized TF-IDF:")
    from benchmark.significance import paired_bootstrap_test, holm_bonferroni_correction

    baseline_key = "atomized_tfidf_logreg"
    if baseline_key in results and "predictions" in results[baseline_key]:
        baseline_preds = results[baseline_key]["predictions"]
        p_values: list[tuple[str, float]] = []

        for key in results:
            if key.startswith("_") or key == baseline_key:
                continue
            if "predictions" not in results[key]:
                continue
            other_preds = results[key]["predictions"]
            sig = paired_bootstrap_test(
                baseline_preds, other_preds, labels.tolist(),
            )
            p_values.append((key, sig["p_value"]))
            delta = results[key]["f1"] - results[baseline_key]["f1"]
            print(
                f"  vs {key:40s} Δ={delta:+.4f}  p={sig['p_value']:.4f}  "
                f"d={sig['cohens_d']:.3f}"
            )

        if len(p_values) >= 2:
            names, pvals = zip(*p_values)
            corrected = holm_bonferroni_correction(list(pvals))
            print()
            print("  Holm-Bonferroni corrected p-values:")
            for name, orig, corr in zip(names, pvals, corrected):
                sig_marker = "*" if corr < 0.05 else ""
                print(f"    {name:40s} p_raw={orig:.4f}  p_corr={corr:.4f} {sig_marker}")

    # ─── Save results (strip predictions for clean JSON) ──────────────
    clean_results = {}
    for k, v in results.items():
        if isinstance(v, dict):
            clean_results[k] = {kk: vv for kk, vv in v.items() if kk != "predictions"}
        else:
            clean_results[k] = v

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(clean_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
