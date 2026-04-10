# kok-turk

**Turkish Morphological Atomizer** -- Decompose Turkish words into linguistic atoms (root + ordered suffix tags) using neural disambiguation.

```python
from kokturk import MorphoAnalyzer

analyzer = MorphoAnalyzer(backends=["disambiguator"])
results = analyzer.analyze_sentence("Cocuklar evlerinden cikti")
# -> ["cocuk +PLU", "ev +PLU +POSS.3SG +ABL", "cik +PAST +3SG"]
```

## Why kok-turk?

Turkish is agglutinative -- a single word can pack 5+ morphemes. BPE/WordPiece splits these at arbitrary positions, destroying morphological structure. kok-turk splits at **morpheme boundaries** with **98.3% accuracy**.

| Input | BPE (BERTurk) | kok-turk |
|-------|---------------|----------|
| evlerinden | ev ##ler ##inden | ev +PLU +POSS.3SG +ABL |
| gidiyordum | gidi ##yor ##dum | git +PROG +PAST +1SG |
| kitapcilardan | kitap ##ci ##lar ##dan | kitap +AGT +PLU +ABL |

## Key Results

| Task | Metric | Score |
|------|--------|-------|
| **Morphological Disambiguation** | **Test Exact Match** | **98.3%** |
| Morphological Disambiguation | Ambiguous-only EM | 93.8% |
| Morphological Generation | Test Exact Match | 84.7% |
| Text Classification (TTC-3600) | Macro-F1 (Hybrid) | 94.7% |
| Text Classification (TTC-3600) | Macro-F1 (Atomized TF-IDF) | 93.9% |

## SOTA Comparison

| System | EM | Type | Note |
|--------|-----|------|------|
| MorseDisamb (Seker & Eryigit 2017) | 98.59% | Disambiguator | Published SOTA |
| **kok-turk v6 (ours)** | **98.3%** | **Disambiguator** | **1M params, 14 min CPU** |
| TransMorph (Akyurek et al. 2022) | 96.25% | Transformer | Disambiguation |
| SIGMORPHON baseline (2019) | 92.27% | Seq2seq | Generation |
| kok-turk v5.2 (ours, generation) | 84.7% | Dual-Head Decoder | Context-free generation |
| Yildiz et al. 2016 | 84.12% | Analysis from scratch | Comparable to generation |

SOTA-competitive performance (0.3pp gap) achieved with **1M trainable parameters** and **14 minutes CPU training** -- no GPU required.

## Novel Contributions

1. **BERTurk Disambiguation** -- Frozen BERTurk sentence embeddings + lightweight candidate reranker (~1M params). Selects from Zeyrek's morphological candidates using sentence context. Achieves 98.3% EM -- within 0.3pp of published SOTA.

2. **Dual-Head Decoder** -- Root classification (single-step) + conditional tag generation (seq2seq). Eliminates autoregressive root-error propagation. Achieves 84.7% EM without any pretrained model or sentence context.

3. **MIS Metric** (Morphological Informativeness Score) -- Token-level metric measuring atomization benefit: `MIS(x) = a*H_morph + b*D_canon + c*C_struct`. First metric to quantify per-token benefit of morphological vs statistical tokenization.

4. **TAAC** (Tier-Aware Adaptive Curriculum) -- Automatic phase transitions between data quality tiers based on validation loss plateau detection. Component-aware variant monitors root and tag heads independently.

## Architecture

### Disambiguation Model (98.3% EM -- Primary)

```
Input: "Cocuklar evlerinden cikti" (sentence)
                    |
    +---------------+---------------+
    |               |               |
"Cocuklar"    "evlerinden"       "cikti"
    |               |               |
  Zeyrek          Zeyrek          Zeyrek
  1 candidate     3 candidates    2 candidates
    |               |               |
    |         BERTurk context       |
    |          (frozen, 768d)       |
    |               |               |
    |         Score each candidate  |
    |          via reranker MLP     |
    |               |               |
    |          Select best          |
    |               |               |
 "cocuk +PLU"  "ev +PLU +POSS.3SG +ABL"  "cik +PAST +3SG"
```

### Generation Model (84.7% EM -- Fallback for OOV)

```
Input: "evlerinden" (single word, no context)
         |
    Character BiGRU Encoder
         |
    +----+----------------------+
    |                            |
  Root Head                  Tag Decoder
  (classification)    (conditional seq2seq)
  "ev" (1 step)       +PLU +POSS.3SG +ABL
    |                            |
    +------------+---------------+
                 |
    Output: "ev +PLU +POSS.3SG +ABL"
```

## TR-Gold-Morph Resource

The largest Turkish morphological resource: **2.5M entries** with confidence tiers, morpheme boundaries, and multi-schema export.

| | TR-Gold-Morph (ours) | UniMorph Turkish | BOUN Treebank |
|--|---------------------|-----------------|---------------|
| Entries | **2,512,034** | 275,460 | ~121,000 |
| Confidence tiers | gold/silver/bronze | -- | -- |
| Morpheme boundaries | 95.6% | -- | -- |
| Frequency data | yes | -- | -- |
| Export formats | Canonical + UD + UniMorph | UniMorph only | UD only |

## Installation

```bash
pip install kokturk
```

Or from source:

```bash
git clone https://github.com/melikkul/KokTurk.git
cd KokTurk
pip install -e .
```

## Quick Start

### Sentence-Level Disambiguation (98.3% EM)

```python
from kokturk import MorphoAnalyzer

analyzer = MorphoAnalyzer(backends=["disambiguator"])
results = analyzer.analyze_sentence("Cocuklar evlerinden cikti")
for r in results:
    print(r)
```

### Single-Word Analysis (84.7% EM)

```python
from kokturk import Atomizer

atomizer = Atomizer()
atomizer.to_canonical("evlerinden")  # -> "ev +PLU +POSS.3SG +ABL"
atomizer.analyze_batch(["ev", "git", "guzel"])
```

### sklearn Integration

```python
from kokturk.sklearn_ext import MorphoTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("morph", MorphoTransformer(output="atomized")),
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression()),
])
```

### Text Cleaning (ariturk)

```python
from ariturk import TextCleaner, turkish_lower

cleaner = TextCleaner()
cleaner.clean("  TURKCE   metIn  ")  # -> "turkce metin"
turkish_lower("I")  # -> "i" (Turkish-correct)
```

### CLI

```bash
python -m kokturk.cli.main analyze "evlerinden"
# -> ev +PLU +POSS.3SG +ABL
```

## Benchmarks

### Morphological Analysis -- Disambiguation (7,144 test tokens)

| Model | Test EM | Ambig EM | Params | Training |
|-------|---------|----------|--------|----------|
| **v6 Ensemble (5 seeds)** | **98.3%** | **93.8%** | 1M x 5 | 60 min CPU |
| v6 seed=789 (best single) | 98.3% | 93.6% | 1M | 14 min CPU |
| v6 seed=42 | 98.2% | 93.4% | 1M | 14 min CPU |
| v6 seed=123 | 98.2% | 93.2% | 1M | 14 min CPU |

### Morphological Analysis -- Generation (8,140 test tokens)

| Model | Test EM | Root Acc | Tag F1 | Params |
|-------|---------|----------|--------|--------|
| v5.2 Dual-Head (106K) | 84.7% | 87.6% | 93.1% | 5.2M |
| Dual-Head (80K) | 84.5% | 90.6% | 91.5% | 3.0M |
| DH + Context (80K) | 83.8% | 92.3% | 91.9% | 3.9M |
| Seq2seq baseline (80K) | 79.2% | 88.2% | 89.7% | 2.25M |

### TTC-3600 Text Classification (5-fold CV)

| Method | Macro-F1 | Type |
|--------|----------|------|
| **HYBRID: Atomized + BERTurk** | **0.947 +/- 0.009** | Ours |
| BERTurk [CLS] + LogReg | 0.945 +/- 0.008 | Pretrained |
| BERTurk [CLS] + SVM | 0.945 +/- 0.010 | Pretrained |
| HYBRID: Atomized + FastText | 0.941 +/- 0.004 | Ours |
| Atomized TF-IDF + LogReg | 0.939 +/- 0.010 | Ours |
| Raw TF-IDF + LogReg | 0.936 +/- 0.007 | Baseline |
| FastText embeddings + LogReg | 0.934 +/- 0.006 | Embedding |

All pairwise differences tested via paired bootstrap (10K iterations, Holm-Bonferroni corrected).

### Efficiency

| System | Training Time | Model Size | Params |
|--------|-------------|------------|--------|
| v6 Disambiguator | 14 min (CPU) | ~4 MB | 1M trainable + 110M frozen |
| v5.2 Dual-Head | 2 hours (CPU) | ~20 MB | 5.2M |
| Zeyrek (rule-based) | -- | -- | ~6,380 tok/s |

### Training Data

| Tier | Tokens | Source |
|------|--------|--------|
| Gold | 2,496 | Human-annotated |
| Silver-agreed | 16,525 | Multi-backend consensus |
| Silver-auto | 87,016 | Zeyrek + resource augmentation |
| **Total** | **106,037** | 7,732 unique roots, 71 tags |

## Project Structure

```
src/
├── kokturk/              # Core atomizer library
│   ├── core/             # MorphoAnalyzer, datatypes, cache, phonology
│   ├── models/           # Disambiguator, Dual-Head, GRU Seq2Seq, context encoders
│   ├── sklearn_ext/      # sklearn Pipeline integration
│   └── cli/              # Command-line interface
├── ariturk/              # Text cleaning & normalization library
├── train/                # Training scripts, curriculum, losses, HPO
├── benchmark/            # Evaluation: intrinsic, classification, MIS, robustness
├── resource/             # TR-Gold-Morph resource pipeline
├── data/                 # Corpus processing pipeline
└── classify/             # TTC-3600 classification experiments
```

## Citation

```bibtex
@thesis{kul2026kokturk,
  title={Neural Morphological Atomization for Turkish: Gold Standard Corpus Construction and Hybrid Text Classification},
  author={Kul, Melik},
  year={2026},
  school={Ostim Technical University},
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [Zeyrek](https://github.com/obulat/zeyrek) -- Python port of Zemberek
- [BOUN Treebank](https://github.com/UniversalDependencies/UD_Turkish-BOUN)
- [UniMorph](https://unimorph.github.io/)
- [BERTurk](https://github.com/stefan-it/turkish-bert) by Stefan Schweter
