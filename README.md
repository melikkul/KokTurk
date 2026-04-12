# kök-türk 🌳

**Turkish Morphological Atomizer** — Decompose Turkish words into their linguistic atoms using neural disambiguation. SOTA-competitive accuracy with minimal compute.

```python
from kokturk import MorphoAnalyzer

analyzer = MorphoAnalyzer(backends=["disambiguator"])
results = analyzer.analyze_sentence("Çocuklar evlerinden çıktı")
# → ["çocuk +PLU", "ev +PLU +POSS.3SG +ABL", "çık +PAST +3SG"]
```

## Highlights

| | |
|---|---|
| 🎯 **98.3% Exact Match** | SOTA-competitive morphological disambiguation |
| ⚡ **14 min CPU training** | No GPU required — frozen BERTurk + 1M param reranker |
| 📚 **2.5M entry resource** | Largest Turkish morphological database |
| 🔬 **94.7% F1 classification** | Hybrid atomization beats BERTurk-only on TTC-3600 |

## How It Works

kök-türk replaces statistical subword splitting (BPE/WordPiece) with linguistically motivated morpheme decomposition:

| Input | BPE (BERTurk) | kök-türk |
|-------|---------------|----------|
| evlerinden | ev ##ler ##inden | ev +PLU +POSS.3SG +ABL |
| gidiyordum | gidi ##yor ##dum | git +PROG +PAST +1SG |
| kitapçılardan | kitap ##çı ##lar ##dan | kitap +AGT +PLU +ABL |

The system operates in two modes:

**Disambiguation** (primary) — Given a sentence, generate morphological candidates via Zeyrek, then select the best parse using BERTurk sentence context. This is how SOTA systems work.

**Generation** (fallback) — For out-of-vocabulary words, a Dual-Head Decoder generates the parse character-by-character without requiring sentence context.

## Performance

### Morphological Analysis

| System | Exact Match | Type |
|--------|------------|------|
| MorseDisamb (Şeker & Eryiğit 2017) | 98.59% | Published SOTA |
| **kök-türk** | **98.3%** | **Disambiguation** |
| TransMorph (Akyürek et al. 2022) | 96.25% | Disambiguation |
| SIGMORPHON baseline (2019) | 92.27% | Generation |
| Yıldız et al. 2016 | 84.12% | Generation |

### Text Classification (TTC-3600, 5-fold CV)

| Method | Macro-F1 |
|--------|----------|
| Atomized TF-IDF + BERTurk (hybrid) | 0.947 ± 0.009 |
| BERTurk [CLS] + LogReg | 0.945 ± 0.008 |
| BERTurk [CLS] + SVM | 0.945 ± 0.010 |
| Atomized TF-IDF + FastText (hybrid) | 0.941 ± 0.004 |
| Atomized TF-IDF + LogReg | 0.939 ± 0.010 |
| Raw TF-IDF + LogReg | 0.936 ± 0.007 |
| FastText embeddings + LogReg | 0.934 ± 0.006 |

All differences tested via paired bootstrap (10K iterations, Holm-Bonferroni corrected).

### Efficiency

| Component | Training | Size | Params |
|-----------|---------|------|--------|
| Disambiguator | 14 min CPU | ~4 MB | 1M trainable |
| Dual-Head Decoder | 2 hours CPU | ~20 MB | 5.2M |
| BERTurk (frozen) | — | ~440 MB | 110M (no gradient) |

## Architecture

```
Sentence: "Çocuklar evlerinden çıktı"
                    │
     ┌──────────────┼──────────────┐
     │              │              │
 "Çocuklar"   "evlerinden"     "çıktı"
     │              │              │
   Zeyrek         Zeyrek         Zeyrek
   1 candidate    3 candidates   2 candidates
     │              │              │
     │        ┌─────┴─────┐       │
     │        │ BERTurk   │       │
     │        │ (frozen)  │       │
     │        │ 768-dim   │       │
     │        └─────┬─────┘       │
     │              │              │
     │        Score + Select       │
     │              │              │
 "çocuk +PLU"  "ev +PLU           "çık +PAST
               +POSS.3SG           +3SG"
               +ABL"
```

## TR-Gold-Morph

The largest Turkish morphological resource:

| | kök-türk | UniMorph Turkish | BOUN Treebank |
|--|---------|-----------------|---------------|
| Entries | **2,512,034** | 275,460 | ~121,000 |
| Confidence tiers | gold / silver / bronze | — | — |
| Morpheme boundaries | 95.6% | — | — |
| Export formats | Canonical + UD + UniMorph | UniMorph | UD |

## Installation

```bash
pip install kokturk
```

From source:

```bash
git clone https://github.com/melikkul/KokTurk.git
cd KokTurk
pip install -e .
```

## Usage

### Sentence Disambiguation

```python
from kokturk import MorphoAnalyzer

analyzer = MorphoAnalyzer(backends=["disambiguator"])
results = analyzer.analyze_sentence("Çocuklar evlerinden çıktı")
```

### Single Word Analysis

```python
from kokturk import Atomizer

atomizer = Atomizer()
atomizer.to_canonical("evlerinden")
# → "ev +PLU +POSS.3SG +ABL"
```

### sklearn Pipeline

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

### Text Cleaning (arı-türk)

```python
from ariturk import TextCleaner, turkish_lower

cleaner = TextCleaner()
cleaner.clean("  TÜRKÇE   metİn  ")  # → "türkçe metin"
turkish_lower("I")  # → "ı"
```

### CLI

```bash
python -m kokturk.cli.main analyze "evlerinden"
# → ev +PLU +POSS.3SG +ABL
```

## Project Structure

```
src/
├── kokturk/          # Core morphological atomizer
│   ├── core/         # MorphoAnalyzer, datatypes, cache, phonology
│   ├── models/       # Disambiguator, Dual-Head Decoder, context encoders
│   ├── sklearn_ext/  # sklearn integration
│   └── cli/          # Command-line interface
├── ariturk/          # Turkish text cleaning & normalization
├── train/            # Training scripts, curriculum, losses
├── benchmark/        # Evaluation suite
├── resource/         # TR-Gold-Morph pipeline
└── classify/         # TTC-3600 experiments
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

- [Zeyrek](https://github.com/obulat/zeyrek) — Python port of Zemberek
- [BOUN Treebank](https://github.com/UniversalDependencies/UD_Turkish-BOUN)
- [UniMorph](https://unimorph.github.io/)
- [BERTurk](https://github.com/stefan-it/turkish-bert) by Stefan Schweter
