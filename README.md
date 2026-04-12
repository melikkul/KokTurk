# kök-türk

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

### Detailed System Comparison

| System | EM | Approach | Pretrained | Context | Language | Params |
|--------|-----|---------|-----------|---------|----------|--------|
| MorseDisamb | 98.59% | Candidate selection | — | Sentence | Java | — |
| **kök-türk** | **98.3%** | **Candidate selection** | **BERTurk (frozen)** | **Sentence** | **Python** | **1M** |
| TransMorph | 96.25% | Transformer disamb. | — | Sentence | — | — |
| Sak et al. 2009 | 97.81% | Perceptron | — | Sentence | — | — |
| Morse (generation) | 97.67% | Seq2seq | — | — | Java | — |
| SIGMORPHON 2019 | 92.27% | Seq2seq | — | — | — | — |
| Stanza | ~90.4% | Pipeline (UD) | Word2Vec | Sentence | Python | — |
| spaCy (trf) | ~87.8% | Pipeline (UD) | Transformer | Sentence | Python | — |
| kök-türk (generation) | 84.7% | Dual-Head Decoder | — | — | Python | 5.2M |
| Yıldız et al. 2016 | 84.12% | Perceptron | — | — | — | — |
| GPT-4o (zero-shot) | ~36.7% | LLM prompting | GPT-4o | Sentence | API | 1.8T |

### Text Classification — Why Atomization Matters

TTC-3600 (3,600 documents, 6 categories), 5-fold stratified CV.

| Feature Type | Classifier | Macro-F1 | Delta vs Raw |
|-------------|-----------|----------|----------|
| Atomized + BERTurk | LogReg | **0.947** | +1.1% |
| BERTurk [CLS] only | LogReg | 0.945 | +0.9% |
| Atomized + FastText | LogReg | 0.941 | +0.5% |
| Atomized TF-IDF | LogReg | 0.939 | +0.3% |
| Raw TF-IDF | LogReg | 0.936 | baseline |
| FastText embeddings | LogReg | 0.934 | -0.2% |

Morphological atomization consistently improves every classifier — including BERTurk. All differences tested via paired bootstrap (10K iterations, Holm-Bonferroni corrected).

### Training Efficiency

| System | Training Time | Hardware | Accuracy |
|--------|-------------|----------|----------|
| **kök-türk** | **14 min** | **CPU only** | **98.3%** |
| Typical Transformer NLP | Hours–days | GPU required | Varies |
| BERTurk fine-tuning | ~2 hours | GPU required | ~95% |
| Morse | Not reported | — | 98.59% |

### Inference

| Component | Speed | Memory |
|-----------|-------|--------|
| Zeyrek candidate generation | ~6,380 tok/s | ~50 MB |
| BERTurk embedding (cached) | ~3,200 sent/s | ~1.5 GB |
| Reranker scoring | ~50,000 tok/s | ~10 MB |
| Dual-Head generation | ~2,000 tok/s | ~20 MB |

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

| Resource | Entries | Annotation | Tiers | Boundaries | Formats | License |
|----------|---------|-----------|-------|-----------|---------|---------|
| **TR-Gold-Morph** | **2,512,034** | **Auto + manual** | **gold/silver/bronze** | **95.6%** | **3 schemas** | **MIT** |
| UniMorph Turkish | 275,460 | Rule-generated | — | — | UniMorph | CC BY-SA |
| BOUN Treebank | ~121,000 | Manual | — | — | UD CoNLL-U | CC BY-SA 4.0 |
| IMST Treebank | ~56,000 | Manual | — | — | UD CoNLL-U | CC BY-NC-SA 3.0 |
| TrMor (Zemberek) | ~50,000 roots | Rule-based FSA | — | — | Custom | — |

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
