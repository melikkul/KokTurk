# kok-turk

**Turkish Morphological Atomizer** — Decomposes Turkish surface forms into linguistic atoms (root + ordered suffix tags).

```python
from kokturk import Atomizer

atomizer = Atomizer()
atomizer.to_canonical("evlerinden")
# → "ev +PLU +POSS.3SG +ABL"
```

## Why kok-turk?

Turkish is an agglutinative language: a single word can carry 8-10 suffixes. Subword methods like BPE/WordPiece ignore morphological boundaries. kok-turk decomposes each word into **root + ordered canonical tags**, preserving linguistic structure.

| Input | BPE | kok-turk |
|-------|-----|----------|
| evlerinden | ev \| ler \| inden | ev +PLU +POSS.3SG +ABL |
| gidiyormuşsunuz | gidi \| yor \| muş \| sunuz | git +PROG +NARR +2PL |
| güzelleştirilemeyebileceklerdenmişsiniz | 6+ subwords | güzel +BECOME +CAUS +ABIL.NEG +POT +PLU +ABL +NARR +2PL |

## Installation

```bash
pip install kokturk
# or from the project:
pip install -e ".[dev]"
```

**Requirements:** Python >= 3.10, zeyrek >= 0.1.3, torch >= 2.0.0

## Quick Start

### Basic Analysis

```python
from kokturk import Atomizer, MorphoAnalyzer

# High-level API
atomizer = Atomizer()
print(atomizer.to_canonical("okullarımızdan"))
# → "okul +PLU +POSS.1PL +ABL"

# All candidate parses
for analysis in atomizer.analyze_all("yüzme"):
    print(f"  {analysis.root} {' '.join(analysis.tags)}")
# → yüz +INF.MA      (from the verb yüzmek)
# → yüz +DAT          (from the noun yüz)
```

### Multi-Backend Analysis

```python
from kokturk import MorphoAnalyzer

analyzer = MorphoAnalyzer(backends=["zeyrek"])

result = analyzer.analyze("kitaplarımızı")
print(result.best.to_str())
# → "kitap +PLU +POSS.1PL +ACC"
print(f"Candidates: {result.parse_count}")
print(f"Ambiguous: {result.is_ambiguous}")
```

### Batch Processing

```python
# Iterator-based pipeline
for analysis in analyzer.pipe(["ev", "araba", "güzel"], batch_size=64):
    print(f"{analysis.surface} → {analysis.best.to_str()}")
```

### Grammar Checking

```python
from kokturk import GrammarChecker

checker = GrammarChecker()

# Spelling + grammar + punctuation check
result = checker.check("Turkiyede benim ev cok guzel")
print(result.corrected)
# → "Türkiye'de benim evim çok güzel"

for issue in result.issues:
    print(f"  [{issue.error_type}] {issue.message}")

# Auto-correct
fixed = checker.correct("Istanbul guzel")
# → "İstanbul güzel"
```

### Spell Checking

```python
from kokturk.models import TurkishSpellChecker

spell = TurkishSpellChecker()
suggestions = spell.check("turkce ogretmen")
for s in suggestions:
    print(f"  {s.original} → {s.suggestion}  ({s.error_type})")
# → turkce → türkçe  (deasciify)
# → ogretmen → öğretmen  (deasciify)
```

### Grammar Rules

```python
from kokturk.models import TurkishGrammarChecker

grammar = TurkishGrammarChecker()
errors = grammar.check("benim ev kitap de")
for e in errors:
    print(f"  [{e.severity}] {e.message}")
# → [error] genitive 'benim' requires possessive on 'ev'
# → [error] voiceless consonant requires 'te', not 'de'
```

### scikit-learn Integration

```python
from kokturk.sklearn_ext import MorphoTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

pipe = Pipeline([
    ("morph", MorphoTransformer(output="atomized")),
    ("tfidf", TfidfVectorizer()),
    ("svm", SVC()),
])
pipe.fit(train_texts, train_labels)
```

## Architecture

```
kokturk/
├── core/                  # Morphological analysis core
│   ├── analyzer.py        # MorphoAnalyzer — unified multi-backend interface
│   ├── datatypes.py       # Morpheme, MorphologicalAnalysis, TokenAnalyses
│   ├── grammar.py         # GrammarChecker — unified grammar/spell/punct
│   ├── cache.py           # Two-tier LRU cache (memory + SQLite)
│   ├── phonology.py       # Vowel harmony, consonant voicing
│   ├── constants.py       # Canonical tag mappings (150+ tags)
│   ├── code_switch.py     # Code-switch detection (Google'ladım)
│   ├── compound_lexicon.py # Light verb decomposition (reddetti → ret +LVC.ET)
│   ├── special_tokens.py  # Abbreviations, numerics, reduplication
│   └── output_formats.py  # text/json/minimal output modes
│
├── models/                # Neural models and checkers
│   ├── disambiguator.py   # BERTurk disambiguation (98.3% EM)
│   ├── grammar_checker.py # Rule-based grammar checking
│   ├── spell_checker.py   # ASCII deasciification, I/İ, edit distance
│   ├── punctuation_restorer.py # BERTurk + classifier (7 classes)
│   ├── dual_head.py       # Root classifier + tag decoder
│   ├── char_gru.py        # Character GRU Seq2Seq + Bahdanau attention
│   ├── morphotactic_mask.py # Turkish suffix ordering constraints (FSA)
│   └── ...
│
├── cli/                   # Command-line interface
├── hf_integration/        # HuggingFace integration (planned)
├── sklearn_ext/           # scikit-learn MorphoTransformer
└── plugins/               # Plugin discovery (planned)
```

## Backends

| Backend | Speed | Dependency | Description |
|---------|-------|------------|-------------|
| `zeyrek` | ~6,380 tok/s | zeyrek | Zemberek Python port, rule-based |
| `trmorph` | ~1,000 tok/s | foma | FST-based, complementary lexicon |
| `neural` | ~2,000 tok/s | torch | Trained GRU Seq2Seq |
| `disambiguator` | ~50,000 tok/s | torch, transformers | BERTurk reranking, **98.3% EM** |

```python
# Single backend
analyzer = MorphoAnalyzer(backends=["zeyrek"])

# With disambiguation
analyzer = MorphoAnalyzer(backends=["disambiguator"])
```

## Caching

Default 50K-entry in-memory LRU cache covers ~91% of Turkish running text (~17.5 MB RAM):

```python
analyzer = MorphoAnalyzer(backends=["zeyrek"])
analyzer.enable_cache(memory_size=100_000, disk_path="cache.db")

stats = analyzer.cache_stats
print(f"Hit rate: {stats['hit_rate']:.1%}")
```

## Special Token Support

```python
# Light verb decomposition
result = analyzer.analyze("reddetti", decompose_lvc=True)
# → ret +LVC.ET +PAST

# Code-switching (foreign root + Turkish suffix)
result = analyzer.analyze("Google'ladım", handle_code_switch=True)
# → Google +FOREIGN +VERB.LA +PAST +1SG

# Abbreviation and numeral handling
result = analyzer.analyze("NATO'nun", handle_special_tokens=True)
# → NATO +GEN
```

## Data Structures

All outputs are immutable frozen dataclasses:

```python
@dataclass(frozen=True, slots=True)
class Morpheme:
    surface: str      # "ler", "den", ...
    canonical: str    # "+PLU", "+ABL", ...
    category: str     # "inflectional" | "derivational"

@dataclass(frozen=True, slots=True)
class MorphologicalAnalysis:
    surface: str                       # "evlerinden"
    root: str                          # "ev"
    tags: tuple[str, ...]              # ("+PLU", "+POSS.3SG", "+ABL")
    morphemes: tuple[Morpheme, ...]
    source: str                        # "zeyrek" | "neural" | "gold"
    score: float                       # [0.0, 1.0]

@dataclass(frozen=True, slots=True)
class TokenAnalyses:
    surface: str
    analyses: tuple[MorphologicalAnalysis, ...]
    # .best → highest-scoring analysis
    # .is_ambiguous → has multiple candidates
    # .parse_count → number of candidates
```

## Tag Set

~150 canonical tags, ordered (Turkish suffix ordering is non-commutative):

| Category | Tags |
|----------|------|
| Number | +PLU, +SGN |
| Possession | +POSS.1SG, +POSS.2SG, +POSS.3SG, +POSS.1PL, +POSS.2PL, +POSS.3PL |
| Case | +NOM, +ACC, +DAT, +LOC, +ABL, +GEN, +INS, +EQU |
| Tense | +PAST, +NARR, +PROG, +AOR, +FUT |
| Mood | +IMP, +OPT, +NEC, +COND |
| Voice | +PASS, +CAUS, +RECIP, +REFLEX |
| Derivation | +BECOME, +INF, +PART, +ADV, +AGT, +NESS |
| Person | +1SG, +2SG, +3SG, +1PL, +2PL, +3PL |

## License

Apache License 2.0
