# TR-Gold-Morph Vocabulary (2.5M)

Compiled word and root vocabularies for Turkish morphological analysis. Built from ~1.18M tokens across three major sources (BOUN + IMST + TTC-3600).

## Files

| File | Size | Entries | Description |
|------|------|---------|-------------|
| **word_vocab_2.5M.json** | 17 MB | 545,595 | Word → index mapping (frequency-sorted) |
| **root_vocab_2.5M.json** | 760 KB | 63,198 | Root morphemes (lemmas) array |
| word_vocab.json | 107 KB | ~3,000 | Small word vocabulary (for quick experiments) |
| root_vocab.json | 43 KB | ~1,500 | Small root vocabulary |
| root_vocab_15K.json | 249 KB | ~15,000 | Mid-scale root vocabulary |
| tag_vocab.json | 88 KB | ~150 | Morphological tag vocabulary (canonical format) |
| char_vocab.json | 682 B | ~80 | Character vocabulary (for GRU encoder) |

## Format

### word_vocab_2.5M.json

JSON object: word → integer index. Special tokens first:

```json
{
  "<PAD>": 0,
  "<UNK>": 1,
  ".": 2,
  ",": 3,
  "bir": 4,
  "ve": 5,
  "ki": 6,
  "da": 7,
  "bu": 8,
  ...
}
```

Sorted from highest to lowest frequency. Due to Zipf's law, the top 10K words cover ~92% of running text.

### root_vocab_2.5M.json

JSON array: index → root morpheme (lemma):

```json
[
  "<PAD>",
  "<UNK_ROOT>",
  "Unk",
  "olmak",
  "yapmak",
  "görmek",
  "kalmak",
  "etmek",
  "vermek",
  "kullanmak",
  ...
]
```

### tag_vocab.json

~150 canonical morphological tags:

```json
{
  "<PAD>": 0,
  "<SOS>": 1,
  "<EOS>": 2,
  "<UNK>": 3,
  "+Noun": 4,
  "+PLU": 5,
  "+POSS.3SG": 6,
  "+ABL": 7,
  ...
}
```

## Sources

| Source | Tokens | Sentences | Domain | License |
|--------|--------|-----------|--------|---------|
| **BOUN Treebank** | ~121K | 9,761 | Biography, literature | CC BY-SA 4.0 |
| **IMST Treebank** | ~56K | 5,635 | News | CC BY-NC-SA 3.0 |
| **TTC-3600** | ~1M | 3,600 documents | News (6 categories) | CC BY 4.0 |
| **Total** | **~1.18M** | — | — | — |

### License Warning

The IMST Treebank is licensed under **CC BY-NC-SA 3.0** — it prohibits commercial use. This vocabulary contains IMST-derived data, so the NC restriction applies to derivative works. See `docs/DATA_LICENSE_AUDIT.md` for details.

## Statistics

- **Vocabulary coverage**: 96.89% on the Zeyrek test set
- **Ambiguity rate**: 22.75% of tokens have multiple candidate parses
- **Average candidates**: 1.41 parses per word
- **Root diversity**: 63,198 unique root morphemes
- **Frequency distribution**: Zipf's law (alpha ~ 1.07)

### Frequency Classes

| Class | Definition | Coverage |
|-------|------------|----------|
| HIGH | Top 1K words | ~75% of tokens |
| MID | 1K–10K | ~17% of tokens |
| LOW | 10K–50K | ~6% of tokens |
| RARE | 50K+ | ~2% of tokens |

## Usage

### Loading in Python

```python
from train.datasets import Vocab

# Word vocabulary
word_vocab = Vocab.load("models/vocabs/word_vocab_2.5M.json")
idx = word_vocab.encode("ev")        # → integer index
word = word_vocab.decode(idx)         # → "ev"

# Root vocabulary
root_vocab = Vocab.load("models/vocabs/root_vocab_2.5M.json")
```

### Cache Sizing

Vocabulary frequency statistics are used to optimize the analyzer cache size:

```python
from kokturk import MorphoAnalyzer

analyzer = MorphoAnalyzer(backends=["zeyrek"])
# Top 50K words yield ~91% cache hit rate
analyzer.enable_cache(memory_size=50_000)
```

### Model Training

Training scripts load vocabulary files automatically:

```bash
PYTHONPATH=src python src/train/train_v4_master.py \
    --vocab-dir models/vocabs \
    --root-vocab-path models/vocabs/root_vocab_2.5M.json \
    ...
```

## Production Pipeline

```
BOUN (CoNLL-U) ───┐
IMST (CoNLL-U) ───┼──→ Ingest → Pre-labeling → Bronze
TTC-3600 (text) ──┘         │
                       Label Model (skweak HMM) → Silver
                            │
                       Active Learning → Gold (human verification)
                            │
                       Vocabulary Export → word_vocab_2.5M.json
                                        → root_vocab_2.5M.json
```
