# ari-turk

**Turkish Text Cleaning & Normalization Library**

Lightweight library providing Turkish-specific casing rules (I/İ, ı/i), Unicode normalization, diacritics restoration, quality tier assignment, and morpheme boundary extraction.

```python
from ariturk import TextCleaner, turkish_lower

cleaner = TextCleaner(fix_diacritics=True)
cleaner.clean("  TURKCE   ISLEM  ")
# → "türkçe işlem"

turkish_lower("ISTANBUL")
# → "ıstanbul"  (I → ı, standard Python .lower() gets this wrong)
```

## Features

- **Turkish-correct casing**: `I → ı`, `i → İ` (Python `str.lower()` cannot do this)
- **Unicode NFC normalization** + whitespace cleanup
- **Diacritics restoration**: "turkce" → "türkçe", "universite" → "üniversite"
- **Quality tiering**: gold (human-verified) / silver (multi-source agreement) / bronze (single source)
- **Morpheme boundary extraction**: "evlerinden" + "ev +PLU +POSS.3SG +ABL" → "ev|ler|i|nden"
- **Zero external dependencies**: Pure Python stdlib (`unicodedata`, `re`)

## Installation

```bash
pip install ariturk
# or from the project:
pip install -e ".[dev]"
```

**Requirements:** Python >= 3.10. No external dependencies.

## API

### Normalization Functions

```python
from ariturk import normalize_surface, turkish_lower, turkish_upper, is_valid_turkish
from ariturk.normalize import restore_diacritics
```

| Function | Description | Example |
|----------|-------------|---------|
| `normalize_surface(text)` | NFC + whitespace cleanup | `"  merhaba   dünya  "` → `"merhaba dünya"` |
| `turkish_lower(text)` | Turkish lowercasing | `"ISTANBUL"` → `"ıstanbul"`, `"I"` → `"ı"` |
| `turkish_upper(text)` | Turkish uppercasing | `"istanbul"` → `"İSTANBUL"`, `"i"` → `"İ"` |
| `is_valid_turkish(text)` | Validate Turkish characters | `"merhaba"` → `True`, `"@#$"` → `False` |
| `restore_diacritics(text)` | Restore missing diacritics | `"turkce guzel"` → `"türkçe güzel"` |

#### The Turkish I/İ Problem

Python's standard `str.lower()` and `str.upper()` produce incorrect results for Turkish:

```python
# WRONG — Python standard
"ISTANBUL".lower()        # → "istanbul"   (correct: "ıstanbul")
"istanbul".upper()        # → "ISTANBUL"   (correct: "İSTANBUL")

# CORRECT — ariturk
turkish_lower("ISTANBUL") # → "ıstanbul"
turkish_upper("istanbul") # → "İSTANBUL"
```

| Character | `str.lower()` | `turkish_lower()` | `str.upper()` | `turkish_upper()` |
|-----------|---------------|-------------------|---------------|-------------------|
| I | i | ı | — | — |
| İ | İ | i | İ | İ |
| i | — | — | I | İ |
| ı | — | — | I | I |

### TextCleaner

Configurable text cleaning pipeline:

```python
from ariturk import TextCleaner

# Default: lowercase + normalization only
cleaner = TextCleaner()
cleaner.clean("  MERHABA   Dünya  ")
# → "merhaba dünya"

# Full pipeline
cleaner = TextCleaner(
    lowercase=True,           # Turkish-correct lowercasing
    fix_diacritics=True,      # c→ç, g→ğ, i→ı, o→ö, s→ş, u→ü
    remove_punctuation=True,  # Strip punctuation
    min_word_length=2,        # Drop single-character words
)
cleaner.clean("TURKCE, guzel bir dil!")
# → "türkçe güzel bir dil"

# Batch processing
cleaner.clean_batch(["text 1", "text 2"])
# → ["text 1", "text 2"]

# Cleanliness check
cleaner.is_clean("türkçe metin")  # → True
cleaner.is_clean("TURKCE @@")     # → False
```

**Pipeline order:**
1. `normalize_surface` (NFC + whitespace)
2. `turkish_lower` (if `lowercase=True`)
3. `restore_diacritics` (if `fix_diacritics=True`)
4. Punctuation removal (if `remove_punctuation=True`)
5. Minimum length filter (if `min_word_length > 1`)

### QualityChecker

Quality tier assignment for morphological data:

```python
from ariturk import QualityChecker

qc = QualityChecker()

# Tier assignment
qc.assign_tier(["boun", "zeyrek"], tags_agree=True)
# → "gold"  (BOUN is a human-verified source)

qc.assign_tier(["zeyrek", "trmorph"], tags_agree=True)
# → "silver"  (2+ agreeing sources)

qc.assign_tier(["zeyrek"], tags_agree=False)
# → "bronze"  (single source)
```

| Tier | Condition | Reliability |
|------|-----------|-------------|
| **gold** | From BOUN or IMST source | Highest (human-verified UD treebank) |
| **silver** | >= 2 independent sources AND tags agree | High |
| **bronze** | Single source or cross-source disagreement | Low |

```python
# Entry validation
errors = qc.validate_entry("ev", "ev +PLU +ABL", "NOUN")
# → []  (valid)

errors = qc.validate_entry("ev", "ev PLU", "NOUN")
# → ["Tags (except first) must start with '+': PLU"]
```

### BoundaryExtractor

Extract morpheme boundaries from surface form + canonical tags:

```python
from ariturk import BoundaryExtractor

ext = BoundaryExtractor()

ext.extract("evlerinden", "ev +PLU +POSS.3SG +ABL")
# → "ev|ler|i|nden"

ext.extract("gidiyorum", "git +PROG +1SG")
# → "gid|iyor|um"

# Batch
ext.extract_batch([
    ("evler", "ev +PLU"),
    ("okul", "okul"),
])
# → ["ev|ler", "okul"]
```

## Project Structure

```
ariturk/
├── __init__.py      # Public API exports
├── normalize.py     # turkish_lower/upper, NFC, diacritics
├── cleaner.py       # TextCleaner pipeline
├── quality.py       # QualityChecker tiering
└── boundaries.py    # BoundaryExtractor wrapper
```

## Relationship with kok-turk

ariturk is the text preprocessing layer for the kok-turk morphological atomizer project:

```
Raw text → [ariturk] → Clean text → [kokturk] → Root + Tags
              │                          │
        Normalization            Morphological analysis
        Diacritics fix           Disambiguation
        Quality tiering          Grammar checking
```

ariturk can be used independently — it does not require kokturk.

## License

Apache License 2.0
