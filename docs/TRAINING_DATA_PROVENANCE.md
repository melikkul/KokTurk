# Training Data Provenance — kokturk Morphological Analyzer

**Date:** 2026-04-09

## Purpose

This document records the origin, processing, and usage of all data sources in the
kokturk morphological analyzer pipeline. Each entry provides a chain of custody from
source to training-ready format.

---

## External Data Sources

### BOUN Treebank (Universal Dependencies)

- **Origin:** https://github.com/UniversalDependencies/UD_Turkish-BOUN
- **Citation:** Turk et al. (2022). "Resources for Turkish Dependency Parsing: Introducing the BOUN Treebank and the BoAT Annotation Tool." Language Resources and Evaluation.
- **License:** CC BY-SA 4.0
- **Format:** CoNLL-U
- **Download:** Cloned to `data/external/boun_treebank/`
- **Files:** `tr_boun-ud-train.conllu` (~127K lines), `tr_boun-ud-dev.conllu`, `tr_boun-ud-test.conllu`
- **Sentences:** 9,761 (train + dev + test)
- **Preprocessing:** Parsed with CoNLL-U reader; lemma + UPOS + features extracted; converted to canonical tag format via `ud_to_canonical` mapping in `src/benchmark/standard_benchmarks.py`
- **Used in:** Gold tier seed data, silver tier via labeling functions, evaluation benchmarks
- **Known issues:** Contains biographical texts with real names (public figures); some annotation inconsistencies in derivational morphology

### IMST Treebank (Universal Dependencies)

- **Origin:** https://github.com/UniversalDependencies/UD_Turkish-IMST
- **Citation:** Sulubacak et al. (2016). "Imst: A revisited Turkish dependency treebank." TurCLing.
- **License:** CC BY-NC-SA 3.0 (NonCommercial)
- **Format:** CoNLL-U
- **Download:** Cloned to `data/external/imst_treebank/`
- **Files:** `tr_imst-ud-train.conllu` (~52K lines), `tr_imst-ud-dev.conllu`, `tr_imst-ud-test.conllu`
- **Sentences:** 5,635 (train + dev + test)
- **Preprocessing:** Same CoNLL-U pipeline as BOUN
- **Used in:** Gold tier seed, silver tier, evaluation
- **Known issues:** Derived from METU-Sabanci Treebank (news domain); NC license blocks commercial use; contains news text with public figure names (Erdogan, Arinc, etc.)

### UniMorph (Turkish)

- **Origin:** https://github.com/unimorph/tur
- **Citation:** Kirov et al. (2018). "UniMorph 2.0: Universal Morphology." LREC.
- **License:** CC BY-SA 3.0
- **Format:** 3-column TSV (lemma, inflected form, feature bundle)
- **Download:** Cloned to `data/external/unimorph_tur/`
- **Files:** `tur` (570,420 inflected forms), `tur.derivations`
- **Preprocessing:** Feature bundles converted to canonical tag sequences
- **Used in:** Labeling functions (as reference lexicon), paradigm augmentation
- **Known issues:** Generated from FST; may contain forms not attested in natural text; some feature labels differ from UD conventions

### TTC-3600

- **Origin:** Academic dataset (Kılınc et al., 2017)
- **Citation:** Kılınc et al. (2017). "TTC-3600: A new benchmark dataset for Turkish text categorization." Journal of Information Science.
- **License:** CC BY 4.0
- **Format:** Text documents organized by category
- **Download:** `data/external/ttc3600/` and `data/external/ttc3600_raw/`
- **Documents:** 3,600 (6 categories x 600 documents)
- **Preprocessing:** Text extraction, sentence splitting, tokenization
- **Used in:** Classification experiments (TTC-3600 benchmark), corpus statistics
- **Known issues:** News domain dominated; some documents have encoding artifacts

### OSCAR (Turkish subset)

- **Origin:** https://oscar-project.org/
- **License:** CC0 (annotations); underlying text retains original copyright
- **Format:** JSONL
- **Used in:** Reference corpus for frequency statistics only (not directly in training)
- **Known issues:** Web-scraped text; copyright of underlying content is ambiguous; may contain PII from web pages

---

## Project-Generated Data

### Gold Tier

- **Location:** `data/gold/`
- **Files:** `combined_gold.jsonl` (250 samples), `tr_gold_morph_v1.jsonl` (full export)
- **Total samples:** 2,496
- **Generation:** Manual annotation via Prodigy + Zeyrek verification + active learning selection
- **Quality:** Human-verified; highest confidence
- **Derived from:** BOUN + IMST treebank sentences (inherits their license restrictions)

### Silver-Auto Tier

- **Total samples:** 61,516
- **Generation:** Multi-source automatic agreement (Zemberek + Zeyrek + TRMorph majority vote via skweak HMM)
- **Quality:** High confidence (3-source agreement)
- **Derived from:** BOUN + IMST + UniMorph (inherits restrictions)

### Silver-Agreed Tier

- **Total samples:** 16,525
- **Generation:** Labeling function agreement
- **Quality:** Medium-high confidence
- **Derived from:** Same sources as silver-auto

### Resource DB (Bronze)

- **Location:** `data/resource/`
- **Files:** `training_export.jsonl` (~40K records), `tr_gold_morph.db` (SQLite)
- **Generation:** Automated pipeline with single-source labels
- **Quality:** Lower confidence; suitable for pre-training and curriculum warm-up
- **Derived from:** All external sources above

---

## Data Flow

```
External Sources           Pipeline Stages              Output Tiers
─────────────────         ─────────────────            ──────────────
BOUN (CoNLL-U)    ──┐
IMST (CoNLL-U)    ──┼──→  Ingest → Prelabel ──→  Bronze (resource DB)
UniMorph (TSV)    ──┤         │
TTC-3600 (text)   ──┘         ▼
                         Label Model (skweak) ──→  Silver-auto / Silver-agreed
                              │
                              ▼
                         Active Learning ──→  Gold (human-verified)
                              │
                              ▼
                         Training Export ──→  Model checkpoints
```

## Preprocessing Modules

| Stage | Module | Description |
|-------|--------|-------------|
| Ingest | `src/data/ingest.py` | Raw file reading, format normalization |
| Prelabel | `src/data/prelabel.py` | Multi-source prelabeling |
| Label Model | `src/data/label_model.py` | skweak HMM aggregation |
| Noise Audit | `src/data/noise_audit.py` | Cleanlab quality flagging |
| Silver Correction | `src/data/silver_correction.py` | Heuristic NOUN/ADJ fixes |
| Augmentation | `src/data/paradigm_augmentation.py` | Rarity-targeted inflection generation |
| Char Augmentation | `src/data/char_augmentation.py` | Keyboard/diacritic/stemcorrupt noise |
