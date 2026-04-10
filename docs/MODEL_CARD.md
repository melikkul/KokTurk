# Model Card: kokturk Turkish Morphological Analyzer

## Model Details

| Field | Value |
|-------|-------|
| **Name** | kokturk Neural Morphological Atomizer |
| **Version** | 0.5.0 |
| **Date** | 2026-04-09 |
| **Architecture** | Char-level GRU Seq2Seq / Dual-Head with optional context encoder |
| **Parameters** | 2.25M (teacher) / 146K (Conv1D student) |
| **Task** | Character-level morphological analysis: surface form → root + ordered suffix tags |
| **Language** | Turkish (tr) — Istanbul standard |
| **Framework** | PyTorch |
| **License** | CC BY-NC-SA 3.0 (due to IMST training data — see DATA_LICENSE_AUDIT.md) |

## Intended Use

- Linguistic research and morphological analysis pipelines
- Dependency parsing preprocessing
- Academic NLP experimentation
- Turkish text normalization and lemmatization
- Educational tools for Turkish morphology

## Out of Scope

- **Commercial deployment** — blocked by IMST NC license (see DATA_LICENSE_AUDIT.md)
- Processing unredacted clinical or legal documents (PII memorization risk)
- Real-time production serving without caching layer (see Category G)
- Dialectal Turkish analysis (trained on Istanbul standard only)
- Anatolian, Rumelian, and diaspora Turkish variants
- Safety-critical applications

## Training Data

Full details: [DATA_LICENSE_AUDIT.md](DATA_LICENSE_AUDIT.md) and [TRAINING_DATA_PROVENANCE.md](TRAINING_DATA_PROVENANCE.md).

| Source | Samples | License | Domain |
|--------|---------|---------|--------|
| BOUN Treebank | 9,761 sentences | CC BY-SA 4.0 | Formal/mixed |
| IMST Treebank | 5,635 sentences | CC BY-NC-SA 3.0 | News |
| UniMorph (Turkish) | 570,420 forms | CC BY-SA 3.0 | Lexicon |
| TTC-3600 | 3,600 documents | CC BY 4.0 | News categories |
| Gold tier (project) | 2,496 samples | Project-internal | Mixed |
| Silver tier (project) | 78,041 samples | Project-internal | Mixed |

- Root vocabulary: 7,732 unique roots
- Tag vocabulary: 71 unique tags
- Total training tokens: 80,537

## Evaluation Results

### Primary Metrics (Full Test Set)

| Metric | Value |
|--------|-------|
| Exact Match (EM) | 84.45% |
| Root Accuracy | 90.63% |
| Tag F1 | 91.48% |

### Comparison to Published Systems

| System | EM | Source |
|--------|-----|--------|
| MorseDisamb | 98.59% | Seker & Eryigit 2017 |
| Morse | 97.67% | Seker & Eryigit 2017 |
| TransMorph | 96.25% | Akyurek et al. 2022 |
| SIGMORPHON baseline | 92.27% | McCarthy et al. 2019 |
| **kokturk (ours)** | **84.45%** | This project |
| GPT-4o | ~36.7% | Research estimate |

Note: Comparison is approximate — test sets and tag schemas differ across systems.

## Known Limitations & Biases

1. **Domain bias**: Trained primarily on formal text (BOUN + IMST treebanks, both
   formal/news domain). Expected 15-25% accuracy drop on informal/social media text
   based on robustness suite measurements.
2. **Frequency bias**: Rare morphological chains (5+ suffixes) have lower accuracy
   than common patterns. HIGH-frequency tags achieve near-perfect accuracy; RARE tags
   drop significantly.
3. **No dialectal coverage**: Only Istanbul standard Turkish. Anatolian dialects,
   Rumelian variants, and diaspora Turkish are untested and likely underperform.
4. **OOV handling**: Novel roots not in training vocabulary may produce incorrect
   analyses. Partially mitigated by character-level architecture.
5. **Ambiguity**: ~18% of Turkish tokens are morphologically ambiguous in isolation.
   Context-aware models (ContextualDualHeadAtomizer) partially address this.

## Privacy Considerations

- **Memorization capacity**: 2.25M parameters ≈ 8.1M bits. Character-level
  autoregressive decoder can theoretically reconstruct short strings from training data.
- **Training data contents**: BOUN contains biographical texts with real names.
  IMST contains news articles referencing public figures.
- **PII scan**: Heuristic PII scan performed on all training data directories.
  Results documented in PII audit report. Scanner checks for: Turkish national ID
  numbers, phone numbers, email addresses, IBANs.
- **Mitigation**: No DP-SGD or differential privacy applied (out of current scope).
  Users should not use this model to extract personal information.

## Environmental Impact

- **Training hardware**: CPU cluster (v6 disambiguator: 14 min on 56-core node)
- **Estimated training time**: <1 hour for full disambiguation training campaign
- **Carbon footprint**: Minimal — CPU-only training, no GPU required
- **Inference**: CPU-only inference supported via Conv1D student model (<0.5ms/token)

## Ethical Considerations

- **License compliance**: IMST's NC clause is the primary ethical/legal constraint.
  See DATA_LICENSE_AUDIT.md for full analysis.
- **No demographic bias testing**: Morphological analysis is word-level, not
  identity-level. Standard demographic fairness metrics do not directly apply.
  Domain bias (formal vs. informal) is the relevant fairness axis and is measured.
- **Dual use**: Morphological analysis is a low-risk NLP task. No foreseeable
  harmful applications specific to this model.

## Citation

```bibtex
@misc{kokturk2026,
  title={kokturk: Neural Morphological Atomizer for Turkish},
  year={2026},
  note={Work in progress}
}
```
