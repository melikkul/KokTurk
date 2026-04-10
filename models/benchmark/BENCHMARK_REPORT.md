# Benchmark Report — Neural Morphological Atomizer

## 1. Open-Vocabulary Morphological Analysis (Context-Free)

> **Note:** Our model performs open-vocabulary morphological ANALYSIS — generating the complete decomposition character-by-character from surface forms without an FST candidate generator or sentential context. This is fundamentally different from morphological DISAMBIGUATION (selecting among FST-generated candidates with context), which is the task measured by Sak et al. (2007, 96.8%) and Morse (Akyurek et al., 2019, 98.6%). Direct numerical comparison between these tasks is not appropriate. For context-free analysis systems, Yildiz et al. (2016) report 84.12% on ambiguous tokens, placing our results in the expected range.

| Metric | Score |
|--------|-------|
| Exact Match | 82.0% |
| Root Accuracy | 85.0% |
| Tag F1 | 93.5% |
| Tag Precision | 93.0% |
| Tag Recall | 93.9% |

Error breakdown: 1220 root errors, 249 tag errors (out of 8140 tokens)

## 2. TTC-3600 Classification (5-fold CV)

| Method | Macro-F1 | Note |
|--------|----------|------|
| ttc3600_original_tfidf | 0.916 ± 0.007 | UCI pre-computed TF-IDF (7508 features) + LogReg |
| ttc3600_zemberek_tfidf | 0.903 ± 0.009 | Zemberek-stemmed TF-IDF + LogReg |
| tfidf_raw | 0.176 ± 0.020 |  |
| logreg_morph_features | 0.172 ± 0.013 | TF-IDF + POS/case/tense distributions + lexical diversity |
| ensemble_classifier | 0.169 ± 0.014 | Soft voting: LogReg + CalibratedSVC + SGD |
| logreg_atomized_bigram | 0.167 ± 0.015 | Optimized TF-IDF (50K features, max_df=0.95, min_df=2) |
| tfidf_atomized | 0.166 ± 0.015 |  |
| logreg_stacked_features | 0.164 ± 0.019 | Word bigrams + char 3-5 grams, 50K features total |

## 3. Efficiency

| System | Params | Size | Speed (tok/s) |
|--------|--------|------|---------------|
| zeyrek | — | rule-based | 6,380 |
| gru_atomizer | 2,255,487 | 8.6 MB | — |

## 4. BPE Failure Cases

| # | Surface | BPE Tokens | Our Atoms | Issue |
|---|---------|-----------|-----------|-------|
| 1 | evlerinden | ev ##ler ##inden | ev +PLU +POSS.3SG +ABL | BPE splits suffix chain arbitrarily |
| 2 | gidiyordum | gidi ##yor ##dum | git +PROG +PAST | Root allomorphy (git→gid) invisible to BPE |
| 3 | okuduklarımız | oku ##duk ##ları ##mız | oku +PASTPART +PLU +POSS.1PL | BPE fragments cross morpheme boundaries |
| 4 | güzelleştirmek | güzel ##leş ##tir ##mek | güzel +BECOME +CAUS +INF | Derivation chain split without linguistic meaning |
| 5 | görmüyorsunuz | gör ##müyor ##sunuz | gör +NEG +PROG +2PL | BPE merges negation with tense |
| 6 | okullarımızdaki | okul ##ları ##mız ##daki | okul +PLU +POSS.1PL +LOC +REL | 5-suffix chain only partially captured by BPE |
| 7 | kitapçılardan | kitap ##çı ##lar ##dan | kitap +AGT +PLU +ABL | BPE fragments have no linguistic labels |
| 8 | sevmediklerimizden | sev ##me ##dik ##leri ##miz ##den | sev +NEG +PASTPART +PLU +POSS.1PL +ABL | Same token count but BPE has no semantic labels |
| 9 | başarısızlaştırılmak | başarı ##sız ##laş ##tır ##ıl ##mak | başarı +WITHOUT +BECOME +CAUS +PASS +INF | 6-derivation chain: atoms are linguistically meaningful |
| 10 | Çekoslovakyalılaştıramadıklarımızdan | [UNK] | Çekoslovakya +BECOME +CAUS +ABIL +NEG +PASTPART +PLU +POSS.1PL +ABL | BPE fails completely on extreme agglutination |

## 5. Error Analysis

### 5.1 Root Identification Errors (15.0% of test tokens)
- Consonant mutation reversal failures: kitab->kitap, sokag->sokak
- Vowel deletion reconstruction: burn->burun, agz->agiz
- Loanword phonotactic violations: robot, saat (resist Turkish phonology)
- Compound root opacity: kahvalti != kahve+alti for the model

### 5.2 Tag Sequence Errors (3.1% of test tokens)
- Accusative/possessive confusion: -(y)I marks both ACC and POSS.3SG
- Aorist/progressive overlap: both encode habitual aspect
- Derivational suffix chains: change POS, creating cascading errors

### 5.3 Gap Decomposition

| Factor | Estimated EM impact | Evidence |
|--------|-------------------|---------|
| No sentential context | 5-8 points | Morse (contextual) 98.6% vs word-level ~84% |
| Silver data noise (~97%) | 3-5 points | Zeyrek trained on auto-generated data |
| GRU vs Transformer | 2-3 points | SIGMORPHON shared task evidence |
| No copy mechanism | 1-2 points | Roots are substrings of input |
