# Data License Audit — kokturk Morphological Analyzer

**Date:** 2026-04-09
**Scope:** All training data used in the kokturk neural morphological atomizer.

## Purpose

Neural network weights are legally considered derivative works of their training data.
License restrictions on training data propagate to the model. This document audits the
licensing terms of all data sources to identify constraints on model distribution.

## Training Data Inventory

| Dataset | Size | License | SPDX ID | Commercial Use | ShareAlike | Source |
|---------|------|---------|---------|----------------|------------|--------|
| BOUN Treebank | 9,761 sentences (~127K lines) | CC BY-SA 4.0 | CC-BY-SA-4.0 | Yes | **Yes** — derivatives must be open-sourced | [UD GitHub](https://github.com/UniversalDependencies/UD_Turkish-BOUN) |
| IMST Treebank | 5,635 sentences (~52K lines) | CC BY-NC-SA 3.0 | CC-BY-NC-SA-3.0 | **NO** | **Yes** | [UD GitHub](https://github.com/UniversalDependencies/UD_Turkish-IMST) |
| UniMorph (Turkish) | 570,420 forms | CC BY-SA 3.0 | CC-BY-SA-3.0 | Yes | **Yes** | [UniMorph GitHub](https://github.com/unimorph/tur) |
| TTC-3600 | 3,600 documents | CC BY 4.0 | CC-BY-4.0 | Yes | No | Academic publication |
| OSCAR (Turkish) | Billions of tokens | CC0 (annotations) / Mixed (text) | N/A | **Ambiguous** | No | [OSCAR Project](https://oscar-project.org/) |
| Gold tier (project) | 2,496 samples | Project-internal (derived from above) | N/A | Inherits source restrictions | Inherits | Manual + Zeyrek annotation |
| Silver-auto tier | 61,516 samples | Project-internal (derived from above) | N/A | Inherits source restrictions | Inherits | Multi-source agreement |
| Silver-agreed tier | 16,525 samples | Project-internal (derived from above) | N/A | Inherits source restrictions | Inherits | LF agreement |
| Resource DB (bronze) | ~40K training export | Derived from above | N/A | Inherits most restrictive | Inherits | Resource pipeline |

## Critical Findings

### IMST Treebank Blocks Commercial Use

The IMST Treebank is licensed **CC BY-NC-SA 3.0**. The **NonCommercial (NC)** clause
prohibits using the data, or any model trained on it, for commercial purposes.

**Impact:** Any model checkpoint trained with IMST data in the training set CANNOT be:
- Sold as a product or service
- Used in a commercial API
- Deployed in revenue-generating software
- Distributed as part of a proprietary system

**Mitigation:** To enable commercial deployment, retrain the model **excluding all
IMST-derived data**. Use only BOUN + UniMorph + project gold/silver tiers (with IMST
samples removed from derived tiers).

### ShareAlike Propagation (BOUN + UniMorph)

BOUN (CC BY-SA 4.0) and UniMorph (CC BY-SA 3.0) require that derivative works
(including trained model weights) be distributed under the same or compatible license.

**Impact:** Model checkpoints trained on BOUN/UniMorph data must be released as
open-source under CC BY-SA 4.0 (or compatible) if distributed publicly.

### OSCAR Copyright Ambiguity

OSCAR's annotations are CC0 but the underlying web-scraped text retains original
copyright from source websites. Training on OSCAR carries unresolved legal risk,
particularly in jurisdictions with strong copyright enforcement.

**Impact:** Models trained on OSCAR text may face copyright challenges. Legal review
is recommended before any public distribution.

## License Compatibility Matrix

| Scenario | Permitted License | Restriction |
|----------|-------------------|-------------|
| Research use (all data) | Any | None |
| Public release (with IMST) | CC BY-NC-SA 3.0 | Non-commercial only |
| Public release (without IMST) | CC BY-SA 4.0 | Must open-source model weights |
| Commercial deployment | Requires IMST exclusion + SA compliance | Must open-source under BY-SA |

## Recommended License for Model Distribution

- **If IMST included:** CC BY-NC-SA 3.0 (most restrictive source governs)
- **If IMST excluded:** CC BY-SA 4.0 (BOUN's SA requirement)
- **For commercial use:** Retrain on BOUN + UniMorph + TTC-3600 + project gold/silver
  (IMST-free), release under CC BY-SA 4.0

## Attribution Requirements

All distributions must credit:
1. BOUN Treebank: Turk et al. (2022), Universal Dependencies
2. IMST Treebank: Sulubacak et al. (2016), derived from METU-Sabanci Treebank
3. UniMorph: Kirov et al. (2018), UniMorph Project
4. TTC-3600: Kılınc et al. (2017)
5. Any additional data sources per their respective licenses
