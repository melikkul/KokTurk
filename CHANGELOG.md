# Changelog

All notable changes to kokturk will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.5.0] - 2026-04-09

### Added
- **Category A** — Diagnostic infrastructure: tag frequency analysis, FocalLoss/SymmetricCE/LabelSmoothing losses, noise audit (cleanlab seq2seq adapter), paradigm augmentation, domain importers (BounTi, Trendyol, Bilkent), contrastive root head, polysemy evaluation
- **Category B** — Linguistic coverage: fused LVC decomposition, special token preprocessing (abbreviations, numerics, reduplication), morphotactic FSA constraint mask, deep chain & compound evaluation
- **Category C** — Training optimization: R-Drop regularization, variational dropout (locked masks), character augmentation (keyboard/diacritic/stemcorrupt), EMA weights, Optuna HPO with TPE+Hyperband, AdamW optimizer support
- **Category D** — Evaluation & benchmarking: error analysis pipeline with diacritic-aware Levenshtein, weighted-EM metric, robustness perturbation suite, CheckList behavioral tests, speed benchmarking, standard benchmarks (TrMor2018/UD), minimal pairs challenge set, paired bootstrap + Holm-Bonferroni significance tests
- **Category E** — Engineering quality: pyproject.toml with ruff/mypy/pytest config, reproducibility utilities (seed_everything, environment capture), golden regression tests (52 entries), GitHub Actions CI pipeline, MLflow model registry, CHANGELOG

### Changed
- Project renamed from morpho-tr to kokturk
- Label smoothing corrected from epsilon=0.1 to epsilon=0.01 for morphological seq2seq
- Root vocab expanded 3,871 to 63,198 (OOV: 8.2% to 0.6%)
- Reproducibility: seed_everything now sets PYTHONHASHSEED, cudnn.deterministic, cudnn.benchmark

### Fixed
- atomizer_v2 "82%" was val EM — true test EM is 72.68% (9.6pp overfitting gap)
- Training seed handling now covers all random generators (was missing PYTHONHASHSEED and cuDNN flags)
