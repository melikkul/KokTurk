.PHONY: setup test test-all test-regression lint format format-check typecheck ci verify train-local train-smoke benchmark export-onnx dashboard clean tag-freq stratified-eval noise-audit augment import-bounti import-trendyol import-bilkent compound-eval deep-chain-eval constrained-decode-test error-analysis weighted-em robustness-test checklist speed-benchmark standard-benchmarks minimal-pairs significance-test competitor-accuracy llm-baseline corpus-stats cache-benchmark pii-scan domain-bias

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
ACTIVATE := source $(VENV)/bin/activate

setup:
	@echo "=== Setting up kokturk development environment ==="
	@command -v python3 > /dev/null 2>&1 || (echo "ERROR: Python 3 not found. Run: module load comp/python/miniconda3" && exit 1)
	@test -d $(VENV) || python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements/base.txt -r requirements/dev.txt
	$(PIP) install -e .
	@echo "--- Checking optional dependencies ---"
	@which flookup > /dev/null 2>&1 || echo "WARNING: foma-bin not found. TRMorph backend will be disabled."
	@test -f tools/trmorph/trmorph.fst || echo "WARNING: TRMorph FST not found. Run: git clone --branch trmorph2 --depth 1 https://github.com/coltekin/TRmorph.git tools/trmorph"
	@test -f .env || cp .env.example .env && echo "Created .env from .env.example"
	@echo "=== Setup complete ==="

test:
	PYTHONPATH=src $(PYTHON) -m pytest tests/ -m "not gpu and not slow" -v --tb=short --timeout=120

test-all:
	PYTHONPATH=src $(PYTHON) -m pytest tests/ -v --tb=short --timeout=300

test-regression:
	PYTHONPATH=src $(PYTHON) -m pytest tests/regression/ -v --tb=short

lint:
	$(PYTHON) -m ruff check src/ tests/

format:
	$(PYTHON) -m ruff format src/ tests/
	$(PYTHON) -m ruff check --fix src/ tests/

format-check:
	$(PYTHON) -m ruff format --check src/ tests/

typecheck:
	$(PYTHON) -m mypy src/kokturk/ --ignore-missing-imports

ci: format-check lint test

verify: test lint
	@echo "=== Running Phase 0 verification ==="
	$(PYTHON) -c "from kokturk.core.datatypes import MorphologicalAnalysis, TokenAnalyses; print('OK: datatypes import')"
	$(PYTHON) -c "from kokturk.core.analyzer import MorphoAnalyzer; a = MorphoAnalyzer(); r = a.analyze('evlerinden'); print(f'OK: analyzer ({r.parse_count} parses for evlerinden)')"
	$(PYTHON) -c "from kokturk.core.constants import ZEYREK_TO_CANONICAL; assert all(v == '' or v.startswith('+') for v in ZEYREK_TO_CANONICAL.values()); print(f'OK: {len(ZEYREK_TO_CANONICAL)} tag mappings valid')"
	@echo "=== All verifications passed ==="

train-local:
	PYTHONPATH=src $(PYTHON) src/train/train_atomizer.py

train-smoke:
	PYTHONPATH=src $(PYTHON) src/train/smoke_test.py

dashboard:
	PYTHONPATH=src $(PYTHON) src/train/training_dashboard.py

benchmark:
	PYTHONPATH=src $(PYTHON) src/benchmark/intrinsic_eval.py

export-onnx:
	$(PYTHON) src/optimize/export_onnx.py

tag-freq:
	PYTHONPATH=src $(PYTHON) -m benchmark.tag_frequency

stratified-eval:
	PYTHONPATH=src $(PYTHON) -c "from benchmark.stratified_eval import inspect_checkpoint; import sys; [print(p, inspect_checkpoint(p)) for p in __import__('pathlib').Path('models').rglob('best_model.pt')]"

noise-audit:
	PYTHONPATH=src $(PYTHON) -m data.noise_audit --device cpu

augment:
	PYTHONPATH=src $(PYTHON) -c "from resource.importers.synthetic_inflections import import_synthetic_inflections; import_synthetic_inflections('models/benchmark/tag_frequency.json', 'data/gold/synthetic_inflections.jsonl')"

import-bounti:
	PYTHONPATH=src $(PYTHON) -c "from resource.importers.bounti import import_bounti; import_bounti()"

import-trendyol:
	@if [ -z "$${CSV}" ]; then echo "Usage: make import-trendyol CSV=path/to/trendyol.csv"; exit 1; fi
	PYTHONPATH=src $(PYTHON) -c "from resource.importers.trendyol import import_trendyol; import_trendyol('$(CSV)')"

import-bilkent:
	PYTHONPATH=src $(PYTHON) -c "from resource.importers.bilkent import import_bilkent; import_bilkent('data/external/bilkent')"

compound-eval:
	PYTHONPATH=src $(PYTHON) -c "from benchmark.compound_eval import COMPOUND_VERB_ROOTS; print(f'{len(COMPOUND_VERB_ROOTS)} compound verb roots configured: {COMPOUND_VERB_ROOTS}')"

deep-chain-eval:
	PYTHONPATH=src $(PYTHON) -c "from benchmark.stratified_eval import _depth_bucket; print('Depth buckets:', [_depth_bucket(d) for d in range(10)])"

constrained-decode-test:
	PYTHONPATH=src $(PYTHON) -m pytest tests/models/test_morphotactic_mask.py -v --no-cov

error-analysis:
	PYTHONPATH=src $(PYTHON) -c "from benchmark.error_analysis import generate_error_report; generate_error_report('models/benchmark/gold.txt', 'models/benchmark/pred.txt', 'models/benchmark/error_report.md')"

weighted-em:
	PYTHONPATH=src $(PYTHON) -c "from benchmark.weighted_em import ALPHA_LEMMA, BETA_POS, GAMMA_DERIV, LAMBDA_INFL; print(f'Weighted EM coefficients: alpha={ALPHA_LEMMA}, beta={BETA_POS}, gamma={GAMMA_DERIV}, lambda={LAMBDA_INFL}')"

robustness-test:
	PYTHONPATH=src $(PYTHON) -m pytest tests/benchmark/test_robustness.py -v --no-cov

checklist:
	PYTHONPATH=src $(PYTHON) -c "from benchmark.checklist_morpho import generate_mft_tests, generate_inv_tests, generate_dir_tests; print(f'MFT: {len(generate_mft_tests())}, INV: {len(generate_inv_tests())}, DIR: {len(generate_dir_tests())}')"

speed-benchmark:
	PYTHONPATH=src $(PYTHON) -m pytest tests/benchmark/test_speed_benchmark.py -v --no-cov

standard-benchmarks:
	PYTHONPATH=src $(PYTHON) -m pytest tests/benchmark/test_standard_benchmarks.py -v --no-cov

minimal-pairs:
	PYTHONPATH=src $(PYTHON) -c "from benchmark.minimal_pairs import load_pairs; pairs = load_pairs(); print(f'{len(pairs)} minimal pairs across {len({p.phenomenon for p in pairs})} phenomena')"

significance-test:
	PYTHONPATH=src $(PYTHON) -m pytest tests/benchmark/test_significance.py -v --no-cov

competitor-accuracy:
	PYTHONPATH=src $(PYTHON) -m benchmark.competitor_accuracy

llm-baseline:
	PYTHONPATH=src $(PYTHON) -m benchmark.llm_baseline

bert-fasttext-benchmark:
	PYTHONPATH=src $(PYTHON) src/benchmark/run_bert_fasttext_benchmark.py

corpus-stats:
	PYTHONPATH=src $(PYTHON) -m benchmark.corpus_stats --corpus data/splits/train.jsonl

cache-benchmark:
	PYTHONPATH=src $(PYTHON) -c "from kokturk.core.analyzer import MorphoAnalyzer; \
	    a = MorphoAnalyzer(backends=['zeyrek']); a.enable_cache(100000); \
	    words = [line.split('\t')[0] for line in open('data/gold/test.tsv') if line.strip()][:1000]; \
	    _ = [a.analyze(w) for w in words]; _ = [a.analyze(w) for w in words]; \
	    print(a.cache_stats)"

pii-scan:
	PYTHONPATH=src $(PYTHON) -m data.pii_scan

domain-bias:
	PYTHONPATH=src $(PYTHON) -m benchmark.domain_bias

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
