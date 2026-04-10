# Contributing to kokturk — Katkı Rehberi

## Welcome / Hoş Geldiniz

We welcome contributions from everyone regardless of experience level,
background, or identity. This project follows the
[Contributor Covenant](https://www.contributor-covenant.org/) Code of Conduct.

## Getting Started / Başlangıç

### Setup

```bash
git clone https://github.com/your-org/kokturk.git
cd kokturk
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
make test  # verify everything works
```

### Project Structure

- `src/kokturk/` — core library (analyzer, models, phonology)
- `src/train/` — training scripts and utilities
- `src/benchmark/` — evaluation and benchmarking
- `src/data/` — data processing and augmentation
- `tests/` — test suite (900+ tests)
- `configs/` — Hydra YAML configurations

## How to Contribute / Nasıl Katkıda Bulunulur

### Good First Issues / Kolay Başlangıç

Look for issues labeled `good-first-issue` — these are scoped for newcomers.

### Code Style

- Run `make lint` before submitting (ruff)
- Run `make format` to auto-fix formatting
- Run `make test` to verify no regressions
- Google-format docstrings
- Type hints for all public functions (`list[str]`, `X | None` style)
- Frozen dataclasses (`frozen=True, slots=True`) for all public output types

### Turkish-Specific Notes / Türkçe'ye Özel Notlar

- **Never** use Python `str.lower()` / `str.upper()` on Turkish text — use
  `ariturk.normalize.turkish_lower` / `turkish_upper` for correct İ↔i and I↔ı.
- Vowel harmony produces 8+ allomorphic variants per suffix; the atomizer
  must canonicalize (e.g. `-da/-de/-ta/-te` → `+LOC`).
- Tags are **ordered sequences**, never bags — Turkish morphotactics are
  non-commutative.

### Pull Request Process

1. Fork and create a feature branch
2. Make changes with conventional commits (`feat:`, `fix:`, `test:`, `docs:`)
3. Run `make ci` locally (format-check + lint + test)
4. Submit PR with description of changes

### Running Tests

```bash
make test          # Core tests (skip gpu/slow markers)
make test-all      # All tests including slow/gpu
make ci            # Full local CI gate (format + lint + test)
```

## Labels / Etiketler

| Label | Description |
|-------|-------------|
| `good-first-issue` | Kolay başlangıç — newcomer friendly |
| `bug` | Hata düzeltme |
| `enhancement` | Yeni özellik |
| `docs` | Belgelendirme |
| `linguistics` | Dil bilimi uzmanlığı gerektiren |
| `performance` | Performans iyileştirme |
