# Migration Guide: kokturk → aksu (v0.5.x → v1.0.0)

## TL;DR

```bash
pip install -U aksu
```

Existing `from kokturk import ...` and `from ariturk import ...` imports **keep working** with a `DeprecationWarning` through all v1.x releases. Update them when convenient; they will be removed in v2.0.

---

## Symbol-Level Rename Table

| Old import | New import |
|------------|-----------|
| `from kokturk import MorphoAnalyzer` | `from aksu import MorphoAnalyzer` |
| `from kokturk import Atomizer` | `from aksu import Atomizer` |
| `from ariturk import TextCleaner, turkish_lower` | `from aksu import TextCleaner, turkish_lower` |
| `from ariturk import turkish_upper` | `from aksu import turkish_upper` |
| `from kokturk.core.analyzer import MorphoAnalyzer` | `from aksu.kokturk.core.analyzer import MorphoAnalyzer` |
| `from kokturk.sklearn_ext import MorphoTransformer` | `from aksu.kokturk.sklearn_ext import MorphoTransformer` |
| `from kokturk.hf_integration import ...` | `from aksu.kokturk.hf_integration import ...` |
| `from kokturk.models.char_gru import MorphAtomizer` | `from aksu.kokturk.models.char_gru import MorphAtomizer` |
| `from kokturk.models.dual_head import DualHeadAtomizer` | `from aksu.kokturk.models.dual_head import DualHeadAtomizer` |
| `from ariturk.boundaries import BoundaryExtractor` | `from aksu.ariturk.boundaries import BoundaryExtractor` |
| `from ariturk.quality import QualityChecker` | `from aksu.ariturk.quality import QualityChecker` |
| `python -m kokturk.cli.main analyze "X"` | `aksu analyze "X"` |

---

## Behavior Changes

- **Console script**: `python -m kokturk.cli.main` is replaced by the `aksu` console script (`pip install aksu` installs it automatically). Both work in v1.x.
- **Version**: `aksu.__version__` returns `"1.0.0a0"`. The old per-subpackage `"0.1.0"` is gone; `aksu.kokturk.__version__` now returns the same central version string.
- **License**: Code is MIT (was mis-declared as Apache-2.0 in pyproject.toml — corrected in v1.0.0). No behavioral change.
- **EM metric**: The headline benchmark now reports `em_string` (cross-system comparable string equality) alongside `em_argmax` (intra-system candidate-index accuracy). See `src/aksu/benchmark/em.py`.

---

## Deprecation Timeline

| Package | Status in v1.x | Removed |
|---------|---------------|---------|
| `kokturk` (top-level) | Works, emits `DeprecationWarning` | v2.0 |
| `ariturk` (top-level) | Works, emits `DeprecationWarning` | v2.0 |

---

## Bug-Fix Call-Outs

- **F1 path bug fixed**: The disambiguator backend init previously raised `NameError: name 'Path' is not defined` at line 360 of `analyzer.py` because `pathlib.Path` was only imported inside a different method. This is fixed in v1.0.0; the disambiguator backend now initializes correctly.

---

## Provenance

- Pre-rename tag: `v0.5.0-pre-rename` (git tag, recoverable from history)
- Rename PR: `feat/aksu-rename-package`
- CHANGELOG: [`[1.0.0a0]`](../CHANGELOG.md#100a0---2026-05-16)
