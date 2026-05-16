# Honesty Cleanup Pre-flight — Phase 0

Date: 2026-05-16

## Template Variables Found

```
{{ "%.1f"|format(metrics.classification_macro_f1_atomized_berturk * 100) }}  ← UNMEASURED (H-1)
{{ "%.1f"|format(metrics.dualhead_em * 100) if metrics.dualhead_em else "84.7" }}  ← FALLBACK (H-3)
{{ "%.1f"|format(metrics.em_argmax_ensemble * 100) }}  ← measured (0.983)
{{ "%.1f"|format(metrics.em_string_ensemble * 100) }}  ← null, will crash if called
{{ "%.2f"|format(metrics.em_string_single_seed_max * 100) if metrics.em_string_single_seed_max else "98.28" }}  ← FALLBACK (H-3)
{{ "%.2f"|format(metrics.em_string_single_seed_min * 100) if metrics.em_string_single_seed_min else "97.98" }}  ← FALLBACK (H-3)
{{ "%.3f"|format(metrics.classification_macro_f1_atomized_berturk) }}  ← UNMEASURED (H-1)
{{ metrics.berturk_sent_per_sec }}  ← null, renders "None" in table
{{ metrics.training_wall_clock_min }}  ← UNMEASURED (H-2), value=14 from STATUS.md
{{ metrics.zeyrek_tok_per_sec }}  ← login-node (H-2 variant), value=1209
```

## metrics.json Violations

| Key | Current value | Status |
|-----|--------------|--------|
| `classification_macro_f1_atomized_berturk` | `0.947` | UNMEASURED — original README value |
| `training_wall_clock_min` | `14` | UNMEASURED — original STATUS.md value |
| `zeyrek_tok_per_sec` | `1209` | login-node only; Orfoz authoritative run pending |
| `em_argmax_ensemble` | `0.9830` | MEASURED (from ensemble_results.json) |
| `dataset_v1_entries` | `80537` | MEASURED (from tr_gold_morph_v1_stats.json) |

## Template Fallback Violations (H-3)

Lines using `else "<number>"` that bypass honesty rule:
- `em_string_single_seed_min` → fallback `"97.98"`
- `em_string_single_seed_max` → fallback `"98.28"`
- `dualhead_em` → fallback `"84.7"`

## Architecture Diagram (H-4)

Diagram uses `"ev +PLU +POSS.3SG +ABL"` (legacy format).
Table above it uses `"ev +Noun +POSS.3PL +ABL"` (current format).
Both refer to same word "evlerinden" — inconsistent.

## Actions Required

1. Strip metrics.json of unmeasured values (H-1, H-2)
2. Remove fallback hardcodes from template (H-3)
3. Remove TTC-3600 table entirely (R8 resolution)
4. Fix architecture diagram to match code output (H-4)
5. Add diagram-consistency test
6. Re-render README and commit
