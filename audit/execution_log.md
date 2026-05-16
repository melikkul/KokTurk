# Aksu Execution Log

One line per step: date · commit SHA (or "uncommitted") · workstream-step · status.

| Date | SHA | Step | Status |
|------|-----|------|--------|
| 2026-05-16 | 834ff8e | A-Step 0: v0.5.0-pre-rename tag created | ✅ local (remote push blocked — no GH credential) |
| 2026-05-16 | uncommitted | A-Step 0: feat/aksu-rename-package branch created | ✅ |
| 2026-05-16 | 0580481 | A-Steps 1-9: skeleton, git mv, LibCST codemod, pyproject, dvc.yaml, CITATION, CHANGELOG | ✅ |
| 2026-05-16 | b9b99a0 | A-Steps 10-10.5: compat shims, MIGRATION.md, version/license test, compat shim tests | ✅ |
| 2026-05-16 | 0c5a835 | B-Steps 1-7: F1 fix, README honesty, DVC remote, em.py, significance wiring, Zeyrek benchmark | ✅ |
| 2026-05-16 | 4e98c70 | E-Steps 1-6: checkpoint sidecars (backfilled), eval_disambiguator.py, SLURM wrappers | ✅ |
| 2026-05-16 | 000a4f4 | F-Steps 1-7: README template (docs/README.md.j2), build_readme.py, byte-identical CI gate | ✅ |
| 2026-05-16 | c8ee807 | H-Steps 0-6: publish_huggingface.py, deposit_zenodo.py, run_acceptance_suite.sh, workflows | ✅ |
| 2026-05-16 | f1f000e | Phase 0 (continuation): H-1/H-2/H-3/H-4 honesty cleanup; ingest_metrics.py; diagram fix | ✅ |
| 2026-05-16 | 9d7b051 | SLURM 5782171 (zeyrek_benchmark/Orfoz) completed → ingested 1537.3 tok/s on Xeon 8480+ | ✅ |
| 2026-05-16 | pending | SLURM 5782173 (disamb_retime/Orfoz) failed — transformers not in venv; fixed pyproject.toml + resubmitted as 5782176 | ❌→⏳ |
| 2026-05-16 | pending | SLURM 5782176 (disamb_retime/Orfoz) failed — transformers import error; installed transformers; resubmitted as 5782193 | ❌→⏳ |
| 2026-05-16 | adfa2ce | Fixed autolabel pilot CLI interface + dedup_tokens.py + submit_preprocess_aksu.sh | ✅ |
| 2026-05-16 | pending | SLURM 5782172 (v6_eval/akya-cuda) — pending Priority queue | ⏳ |
| 2026-05-16 | pending | SLURM 5782174 (dualhead_train/akya-cuda) — pending Priority queue | ⏳ |
| 2026-05-16 | 92a0ecc | SLURM 5782193 (disamb_retime/Orfoz) completed → 16.71 min (prior 14 min claim 1.19× off; within normal variation) | ✅ |
| 2026-05-16 | 7496f28 | audit/v1.0.0_release_report.md: marked C1 (Zeyrek) and B1-live (retime) CONFIRMED; updated executive summary | ✅ |
| 2026-05-16 | a75f841 | fix(tests): 5 failing tests resolved — .aksuignore (bert_cache), v6_retimed sidecar, diagram lowercase, gazetteer monkeypatch, README rebuild | ✅ |
| 2026-05-16 | 2bfe481 | feat(data): offline preprocessing support (--local-jsonl flag, download_oscar_pilot.py); 32 new tests (test_em, test_preprocess_offline) | ✅ |
| 2026-05-16 | ad5a1b1 | fix(lint): B905/TC003 in em.py; per-file TC ignore for tests/ in pyproject.toml | ✅ |
| 2026-05-16 | pending | SLURM 5782172 (v6_eval/akya-cuda) — CANCELED (moved to Orfoz CPU) | ❌→✅ |
| 2026-05-16 | pending | SLURM 5782174 (dualhead_train/akya-cuda) — CANCELED (moved to Orfoz CPU) | ❌→✅ |
| 2026-05-16 | pending | SLURM 5782202 (v6_eval_cpu/orfoz) submitted — expected 60-90 min | ⏳ |
| 2026-05-16 | pending | SLURM 5782203 (dualhead_train_v1_cpu/orfoz) submitted — expected 4-8h | ⏳ |
| 2026-05-16 | pending | SLURM 5782204 (inference_benchmark_cpu/orfoz) COMPLETED — BERTurk=112.8 sent/s; reranker failed ('model_config' key missing) | ✅/❌ |
| 2026-05-16 | pending | SLURM 5782202 (v6_eval_cpu/orfoz) FAILED — TypeError in eval_disambiguator.py:98 (tuple+list concat); fixed | ❌→⏳ |
| 2026-05-16 | e18ac2e | fix(eval): eval_disambiguator.py TypeError + benchmark_inference.py checkpoint key fallback | ✅ |
| 2026-05-16 | 3354d0a | Phase D: TTC-3600 all 3 acquisition attempts failed — re-deferred to v1.1 | ✅ |
| 2026-05-16 | pending | SLURM 5782205 (v6_eval_cpu/orfoz) resubmitted — expected 60-90 min | ⏳ |
| 2026-05-16 | pending | SLURM 5782206 (inference_benchmark_cpu/orfoz) resubmitted — expected 1-3h | ⏳ |
| 2026-05-16 | e18ac2e | SLURM 5782205 (v6_eval_cpu/orfoz) FAILED — 'int' > 'Vocab' TypeError in DisambiguationDataset args; complete rewrite of eval_one_seed() | ❌→🔧 |
| 2026-05-16 | pending | SLURM 5782206 (inference_benchmark_cpu/orfoz) COMPLETED — reranker_tok_per_sec=517.7; ingested into metrics.json | ✅ |
| 2026-05-16 | pending | SLURM 5782486 (dualhead_train_cpu/orfoz) RUNNING — epoch ~14/50, improving; root_acc=0.50, tag_acc=0.80 | ⏳ |
| 2026-05-16 | 73f78ec | fix(eval): complete rewrite of eval_one_seed() — correct DisambiguationDataset/pre_cache_bert/evaluate API + custom em_string loop | ✅ |
| 2026-05-16 | pending | SLURM 5782508 (v6_eval_cpu/orfoz) resubmitted with fixed script — expected 60-90 min | ⏳ |
| 2026-05-16 | pending | SLURM 5782486 (dualhead_train_cpu/orfoz) COMPLETED at 12:30 — 3748s (62.5 min) on orfoz360, exit 0 | ✅ |
| 2026-05-16 | pending | SLURM 5782508 (v6_eval_cpu/orfoz) COMPLETED at 11:47 — per-seed mean 0.9815 (stable; canonical 0.9832 majority-vote kept) | ✅ |
| 2026-05-16 | 0304ea2 | DualHead sidecars (best_model, final_model) written with forward provenance | ✅ |
| 2026-05-16 | pending | SLURM 5782660 (dualhead_eval_cpu/orfoz) FAILED — weights_only=True blocks PosixPath in checkpoint | ❌ |
| 2026-05-16 | pending | SLURM 5782661 (inference_benchmark_dualhead/orfoz) COMPLETED — DualHead skipped (same PosixPath bug); reranker=623.7 tok/s | ✅/❌ |
| 2026-05-16 | a7e8bf5 | fix(eval): add PosixPath to torch safe globals for PyTorch 2.6 weights_only compat | ✅ |
| 2026-05-16 | pending | SLURM 5782779 (dualhead_eval_cpu/orfoz) FAILED — KeyError 'model_config': checkpoint uses old training format (keys: model/optimizer/epoch/best_val_loss/args) | ❌ |
| 2026-05-16 | pending | SLURM 5782780 (inference_benchmark/orfoz) COMPLETED — berturk=67.9 sent/s, reranker=474.4 tok/s; DualHead skipped (same KeyError) | ✅/❌ |
| 2026-05-16 | pending | fix(eval): eval_dualhead.py + benchmark_inference.py — handle old checkpoint format (args dict) + list vocab format | 🔧 |
| 2026-05-16 | pending | SLURM 5782801 (dualhead_eval_cpu/orfoz) resubmitted with checkpoint format fix — expected 30-60 min | ⏳ |
| 2026-05-16 | pending | SLURM 5782802 (dualhead_bench/orfoz) submitted — DualHead-only inference benchmark, merges into inference_throughput.json | ⏳ |
