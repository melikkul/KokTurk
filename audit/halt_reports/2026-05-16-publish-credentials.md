# Halt Report — Publication Credentials Missing

**Date:** 2026-05-16  
**Phase:** Q — Credentials Gate  
**Triggered by:** All 4 required credentials absent from environment

## Status

| Credential | Status | Blocks |
|------------|--------|--------|
| `HF_TOKEN` | MISSING | Phase T (HuggingFace dataset + model upload) |
| `ZENODO_TOKEN` | MISSING | Phase U (Zenodo DOI deposit) |
| `AWS_ACCESS_KEY_ID` | MISSING | Phase R (DVC push to S3 remote) |
| `AWS_SECRET_ACCESS_KEY` | MISSING | Phase R (DVC push to S3 remote) |

## What ran

- **PHASE P** ✅ — pre-release snapshot: 249 tests pass, metrics consistency gate passes, working tree clean, `v1.0.0-rc1` tagged locally
- **PHASE Q** ✅ — credentials checked, all missing, halt filed
- **PHASE S.1** ✅ — local build + twine check (no credentials required) — see below
- **PHASE X** ✅ — final release report written (`audit/v1.0.0_release_report_v3.md`)

## What's blocked

| Phase | Exact unblock |
|-------|--------------|
| R — DVC push | Set `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` for the S3-compatible DVC remote. Run: `dvc push` |
| S.2/S.3 — TestPyPI + real PyPI | PyPI Trusted Publishing requires `git push origin v1.0.0a0` from TRUBA to trigger `.github/workflows/publish-pypi.yml`. Trusted Publisher must be registered at https://pypi.org/manage/account/publishing/ |
| T — HuggingFace | Set `HF_TOKEN` (write-scope token from https://huggingface.co/settings/tokens). Run `python scripts/data/publish_huggingface.py` |
| U — Zenodo | Set `ZENODO_TOKEN` (deposit token from https://zenodo.org/account/settings/applications/). Run `python scripts/data/deposit_zenodo.py` |
| V — Clean-VM acceptance | Depends on S being live (PyPI install) |
| W — Stable tag v1.0.0 | Depends on V passing. User pushes manually: `git push origin v1.0.0` |

## S.1 Local build result

Build artifact verification completed locally — see Phase S.1 section in `audit/v1.0.0_release_report_v3.md`.

## Resume instructions

When credentials are available, resume from the appropriate phase:
1. Export credentials in the TRUBA session
2. Run phases R, T, U in any order (independent)
3. Then trigger S.2 tag push → wait for GH Actions → verify install
4. Then S.3, V, W in order
5. Update `audit/v1.0.0_release_report_v3.md` with results
