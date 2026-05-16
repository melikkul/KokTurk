# Secrets Inventory — Release Pipeline

All credentials required to run the release pipeline for Aksu v1.0.0. No secret VALUES are stored here — only names, scope, and procurement notes.

**Pre-flight check before tagging v1.0.0:**
```bash
gh secret list --repo melikkul/Aksu
```
Every row in the table below must appear in the output before pushing a release tag.

---

## Required Secrets

| Secret name | Where | Purpose | Procured by | Rotation cadence |
|-------------|-------|---------|-------------|------------------|
| `HF_TOKEN` | repo secret | `huggingface-cli login` for dataset push (D-Step 8) and model push (H-Step 3) | Owner: https://huggingface.co/settings/tokens — create a "write" scope token | Annually or on contributor role change |
| `ZENODO_TOKEN` | repo secret | Zenodo REST API deposit (H-Step 4) | Owner: https://zenodo.org/account/settings/applications/ — create "deposit:actions + deposit:write" token | Rotate immediately after v1.0.0 DOI is minted |
| `AWS_ACCESS_KEY_ID` | repo secret | DVC S3 remote read in CI; write on tag pushes (B-Step 3, G-Step 4) | IAM-scoped read-only key on the `aksu-dvc` bucket prefix | Annually or on contributor role change |
| `AWS_SECRET_ACCESS_KEY` | repo secret | Paired with `AWS_ACCESS_KEY_ID` | Same as above | Same as above |
| `AWS_ACCESS_KEY_ID_WRITER` | environment secret (`pypi`) | DVC remote write on tag-trigger only | Owner — separate IAM key with write scope on `aksu-dvc` bucket | Annually |
| `AWS_SECRET_ACCESS_KEY_WRITER` | environment secret (`pypi`) | Paired with `AWS_ACCESS_KEY_ID_WRITER` | Same | Same |
| PyPI Trusted Publishing | n/a (OIDC, no stored secret) | Publish `aksu` / `kokturk` / `ariturk` to PyPI and TestPyPI on tag | Configured per H-Step 0 instructions below | No rotation needed — OIDC is short-lived |
| `GITHUB_TOKEN` (built-in) | n/a (auto-provided) | GitHub release creation (H-Step 5) | Automatic per workflow run | n/a |

---

## PyPI Trusted Publishing Setup (H-Step 0)

One-time setup; must be done before the `v1.0.0` tag is pushed.

### For `aksu` (main package)

1. Go to https://pypi.org/manage/account/publishing/
2. Add a **pending Trusted Publisher**:
   - **PyPI Project Name**: `aksu`
   - **Owner**: `melikkul`
   - **Repository**: `Aksu`
   - **Workflow filename**: `publish-pypi.yml`
   - **Environment name**: `pypi`
3. Repeat for **TestPyPI** at https://test.pypi.org/manage/account/publishing/ with workflow `publish-testpypi.yml`.

### For `kokturk` and `ariturk` shim packages

1. Add pending publishers for both on PyPI and TestPyPI:
   - **Workflow filename**: `publish-shims.yml`
   - **Environment name**: `pypi-shims`

### Verify OIDC permissions in workflow files

Every publish workflow must include:
```yaml
permissions:
  id-token: write
  contents: read
```
and must NOT include a `password:` field in the `pypa/gh-action-pypi-publish` step.

---

## Notes

- **Forking contributors** get the `--allow-missing` DVC dry-run path (no AWS credentials needed for the dry-run job).
- **Main-repo CI** uses the read-only `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` pair so the DVC DAG can fully resolve outputs.
- **Tag-triggered release** uses `AWS_ACCESS_KEY_ID_WRITER` / `AWS_SECRET_ACCESS_KEY_WRITER` gated inside the `pypi` environment.
- S3 bucket path: `s3://aksu-dvc/` (placeholder until bucket is provisioned; local fallback: `/arf/scratch/scolakoglu/aksu-dvc-cache`).
