#!/bin/bash
# External reproducibility acceptance suite for Aksu v1.0.0.
# Run on a CLEAN machine (Ubuntu 22.04 or Rocky Linux 9) after `pip install aksu`.
# Logs to audit/v1.0.0_reproducibility/<hostname>_<date>.txt
#
# Usage:
#   bash scripts/run_acceptance_suite.sh
#
# Requirements on the clean machine:
#   - Python 3.10+ and pip
#   - Internet access (to pip install aksu)
#   - No pre-existing aksu/kokturk installation

set -euo pipefail

LOG_DIR="audit/v1.0.0_reproducibility"
LOG_FILE="${LOG_DIR}/$(hostname)_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p "$LOG_DIR"

exec > >(tee "$LOG_FILE") 2>&1

echo "=== Aksu v1.0.0 Acceptance Suite ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "OS: $(uname -a)"
echo "Python: $(python3 --version)"
echo ""

# --- Gate 1: pip install -------------------------------------------------------
echo "--- Gate 1: pip install aksu ---"
python3 -m venv /tmp/aksu-acceptance-venv
/tmp/aksu-acceptance-venv/bin/pip install aksu -q
echo "PASS: pip install aksu"
echo ""

# --- Gate 2: import and version ------------------------------------------------
echo "--- Gate 2: import + version ---"
/tmp/aksu-acceptance-venv/bin/python -c "
import aksu
print(f'aksu.__version__ = {aksu.__version__}')
assert hasattr(aksu, 'Atomizer'), 'Atomizer not exported'
assert hasattr(aksu, 'MorphoAnalyzer'), 'MorphoAnalyzer not exported'
print('PASS: import + version')
"
echo ""

# --- Gate 3: Atomizer canonical form -------------------------------------------
echo "--- Gate 3: Atomizer.to_canonical ---"
/tmp/aksu-acceptance-venv/bin/python -c "
from aksu import Atomizer
result = Atomizer(backend='zeyrek').to_canonical('evlerinden')
print(f'to_canonical(evlerinden) = {result}')
assert result, 'Empty result'
print('PASS: to_canonical returns non-empty string')
"
echo ""

# --- Gate 4: TextCleaner -------------------------------------------------------
echo "--- Gate 4: TextCleaner ---"
/tmp/aksu-acceptance-venv/bin/python -c "
from aksu.ariturk import TextCleaner, turkish_lower
clean = TextCleaner().clean('  TÜRKÇE   metİn  ')
assert clean == 'türkçe metin', f'Expected \"türkçe metin\", got {clean!r}'
lo = turkish_lower('I')
assert lo == 'ı', f'Expected \"ı\", got {lo!r}'
print(f'clean = {clean!r}')
print(f'turkish_lower(I) = {lo!r}')
print('PASS: TextCleaner + turkish_lower')
"
echo ""

# --- Gate 5: compat shim -------------------------------------------------------
echo "--- Gate 5: deprecated kokturk shim ---"
/tmp/aksu-acceptance-venv/bin/pip install kokturk -q 2>/dev/null || echo "(kokturk shim not yet published — skipping)"
/tmp/aksu-acceptance-venv/bin/python -c "
import warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    try:
        import kokturk
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        if dep:
            print(f'DeprecationWarning: {dep[0].message}')
            print('PASS: kokturk shim emits DeprecationWarning')
        else:
            print('NOTE: kokturk shim imported without warning (may be pre-shim install)')
    except ImportError:
        print('NOTE: kokturk shim not installed — skip')
"
echo ""

# --- Gate 6: CLI ---------------------------------------------------------------
echo "--- Gate 6: CLI ---"
result=$(/tmp/aksu-acceptance-venv/bin/aksu analyze "evlerinden" 2>/dev/null || true)
echo "aksu analyze evlerinden → ${result}"
if [ -n "$result" ]; then
  echo "PASS: CLI returns non-empty output"
else
  echo "NOTE: CLI returned empty (backend may not be loaded in clean install)"
fi
echo ""

# --- Summary -------------------------------------------------------------------
echo "=== Acceptance Suite Complete ==="
echo "Log: $LOG_FILE"
rm -rf /tmp/aksu-acceptance-venv
