#!/bin/bash
#===============================================================================
# enforcer_03_pipeline_importable.sh — Rule 03: Pipeline Importable
#===============================================================================
#
# Validates that the gen_gec_errant pipeline module is importable.
# Checks:
#   - python -c 'from gen_gec_errant.pipeline import run_pipeline' succeeds
#===============================================================================

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
CONFIGS_DIR="${CONFIGS_DIR:-$PROJECT_DIR/configs/pipeline}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs}"
MODEL_NAME="${MODEL_NAME:-}"

PASS=0
WARN=0
FAIL=0

log_pass() { echo "  ✓ $1"; PASS=$((PASS + 1)); }
log_warn() { echo "  ⚠ $1"; WARN=$((WARN + 1)); }
log_fail() { echo "  ✗ $1"; FAIL=$((FAIL + 1)); }

echo "=== Rule 03: Pipeline Importable ==="
echo ""

# Check pipeline is importable
if python3 -c "from gen_gec_errant.pipeline import run_pipeline" 2>/dev/null; then
    log_pass "gen_gec_errant.pipeline.run_pipeline importable"
else
    log_fail "Cannot import gen_gec_errant.pipeline.run_pipeline (check PYTHONPATH or installation)"
fi

# Check individual stages importable
STAGES=(data_loader generation gec annotation analysis)
for stage in "${STAGES[@]}"; do
    if python3 -c "import gen_gec_errant.$stage" 2>/dev/null; then
        log_pass "Stage importable: gen_gec_errant.$stage"
    else
        log_fail "Stage not importable: gen_gec_errant.$stage"
    fi
done


echo ""
echo "--- Rule 03 Summary: $PASS passed, $WARN warnings, $FAIL failed ---"
[[ $FAIL -eq 0 ]]
