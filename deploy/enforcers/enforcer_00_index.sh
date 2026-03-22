#!/bin/bash
#===============================================================================
# enforcer_00_index.sh — Rule 00: Enforcer Index
#===============================================================================
#
# Validates that the enforcer framework itself is intact.
# Checks:
#   - All enforcer scripts exist (one per rule)
#   - No enforcer is empty or zero-length
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

echo "=== Rule 00: Enforcer Index ==="
echo ""

# Check all enforcer scripts exist
ENFORCER_DIR="$(dirname "$0")"
EXPECTED_COUNT=19
ACTUAL_COUNT=$(ls "$ENFORCER_DIR"/enforcer_*.sh 2>/dev/null | wc -l)
if [[ "$ACTUAL_COUNT" -ge "$EXPECTED_COUNT" ]]; then
    log_pass "All $ACTUAL_COUNT enforcer scripts exist"
else
    log_fail "Only $ACTUAL_COUNT/$EXPECTED_COUNT enforcer scripts found"
fi

# Check no enforcer is empty
EMPTY=0
for f in "$ENFORCER_DIR"/enforcer_*.sh; do
    if [[ ! -s "$f" ]]; then
        log_fail "Empty enforcer: $(basename "$f")"
        EMPTY=$((EMPTY + 1))
    fi
done
if [[ $EMPTY -eq 0 ]]; then
    log_pass "All enforcer scripts are non-empty"
fi


echo ""
echo "--- Rule 00 Summary: $PASS passed, $WARN warnings, $FAIL failed ---"
[[ $FAIL -eq 0 ]]
