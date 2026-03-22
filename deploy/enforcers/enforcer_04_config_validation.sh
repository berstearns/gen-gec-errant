#!/bin/bash
#===============================================================================
# enforcer_04_config_validation.sh — Rule 04: Config Validation
#===============================================================================
#
# Validates YAML pipeline configs.
# Checks:
#   - All YAML configs in configs/pipeline/ parse correctly
#   - Required top-level fields present (data_loader, generation, gec, annotation, models)
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

echo "=== Rule 04: Config Validation ==="
echo ""

# Check all YAML configs parse correctly
CONFIGS_FOUND=0
CONFIGS_VALID=0
REQUIRED_FIELDS="data_loader generation gec annotation models"

for cfg in "$CONFIGS_DIR"/*.yaml "$CONFIGS_DIR"/per-model/*.yaml; do
    [[ -f "$cfg" ]] || continue
    CONFIGS_FOUND=$((CONFIGS_FOUND + 1))
    if python3 -c "import yaml; yaml.safe_load(open('$cfg'))" 2>/dev/null; then
        CONFIGS_VALID=$((CONFIGS_VALID + 1))
    else
        log_fail "Invalid YAML: $(basename "$cfg")"
    fi
done

if [[ $CONFIGS_FOUND -eq 0 ]]; then
    log_fail "No YAML configs found in $CONFIGS_DIR"
elif [[ $CONFIGS_FOUND -eq $CONFIGS_VALID ]]; then
    log_pass "All $CONFIGS_FOUND configs parse correctly"
fi

# Check required fields in a sample config
SAMPLE_CFG=$(ls "$CONFIGS_DIR"/*.yaml 2>/dev/null | head -1)
if [[ -n "$SAMPLE_CFG" && -f "$SAMPLE_CFG" ]]; then
    for field in $REQUIRED_FIELDS; do
        if python3 -c "import yaml; d=yaml.safe_load(open('$SAMPLE_CFG')); assert '$field' in d" 2>/dev/null; then
            log_pass "Required field present: $field (in $(basename "$SAMPLE_CFG"))"
        else
            log_warn "Missing field: $field (in $(basename "$SAMPLE_CFG"))"
        fi
    done
fi


echo ""
echo "--- Rule 04 Summary: $PASS passed, $WARN warnings, $FAIL failed ---"
[[ $FAIL -eq 0 ]]
