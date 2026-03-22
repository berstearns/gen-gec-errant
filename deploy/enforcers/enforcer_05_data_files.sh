#!/bin/bash
#===============================================================================
# enforcer_05_data_files.sh — Rule 05: Data Files
#===============================================================================
#
# Validates that input data files exist.
# Checks:
#   - data_path referenced in each config points to an existing file
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

echo "=== Rule 05: Data Files ==="
echo ""

# Check data files referenced in configs exist
DATA_PATHS_CHECKED=0
DATA_PATHS_OK=0

for cfg in "$CONFIGS_DIR"/*.yaml "$CONFIGS_DIR"/per-model/*.yaml; do
    [[ -f "$cfg" ]] || continue
    DATA_PATH=$(python3 -c "
import yaml
d = yaml.safe_load(open('$cfg'))
dl = d.get('data_loader', {})
print(dl.get('data_path', ''))
" 2>/dev/null)
    [[ -z "$DATA_PATH" ]] && continue

    DATA_PATHS_CHECKED=$((DATA_PATHS_CHECKED + 1))

    # Resolve relative paths against project dir
    if [[ "$DATA_PATH" != /* ]]; then
        DATA_PATH="$PROJECT_DIR/$DATA_PATH"
    fi

    if [[ -f "$DATA_PATH" ]]; then
        DATA_PATHS_OK=$((DATA_PATHS_OK + 1))
    else
        log_warn "Data file missing: $DATA_PATH (from $(basename "$cfg"))"
    fi
done

if [[ $DATA_PATHS_CHECKED -eq 0 ]]; then
    log_warn "No data_path fields found in configs"
elif [[ $DATA_PATHS_OK -eq $DATA_PATHS_CHECKED ]]; then
    log_pass "All $DATA_PATHS_OK data files exist"
else
    log_warn "$DATA_PATHS_OK/$DATA_PATHS_CHECKED data files found"
fi


echo ""
echo "--- Rule 05 Summary: $PASS passed, $WARN warnings, $FAIL failed ---"
[[ $FAIL -eq 0 ]]
