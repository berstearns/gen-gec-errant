#!/bin/bash
#===============================================================================
# enforcer_13_results_validity.sh — Rule 13: Results Validity
#===============================================================================
#
# Validates output file contents (post-experiment).
# Checks:
#   - JSON files parse correctly
#   - Summary contains required keys (mean_ppl, error_rate, etc.)
#   - TSV/CSV files have expected columns
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

echo "=== Rule 13: Results Validity ==="
echo ""

# Check results validity (post-experiment)
if [[ -z "$OUTPUT_DIR" || ! -d "$OUTPUT_DIR" ]]; then
    log_fail "Output directory does not exist: ${OUTPUT_DIR:-<not set>}"
else
    # Validate JSON files parse
    JSON_COUNT=0
    JSON_OK=0
    for jf in "$OUTPUT_DIR"/*.json; do
        [[ -f "$jf" ]] || continue
        JSON_COUNT=$((JSON_COUNT + 1))
        if python3 -c "import json; json.load(open('$jf'))" 2>/dev/null; then
            JSON_OK=$((JSON_OK + 1))
        else
            log_fail "Invalid JSON: $(basename "$jf")"
        fi
    done
    if [[ $JSON_COUNT -gt 0 && $JSON_OK -eq $JSON_COUNT ]]; then
        log_pass "All $JSON_COUNT JSON files parse correctly"
    fi

    # Check summary has required keys
    SUMMARY=$(ls "$OUTPUT_DIR"/*summary*.json 2>/dev/null | head -1)
    if [[ -n "$SUMMARY" && -f "$SUMMARY" ]]; then
        python3 -c "
import json, sys
d = json.load(open('$SUMMARY'))
# Accept either flat dict or nested per-model
if isinstance(d, dict):
    # If it has model sub-keys, check first model
    vals = d
    for k, v in d.items():
        if isinstance(v, dict):
            vals = v
            break
    required = ['mean_ppl', 'error_rate']
    found = [k for k in required if k in vals or any(k in str(kk) for kk in vals)]
    missing = [k for k in required if k not in vals and not any(k in str(kk) for kk in vals)]
    if found:
        print('PASS:Summary has keys: ' + ', '.join(found))
    if missing:
        print('WARN:Summary missing keys: ' + ', '.join(missing))
" 2>&1 | while IFS= read -r line; do
            case "$line" in
                PASS:*) log_pass "${line#PASS:}" ;;
                WARN:*) log_warn "${line#WARN:}" ;;
                FAIL:*) log_fail "${line#FAIL:}" ;;
            esac
        done
    fi

    # Check CSV/TSV files have columns
    for tf in "$OUTPUT_DIR"/*.tsv "$OUTPUT_DIR"/*.csv; do
        [[ -f "$tf" ]] || continue
        COLS=$(head -1 "$tf" | tr '\t' '\n' | wc -l)
        if [[ $COLS -gt 1 ]]; then
            log_pass "$(basename "$tf") has $COLS columns"
        else
            log_warn "$(basename "$tf") has only $COLS column"
        fi
    done
fi


echo ""
echo "--- Rule 13 Summary: $PASS passed, $WARN warnings, $FAIL failed ---"
[[ $FAIL -eq 0 ]]
