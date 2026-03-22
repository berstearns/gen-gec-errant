#!/bin/bash
#===============================================================================
# enforcer_12_output_completeness.sh — Rule 12: Output Completeness
#===============================================================================
#
# Validates that pipeline outputs are complete (post-experiment).
# Checks:
#   - raw_results.json exists in output_dir
#   - Summary JSON file exists
#   - plots/ directory exists (unless skip_plots)
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

echo "=== Rule 12: Output Completeness ==="
echo ""

# Check output completeness (post-experiment)
if [[ -z "$OUTPUT_DIR" || ! -d "$OUTPUT_DIR" ]]; then
    log_fail "Output directory does not exist: ${OUTPUT_DIR:-<not set>}"
else
    # Check raw_results.json
    if [[ -f "$OUTPUT_DIR/raw_results.json" ]]; then
        log_pass "raw_results.json exists"
    else
        log_fail "raw_results.json missing in $OUTPUT_DIR"
    fi

    # Check summary JSON(s)
    SUMMARIES=$(ls "$OUTPUT_DIR"/*summary*.json 2>/dev/null | wc -l)
    if [[ $SUMMARIES -gt 0 ]]; then
        log_pass "$SUMMARIES summary JSON file(s) found"
    else
        log_fail "No summary JSON files in $OUTPUT_DIR"
    fi

    # Check plots directory
    if [[ -d "$OUTPUT_DIR/plots" ]]; then
        PLOT_COUNT=$(ls "$OUTPUT_DIR/plots/"*.png "$OUTPUT_DIR/plots/"*.pdf "$OUTPUT_DIR/plots/"*.svg 2>/dev/null | wc -l)
        log_pass "plots/ directory exists ($PLOT_COUNT plot files)"
    else
        log_warn "plots/ directory missing (may have used skip_plots=true)"
    fi
fi


echo ""
echo "--- Rule 12 Summary: $PASS passed, $WARN warnings, $FAIL failed ---"
[[ $FAIL -eq 0 ]]
