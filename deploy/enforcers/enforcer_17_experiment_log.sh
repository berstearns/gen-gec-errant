#!/bin/bash
#===============================================================================
# enforcer_17_experiment_log.sh — Rule 17: Experiment Log
#===============================================================================
#
# Validates pipeline logging (post-experiment).
# Checks:
#   - Pipeline produced stdout/stderr log
#   - Log contains stage completion markers
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

echo "=== Rule 17: Experiment Log ==="
echo ""

# Check experiment log (post-experiment)
if [[ -z "$OUTPUT_DIR" || ! -d "$OUTPUT_DIR" ]]; then
    log_fail "Output directory does not exist: ${OUTPUT_DIR:-<not set>}"
else
    # Check for pipeline log file
    LOG_FILE=$(ls "$OUTPUT_DIR"/pipeline*.log "$OUTPUT_DIR"/*.log 2>/dev/null | head -1)
    if [[ -n "$LOG_FILE" ]]; then
        log_pass "Pipeline log found: $(basename "$LOG_FILE")"

        # Check for stage completion markers
        for marker in "data_loader" "generation" "gec" "annotation" "analysis"; do
            if grep -qi "$marker" "$LOG_FILE" 2>/dev/null; then
                log_pass "Log contains stage marker: $marker"
            else
                log_warn "Log missing stage marker: $marker"
            fi
        done
    else
        log_warn "No pipeline log file found in $OUTPUT_DIR"
    fi
fi


echo ""
echo "--- Rule 17 Summary: $PASS passed, $WARN warnings, $FAIL failed ---"
[[ $FAIL -eq 0 ]]
