#!/bin/bash
#===============================================================================
# enforcer_15_statistical_results.sh — Rule 15: Statistical Results
#===============================================================================
#
# Validates statistical analysis output (post-experiment).
# Checks:
#   - model_comparison.json exists if >1 model
#   - Contains pairwise tests
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

echo "=== Rule 15: Statistical Results ==="
echo ""

# Check statistical results (post-experiment)
if [[ -z "$OUTPUT_DIR" || ! -d "$OUTPUT_DIR" ]]; then
    log_fail "Output directory does not exist: ${OUTPUT_DIR:-<not set>}"
else
    # Check for model comparison results
    if [[ -f "$OUTPUT_DIR/model_comparison.json" ]]; then
        log_pass "model_comparison.json exists"

        # Check it contains pairwise tests
        python3 -c "
import json
d = json.load(open('$OUTPUT_DIR/model_comparison.json'))
if 'pairwise' in str(d).lower() or 'wilcoxon' in str(d).lower() or 'p_value' in str(d).lower():
    print('PASS:Contains statistical test results')
else:
    print('WARN:No pairwise test results found')
" 2>&1 | while IFS= read -r line; do
            case "$line" in
                PASS:*) log_pass "${line#PASS:}" ;;
                WARN:*) log_warn "${line#WARN:}" ;;
            esac
        done
    else
        # Check if there was only 1 model (comparison not applicable)
        MODEL_COUNT=$(python3 -c "
import json
d = json.load(open('$OUTPUT_DIR/raw_results.json'))
models = set()
results = d if isinstance(d, list) else d.get('results', [])
for r in (results if isinstance(results, list) else []):
    if isinstance(r, dict) and 'model' in r:
        models.add(r['model'])
print(len(models))
" 2>/dev/null)
        if [[ "${MODEL_COUNT:-0}" -le 1 ]]; then
            log_pass "Single model — model_comparison.json not required"
        else
            log_warn "model_comparison.json missing for multi-model run"
        fi
    fi
fi


echo ""
echo "--- Rule 15 Summary: $PASS passed, $WARN warnings, $FAIL failed ---"
[[ $FAIL -eq 0 ]]
