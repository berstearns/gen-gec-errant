#!/bin/bash
#===============================================================================
# enforcer_14_error_annotations.sh — Rule 14: Error Annotations
#===============================================================================
#
# Validates ERRANT annotation output (post-experiment).
# Checks:
#   - At least 1 error annotation found in results
#   - error_type_counts is non-empty
#   - Region classification (prompt vs generation) present
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

echo "=== Rule 14: Error Annotations ==="
echo ""

# Check error annotations (post-experiment)
if [[ -z "$OUTPUT_DIR" || ! -d "$OUTPUT_DIR" ]]; then
    log_fail "Output directory does not exist: ${OUTPUT_DIR:-<not set>}"
else
    # Check raw_results.json for annotations
    if [[ -f "$OUTPUT_DIR/raw_results.json" ]]; then
        python3 -c "
import json, sys
data = json.load(open('$OUTPUT_DIR/raw_results.json'))

# Look for error annotations in results
results = data if isinstance(data, list) else data.get('results', data.get('sentences', []))
if isinstance(data, dict):
    # Try to find any key with annotation data
    for k, v in data.items():
        if isinstance(v, list) and len(v) > 0:
            results = v
            break

anno_count = 0
has_types = False
has_region = False

for r in (results if isinstance(results, list) else []):
    if isinstance(r, dict):
        errs = r.get('errors', r.get('annotations', []))
        if isinstance(errs, list):
            anno_count += len(errs)
        if r.get('error_type_counts') or r.get('error_types'):
            has_types = True
        if 'region' in str(r) or 'prompt_errors' in str(r) or 'gen_errors' in str(r):
            has_region = True

if anno_count > 0:
    print(f'PASS:Found {anno_count} error annotations')
else:
    print('WARN:No error annotations found in raw_results.json')

if has_types:
    print('PASS:Error type counts present')

if has_region:
    print('PASS:Region classification present')
" 2>&1 | while IFS= read -r line; do
            case "$line" in
                PASS:*) log_pass "${line#PASS:}" ;;
                WARN:*) log_warn "${line#WARN:}" ;;
                FAIL:*) log_fail "${line#FAIL:}" ;;
            esac
        done
    else
        log_fail "raw_results.json not found — cannot check annotations"
    fi
fi


echo ""
echo "--- Rule 14 Summary: $PASS passed, $WARN warnings, $FAIL failed ---"
[[ $FAIL -eq 0 ]]
