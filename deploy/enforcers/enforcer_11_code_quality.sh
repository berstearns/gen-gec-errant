#!/bin/bash
#===============================================================================
# enforcer_11_code_quality.sh — Rule 11: Code Quality
#===============================================================================
#
# Validates minimum code quality.
# Checks:
#   - All .py files in src/ pass syntax check
#   - No import crashes on core modules
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

echo "=== Rule 11: Code Quality ==="
echo ""

# Check Python files have no syntax errors
SRC_DIR="$PROJECT_DIR/src"
SYNTAX_ERRORS=0
FILE_COUNT=0

if [[ -d "$SRC_DIR" ]]; then
    while IFS= read -r pyfile; do
        FILE_COUNT=$((FILE_COUNT + 1))
        if ! python3 -c "import py_compile; py_compile.compile('$pyfile', doraise=True)" 2>/dev/null; then
            log_fail "Syntax error: ${pyfile#$PROJECT_DIR/}"
            SYNTAX_ERRORS=$((SYNTAX_ERRORS + 1))
        fi
    done < <(find "$SRC_DIR" -name "*.py" 2>/dev/null)

    if [[ $SYNTAX_ERRORS -eq 0 && $FILE_COUNT -gt 0 ]]; then
        log_pass "All $FILE_COUNT Python files pass syntax check"
    fi
else
    log_fail "Source directory not found: $SRC_DIR"
fi

# Check core module imports
if python3 -c "from gen_gec_errant._config_utils import load_yaml_config" 2>/dev/null; then
    log_pass "Core config utils importable"
else
    log_warn "Cannot import config utils (may need pip install -e .)"
fi


echo ""
echo "--- Rule 11 Summary: $PASS passed, $WARN warnings, $FAIL failed ---"
[[ $FAIL -eq 0 ]]
