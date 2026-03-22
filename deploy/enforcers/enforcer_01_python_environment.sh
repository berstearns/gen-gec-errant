#!/bin/bash
#===============================================================================
# enforcer_01_python_environment.sh — Rule 01: Python Environment
#===============================================================================
#
# Validates Python runtime and required packages.
# Checks:
#   - Python >= 3.10
#   - torch, transformers, errant, spacy, scipy, pandas, matplotlib, pyyaml importable
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

echo "=== Rule 01: Python Environment ==="
echo ""

# Check Python version >= 3.10
PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
if [[ -n "$PY_VERSION" ]]; then
    MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
    MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
    if [[ "$MAJOR" -ge 3 && "$MINOR" -ge 10 ]]; then
        log_pass "Python $PY_VERSION >= 3.10"
    else
        log_fail "Python $PY_VERSION < 3.10"
    fi
else
    log_fail "python3 not found"
fi

# Check required packages
PACKAGES=(torch transformers errant spacy scipy pandas matplotlib yaml)
for pkg in "${PACKAGES[@]}"; do
    if python3 -c "import $pkg" 2>/dev/null; then
        log_pass "Package importable: $pkg"
    else
        log_fail "Package not importable: $pkg"
    fi
done


echo ""
echo "--- Rule 01 Summary: $PASS passed, $WARN warnings, $FAIL failed ---"
[[ $FAIL -eq 0 ]]
