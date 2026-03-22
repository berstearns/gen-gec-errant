#!/bin/bash
#===============================================================================
# enforcer_10_reproducibility.sh — Rule 10: Reproducibility
#===============================================================================
#
# Validates reproducibility requirements.
# Checks:
#   - Git hash available for the project
#   - seed is set in pipeline configs
#   - pyproject.toml exists
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

echo "=== Rule 10: Reproducibility ==="
echo ""

# Check git hash available
if git -C "$PROJECT_DIR" rev-parse HEAD >/dev/null 2>&1; then
    HASH=$(git -C "$PROJECT_DIR" rev-parse --short HEAD)
    log_pass "Git hash available: $HASH"
else
    log_warn "Not a git repository (reproducibility reduced)"
fi

# Check for uncommitted changes
if git -C "$PROJECT_DIR" diff --quiet 2>/dev/null; then
    log_pass "No uncommitted changes"
else
    log_warn "Uncommitted changes exist — results may not be reproducible"
fi

# Check seed is set in configs
if grep -rq "seed:" "$CONFIGS_DIR"/*.yaml "$CONFIGS_DIR"/per-model/*.yaml 2>/dev/null; then
    log_pass "Random seed configured in pipeline configs"
else
    log_warn "No explicit seed in pipeline configs"
fi

# Check pyproject.toml exists
if [[ -f "$PROJECT_DIR/pyproject.toml" ]]; then
    log_pass "pyproject.toml exists"
else
    log_fail "No pyproject.toml found"
fi


echo ""
echo "--- Rule 10 Summary: $PASS passed, $WARN warnings, $FAIL failed ---"
[[ $FAIL -eq 0 ]]
