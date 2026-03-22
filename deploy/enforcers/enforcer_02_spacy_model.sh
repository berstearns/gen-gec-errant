#!/bin/bash
#===============================================================================
# enforcer_02_spacy_model.sh — Rule 02: spaCy Model
#===============================================================================
#
# Validates spaCy model required by ERRANT.
# Checks:
#   - en_core_web_sm is installed and loadable
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

echo "=== Rule 02: spaCy Model ==="
echo ""

# Check en_core_web_sm is installed
if python3 -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null; then
    log_pass "spaCy model en_core_web_sm loaded successfully"
else
    log_fail "spaCy model en_core_web_sm not installed (run: python -m spacy download en_core_web_sm)"
fi


echo ""
echo "--- Rule 02 Summary: $PASS passed, $WARN warnings, $FAIL failed ---"
[[ $FAIL -eq 0 ]]
