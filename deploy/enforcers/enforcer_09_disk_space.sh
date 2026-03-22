#!/bin/bash
#===============================================================================
# enforcer_09_disk_space.sh — Rule 09: Disk Space
#===============================================================================
#
# Validates sufficient disk space for outputs.
# Checks:
#   - >= 2GB free on output directory partition
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

echo "=== Rule 09: Disk Space ==="
echo ""

# Check disk space >= 2GB free on output partition
OUTPUT_PART="${OUTPUT_DIR:-$PROJECT_DIR/outputs}"
# Create dir if needed to resolve mount point
mkdir -p "$OUTPUT_PART" 2>/dev/null
FREE_KB=$(df -k "$OUTPUT_PART" 2>/dev/null | tail -1 | awk '{print $4}')

if [[ -n "$FREE_KB" ]]; then
    FREE_GB=$((FREE_KB / 1024 / 1024))
    if [[ $FREE_KB -ge 2097152 ]]; then
        log_pass "Disk space: ${FREE_GB}GB free (>= 2GB required)"
    else
        log_fail "Disk space: ${FREE_GB}GB free (< 2GB required)"
    fi
else
    log_warn "Cannot determine free disk space for $OUTPUT_PART"
fi


echo ""
echo "--- Rule 09 Summary: $PASS passed, $WARN warnings, $FAIL failed ---"
[[ $FAIL -eq 0 ]]
