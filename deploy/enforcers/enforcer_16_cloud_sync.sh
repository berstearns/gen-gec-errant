#!/bin/bash
#===============================================================================
# enforcer_16_cloud_sync.sh — Rule 16: Cloud Sync
#===============================================================================
#
# Validates rclone cloud sync readiness.
# Checks:
#   - rclone is installed
#   - Remote 'i:' is configured and reachable
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

echo "=== Rule 16: Cloud Sync ==="
echo ""

# Check rclone is installed
if command -v rclone >/dev/null 2>&1; then
    log_pass "rclone is installed"
else
    log_warn "rclone not installed — cloud sync unavailable"
fi

# Check remote 'i:' is configured
if rclone listremotes 2>/dev/null | grep -q "^i:"; then
    log_pass "rclone remote 'i:' configured"

    # Quick check that remote is reachable
    if rclone lsd "i:" --max-depth 0 >/dev/null 2>&1; then
        log_pass "rclone remote 'i:' is reachable"
    else
        log_warn "rclone remote 'i:' configured but not reachable"
    fi
else
    log_warn "rclone remote 'i:' not configured"
fi


echo ""
echo "--- Rule 16 Summary: $PASS passed, $WARN warnings, $FAIL failed ---"
[[ $FAIL -eq 0 ]]
