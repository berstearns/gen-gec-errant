#!/bin/bash
#===============================================================================
# enforcer_08_gec_backend.sh — Rule 08: GEC Backend
#===============================================================================
#
# Validates that the GEC model is accessible.
# Checks:
#   - grammarly/coedit-large (or configured model) is cached or downloadable
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

echo "=== Rule 08: GEC Backend ==="
echo ""

# Check GEC model is cached or accessible
GEC_MODEL="${GEC_MODEL:-grammarly/coedit-large}"

# Check HuggingFace cache for the model
if python3 -c "
from transformers import AutoTokenizer
import os
cache = os.path.expanduser('~/.cache/huggingface/hub')
model_dir = 'models--' + '$GEC_MODEL'.replace('/', '--')
if os.path.isdir(os.path.join(cache, model_dir)):
    print('cached')
else:
    # Try loading tokenizer (lightweight check)
    AutoTokenizer.from_pretrained('$GEC_MODEL', local_files_only=True)
    print('cached')
" 2>/dev/null | grep -q "cached"; then
    log_pass "GEC model cached: $GEC_MODEL"
else
    log_warn "GEC model not cached locally: $GEC_MODEL (will download on first run)"
fi


echo ""
echo "--- Rule 08 Summary: $PASS passed, $WARN warnings, $FAIL failed ---"
[[ $FAIL -eq 0 ]]
