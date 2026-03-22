#!/bin/bash
#===============================================================================
# enforcer_06_model_weights.sh — Rule 06: Model Weights
#===============================================================================
#
# Validates that model checkpoints are available (pre-experiment).
# Checks:
#   - Local checkpoint directory exists if hf_model_id is a local path
#   - HuggingFace model_id is well-formed if remote
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

echo "=== Rule 06: Model Weights ==="
echo ""

# Check model weights availability (pre-experiment)
# If MODEL_NAME is set, check its specific config; otherwise check all
if [[ -n "$MODEL_NAME" ]]; then
    CFG=$(ls "$CONFIGS_DIR"/per-model/"$MODEL_NAME".yaml "$CONFIGS_DIR"/per-model/"${MODEL_NAME}"*.yaml 2>/dev/null | head -1)
    CFGS=("${CFG:-}")
else
    CFGS=("$CONFIGS_DIR"/per-model/*.yaml)
fi

for cfg in "${CFGS[@]}"; do
    [[ -f "$cfg" ]] || continue
    HF_ID=$(python3 -c "
import yaml
d = yaml.safe_load(open('$cfg'))
for m in d.get('models', []):
    print(m.get('hf_model_id', ''))
    break
" 2>/dev/null)
    [[ -z "$HF_ID" ]] && continue

    NAME=$(basename "$cfg" .yaml)
    if [[ "$HF_ID" == /* ]]; then
        # Local path — check directory exists with model files
        if [[ -d "$HF_ID" ]] && ls "$HF_ID"/*.safetensors "$HF_ID"/model*.bin "$HF_ID"/pytorch_model.bin &>/dev/null; then
            log_pass "Local weights found: $NAME ($HF_ID)"
        elif [[ -d "$HF_ID" ]]; then
            log_warn "Directory exists but no model files: $NAME ($HF_ID)"
        else
            log_warn "Local weight dir missing: $NAME ($HF_ID)"
        fi
    else
        # HuggingFace model ID — check it looks valid
        if [[ "$HF_ID" == */* ]] || [[ "$HF_ID" == gpt2* ]]; then
            log_pass "HF model ID well-formed: $NAME ($HF_ID)"
        else
            log_warn "Unusual model ID: $NAME ($HF_ID)"
        fi
    fi
done


echo ""
echo "--- Rule 06 Summary: $PASS passed, $WARN warnings, $FAIL failed ---"
[[ $FAIL -eq 0 ]]
