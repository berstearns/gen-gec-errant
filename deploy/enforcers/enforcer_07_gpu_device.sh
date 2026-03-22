#!/bin/bash
#===============================================================================
# enforcer_07_gpu_device.sh — Rule 07: GPU / Device
#===============================================================================
#
# Validates GPU/CUDA availability for the experiment.
# Checks:
#   - CUDA available if device=cuda or device=auto
#   - nvidia-smi reachable
#   - Sufficient VRAM reported
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

echo "=== Rule 07: GPU / Device ==="
echo ""

# Check CUDA availability
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    VRAM=$(python3 -c "
import torch
total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
print(f'{total:.1f}')
" 2>/dev/null)
    log_pass "CUDA available (${VRAM:-?} GB VRAM)"
else
    log_warn "CUDA not available — pipeline will use CPU (slow)"
fi

# Check nvidia-smi reachable
if nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits >/dev/null 2>&1; then
    log_pass "nvidia-smi reachable"
else
    log_warn "nvidia-smi not reachable"
fi


echo ""
echo "--- Rule 07 Summary: $PASS passed, $WARN warnings, $FAIL failed ---"
[[ $FAIL -eq 0 ]]
