#!/bin/bash
#===============================================================================
# run_all_models_remote.sh — Run gen-gec-errant for ALL models on VastAI V100
#===============================================================================
# Runs ON the remote via SSH pipe.
# Outputs: /workspace/gec-results/{model-name}/
# Syncs each model's results to i:/_p/artificial-learners/generation-gec-errant/outputs/{model-name}/
#
# Usage: ssh -p PORT root@HOST "bash -s" < deploy/run_all_models_remote.sh
#===============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'

REPO_DIR="/workspace/gec-pipeline-v2"
DATA_DIR="/workspace/data"
RESULTS_DIR="/workspace/gec-results"
MODELS_DIR="/workspace/ft-models"
GDRIVE_BASE="i:/_p/artificial-learners/generation-gec-errant/outputs"
DATA_FILE="${DATA_DIR}/norm-CELVA-SP.csv"

PASSED=0; FAILED=0; SKIPPED=0
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"

echo -e "${CYAN}=== gen-gec-errant: Full Model Run ===${NC}"
echo "  Repo:      $REPO_DIR"
echo "  Data:      $DATA_FILE"
echo "  Results:   $RESULTS_DIR"
echo "  GDrive:    $GDRIVE_BASE"
echo "  Timestamp: $TIMESTAMP"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

#===============================================================================
# Phase 0: Setup — copy ft model weights from GDrive
#===============================================================================
echo -e "${CYAN}[Phase 0] Fetching fine-tuned model weights from GDrive${NC}"
mkdir -p "$MODELS_DIR"

declare -A FT_MODELS
FT_MODELS[ft-gpt2-small]="i:/_p/artificial-learners/models/gpt2/gpt2-small-all-data-resume/final"
FT_MODELS[ft-gpt2-medium]="i:/_p/artificial-learners/models/gpt2/gpt2-medium-all-data-resume/final"
FT_MODELS[ft-gpt2-large]="i:/_p/artificial-learners/models/gpt2/gpt2-large-all-data-20260222-092036/final"

for model_name in "${!FT_MODELS[@]}"; do
    src="${FT_MODELS[$model_name]}"
    dest="${MODELS_DIR}/${model_name}"
    if [[ -f "$dest/model.safetensors" ]] || [[ -f "$dest/pytorch_model.bin" ]]; then
        echo -e "  ${GREEN}✓${NC} $model_name already cached"
    else
        echo "  Downloading $model_name from ${src}..."
        mkdir -p "$dest"
        rclone copy "$src" "$dest/" --progress 2>&1 | tail -3
        echo -e "  ${GREEN}✓${NC} $model_name downloaded"
    fi
done
echo ""

#===============================================================================
# Helper: run one model
#===============================================================================
run_model() {
    local name="$1"
    local hf_model_id="$2"
    local model_family="$3"
    local output_dir="${RESULTS_DIR}/${name}"
    local is_learner_tuned="${4:-false}"

    echo ""
    echo -e "${CYAN}================================================================${NC}"
    echo -e "[$(date '+%H:%M:%S')] ${CYAN}Running: $name${NC}"
    echo "  hf_model_id:     $hf_model_id"
    echo "  model_family:    $model_family"
    echo "  output_dir:      $output_dir"
    echo "  is_learner_tuned: $is_learner_tuned"
    echo -e "${CYAN}================================================================${NC}"

    # Skip if output already exists and has raw_results.json
    if [[ -f "$output_dir/raw_results.json" ]] && [[ -z "${FORCE:-}" ]]; then
        echo -e "  ${YELLOW}⚠ Skipping (output exists). Set FORCE=1 to re-run.${NC}"
        SKIPPED=$((SKIPPED + 1))
        # Still sync existing results
        sync_results "$name" "$output_dir"
        return 0
    fi

    mkdir -p "$output_dir"

    # Build inline YAML config
    local config_file="/tmp/config-${name}.yaml"
    cat > "$config_file" <<YAML
data_loader:
  data_path: ${DATA_FILE}
  text_column: text
  max_sentences: null
  min_words: 10
  max_words: 500
  prompt_ratio: 0.5
  min_prompt_words: 5

generation:
  max_new_tokens: 50
  min_new_tokens: 10
  temperature: 1.0
  top_k: 50
  top_p: 0.95
  do_sample: true
  repetition_penalty: 1.2

gec:
  method: dedicated
  model_id: grammarly/coedit-large
  batch_size: 4
  device: auto

annotation:
  lang: en

analysis:
  skip_plots: false
  top_n_error_types: 10

models:
  - name: ${name}
    hf_model_id: ${hf_model_id}
    model_family: ${model_family}
    is_learner_tuned: ${is_learner_tuned}

batch_size: 2
device: auto
seed: 42
output_dir: ${output_dir}
skip_plots: false
include_learner_baseline: true
YAML

    # Run pipeline
    if python3 -m gen_gec_errant.pipeline --config "$config_file" 2>&1 | tee "$output_dir/pipeline.log"; then
        PASSED=$((PASSED + 1))
        echo -e "  ${GREEN}✓ $name completed${NC}"
        sync_results "$name" "$output_dir"
    else
        FAILED=$((FAILED + 1))
        echo -e "  ${RED}✗ $name FAILED${NC}"
        # Sync partial results anyway
        sync_results "$name" "$output_dir"
    fi

    rm -f "$config_file"
}

sync_results() {
    local name="$1"
    local output_dir="$2"
    local dest="${GDRIVE_BASE}/${name}/"

    echo "  Syncing $name to GDrive..."
    if rclone copy "$output_dir/" "$dest" 2>&1; then
        echo -e "  ${GREEN}✓ Synced to $dest${NC}"
    else
        echo -e "  ${YELLOW}⚠ Sync failed for $name${NC}"
    fi
}

#===============================================================================
# Phase 1: Native (HuggingFace-hosted) models — smallest to largest
#===============================================================================
echo -e "${CYAN}[Phase 1] Native models (HuggingFace-hosted)${NC}"

# GPT-2 family
run_model "gpt2-small"    "gpt2"            "gpt2"
run_model "gpt2-medium"   "gpt2-medium"     "gpt2"
run_model "gpt2-large"    "gpt2-large"      "gpt2"
run_model "gpt2-xl"       "gpt2-xl"         "gpt2"

# Pythia family (smallest to largest)
run_model "pythia-70m"    "EleutherAI/pythia-70m"    "pythia"
run_model "pythia-160m"   "EleutherAI/pythia-160m"   "pythia"
run_model "pythia-410m"   "EleutherAI/pythia-410m"   "pythia"
run_model "pythia-1b"     "EleutherAI/pythia-1b"     "pythia"
run_model "pythia-1.4b"   "EleutherAI/pythia-1.4b"   "pythia"

# SmolLM2 family
run_model "smollm2-135m"  "HuggingFaceTB/SmolLM2-135M"  "smollm2"
run_model "smollm2-360m"  "HuggingFaceTB/SmolLM2-360M"  "smollm2"
run_model "smollm2-1.7b"  "HuggingFaceTB/SmolLM2-1.7B"  "smollm2"

# Mamba family (SSM)
run_model "mamba-130m"    "state-spaces/mamba-130m-hf"   "mamba"
run_model "mamba-370m"    "state-spaces/mamba-370m-hf"   "mamba"
run_model "mamba-790m"    "state-spaces/mamba-790m-hf"   "mamba"

# Other architectures
run_model "tinyllama-1.1b" "TinyLlama/TinyLlama_v1.1"    "llama"
run_model "qwen2.5-0.5b"  "Qwen/Qwen2.5-0.5B"           "qwen2"
run_model "qwen2.5-1.5b"  "Qwen/Qwen2.5-1.5B"           "qwen2"
run_model "olmo2-1b"       "allenai/OLMo-2-0425-1B"      "olmo"
run_model "llama3.2-1b"    "meta-llama/Llama-3.2-1B"     "llama"
run_model "gemma2-2b"      "google/gemma-2-2b"            "gemma"
run_model "rwkv6-1.6b"     "RWKV/v6-Finch-1B6-HF"       "rwkv"

#===============================================================================
# Phase 2: Fine-tuned (artificial learner) models
#===============================================================================
echo ""
echo -e "${CYAN}[Phase 2] Fine-tuned (artificial learner) models${NC}"

run_model "ft-gpt2-small"  "${MODELS_DIR}/ft-gpt2-small"  "gpt2" "true"
run_model "ft-gpt2-medium" "${MODELS_DIR}/ft-gpt2-medium" "gpt2" "true"
run_model "ft-gpt2-large"  "${MODELS_DIR}/ft-gpt2-large"  "gpt2" "true"

#===============================================================================
# Phase 3: Sync learner-baseline results
#===============================================================================
echo ""
echo -e "${CYAN}[Phase 3] Collecting learner-baseline results${NC}"
# The learner baseline is automatically included in each model's output
# when include_learner_baseline=true. Collect from the first model's output.
FIRST_OUTPUT="${RESULTS_DIR}/gpt2-small"
if [[ -f "$FIRST_OUTPUT/raw_results.json" ]]; then
    mkdir -p "${RESULTS_DIR}/learner-baseline"
    python3 -c "
import json, os, shutil
raw = json.load(open('$FIRST_OUTPUT/raw_results.json'))
# Extract learner baseline data if present
results = raw if isinstance(raw, dict) else {}
baseline = {}
for k, v in results.items():
    if 'learner' in k.lower() or 'baseline' in k.lower():
        baseline[k] = v
if baseline:
    with open('${RESULTS_DIR}/learner-baseline/raw_results.json', 'w') as f:
        json.dump(baseline, f, indent=2)
    print('Extracted learner baseline data')
else:
    # Copy the full results — baseline is mixed in
    shutil.copy('$FIRST_OUTPUT/raw_results.json', '${RESULTS_DIR}/learner-baseline/raw_results.json')
    print('Copied full results (baseline embedded)')
" 2>&1
    sync_results "learner-baseline" "${RESULTS_DIR}/learner-baseline"
fi

#===============================================================================
# Summary
#===============================================================================
echo ""
echo -e "${CYAN}================================================================${NC}"
echo -e "${CYAN}=== FINAL SUMMARY ===${NC}"
echo "  Passed:  $PASSED"
echo "  Failed:  $FAILED"
echo "  Skipped: $SKIPPED"
echo "  End:     $(date)"
echo -e "${CYAN}================================================================${NC}"

# Final full sync of everything
echo ""
echo "Final sync: all results to GDrive..."
rclone copy "$RESULTS_DIR/" "${GDRIVE_BASE}/" 2>&1 || echo "Final sync warning"
echo -e "${GREEN}All done.${NC}"
