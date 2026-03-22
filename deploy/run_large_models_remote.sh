#!/bin/bash
#===============================================================================
# run_large_models_remote.sh — Run larger models on RTX 8000 (48GB)
#===============================================================================
# Complementary to run_all_models_remote.sh on V100.
# Focuses on larger models + ft models with bigger batch sizes.
# Syncs to same GDrive output structure.
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

echo -e "${CYAN}=== gen-gec-errant: Large Models Run (RTX 8000) ===${NC}"
echo "  Repo:      $REPO_DIR"
echo "  Data:      $DATA_FILE"
echo "  Results:   $RESULTS_DIR"
echo "  GDrive:    $GDRIVE_BASE"
echo "  Timestamp: $TIMESTAMP"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

#===============================================================================
# Phase 0: Fetch ft model weights
#===============================================================================
echo -e "${CYAN}[Phase 0] Fetching fine-tuned model weights${NC}"
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
        echo "  Downloading $model_name..."
        mkdir -p "$dest"
        rclone copy "$src" "$dest/" 2>&1 | tail -3
        echo -e "  ${GREEN}✓${NC} $model_name downloaded"
    fi
done
echo ""

#===============================================================================
# Helper
#===============================================================================
run_model() {
    local name="$1"
    local hf_model_id="$2"
    local model_family="$3"
    local batch_size="${4:-4}"
    local is_learner_tuned="${5:-false}"
    local output_dir="${RESULTS_DIR}/${name}"

    echo ""
    echo -e "${CYAN}================================================================${NC}"
    echo -e "[$(date '+%H:%M:%S')] ${CYAN}Running: $name${NC} (batch_size=$batch_size)"
    echo -e "${CYAN}================================================================${NC}"

    # Skip if already complete
    if [[ -f "$output_dir/raw_results.json" ]] && [[ -z "${FORCE:-}" ]]; then
        echo -e "  ${YELLOW}⚠ Skipping (output exists). Set FORCE=1 to re-run.${NC}"
        SKIPPED=$((SKIPPED + 1))
        sync_results "$name" "$output_dir"
        return 0
    fi

    mkdir -p "$output_dir"

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
  batch_size: 8
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

batch_size: ${batch_size}
device: auto
seed: 42
output_dir: ${output_dir}
skip_plots: false
include_learner_baseline: true
YAML

    if python3 -m gen_gec_errant.pipeline --config "$config_file" 2>&1 | tee "$output_dir/pipeline.log"; then
        PASSED=$((PASSED + 1))
        echo -e "  ${GREEN}✓ $name completed${NC}"
        sync_results "$name" "$output_dir"
    else
        FAILED=$((FAILED + 1))
        echo -e "  ${RED}✗ $name FAILED${NC}"
        sync_results "$name" "$output_dir"
    fi
    rm -f "$config_file"
}

sync_results() {
    local name="$1"
    local output_dir="$2"
    echo "  Syncing $name to GDrive..."
    rclone copy "$output_dir/" "${GDRIVE_BASE}/${name}/" 2>&1 || echo "  Sync warning"
    echo -e "  ${GREEN}✓ Synced${NC}"
}

#===============================================================================
# Run models — larger ones with bigger batch sizes (48GB VRAM)
#===============================================================================

# Large native models (batch_size=4-8, need more VRAM)
run_model "gpt2-xl"       "gpt2-xl"                       "gpt2"   4
run_model "gemma2-2b"     "google/gemma-2-2b"              "gemma"  4
run_model "llama3.2-1b"   "meta-llama/Llama-3.2-1B"       "llama"  4
run_model "rwkv6-1.6b"    "RWKV/v6-Finch-1B6-HF"         "rwkv"   4
run_model "qwen2.5-1.5b"  "Qwen/Qwen2.5-1.5B"            "qwen2"  4
run_model "smollm2-1.7b"  "HuggingFaceTB/SmolLM2-1.7B"   "smollm2" 4
run_model "olmo2-1b"      "allenai/OLMo-2-0425-1B"        "olmo"   4
run_model "pythia-1.4b"   "EleutherAI/pythia-1.4b"        "pythia" 4
run_model "pythia-1b"     "EleutherAI/pythia-1b"          "pythia" 4
run_model "tinyllama-1.1b" "TinyLlama/TinyLlama_v1.1"    "llama"  4

# Medium models
run_model "gpt2-large"    "gpt2-large"                    "gpt2"   8
run_model "gpt2-medium"   "gpt2-medium"                   "gpt2"   8
run_model "pythia-410m"   "EleutherAI/pythia-410m"        "pythia" 8
run_model "mamba-790m"    "state-spaces/mamba-790m-hf"    "mamba"  8

# Smaller models (batch_size=16)
run_model "gpt2-small"    "gpt2"                          "gpt2"   16
run_model "pythia-160m"   "EleutherAI/pythia-160m"        "pythia" 16
run_model "pythia-70m"    "EleutherAI/pythia-70m"         "pythia" 16
run_model "smollm2-360m"  "HuggingFaceTB/SmolLM2-360M"   "smollm2" 16
run_model "smollm2-135m"  "HuggingFaceTB/SmolLM2-135M"   "smollm2" 16
run_model "mamba-370m"    "state-spaces/mamba-370m-hf"    "mamba"  16
run_model "mamba-130m"    "state-spaces/mamba-130m-hf"    "mamba"  16
run_model "qwen2.5-0.5b"  "Qwen/Qwen2.5-0.5B"           "qwen2"  16

# Fine-tuned models
run_model "ft-gpt2-small"  "${MODELS_DIR}/ft-gpt2-small"  "gpt2" 8 "true"
run_model "ft-gpt2-medium" "${MODELS_DIR}/ft-gpt2-medium" "gpt2" 8 "true"
run_model "ft-gpt2-large"  "${MODELS_DIR}/ft-gpt2-large"  "gpt2" 4 "true"

#===============================================================================
# Learner baseline extraction
#===============================================================================
echo ""
echo -e "${CYAN}[Phase 3] Collecting learner-baseline${NC}"
FIRST_OUTPUT="${RESULTS_DIR}/gpt2-small"
if [[ -f "$FIRST_OUTPUT/raw_results.json" ]]; then
    mkdir -p "${RESULTS_DIR}/learner-baseline"
    python3 -c "
import json, shutil
raw = json.load(open('$FIRST_OUTPUT/raw_results.json'))
baseline = {k: v for k, v in raw.items() if 'learner' in k.lower() or 'baseline' in k.lower()}
if baseline:
    json.dump(baseline, open('${RESULTS_DIR}/learner-baseline/raw_results.json', 'w'), indent=2)
    print('Extracted learner baseline')
else:
    shutil.copy('$FIRST_OUTPUT/raw_results.json', '${RESULTS_DIR}/learner-baseline/raw_results.json')
    print('Copied full results')
" 2>&1
    sync_results "learner-baseline" "${RESULTS_DIR}/learner-baseline"
fi

#===============================================================================
echo ""
echo -e "${CYAN}=== SUMMARY ===${NC}"
echo "  Passed:  $PASSED"
echo "  Failed:  $FAILED"
echo "  Skipped: $SKIPPED"
echo "  End:     $(date)"

rclone copy "$RESULTS_DIR/" "${GDRIVE_BASE}/" 2>&1 || true
echo -e "${GREEN}Done.${NC}"
