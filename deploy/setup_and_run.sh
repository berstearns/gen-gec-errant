#!/bin/bash
#===============================================================================
# setup_and_run.sh — Deploy gen-gec-errant pipeline to a new GPU server
#===============================================================================
# One-shot script: clones from GitHub, sets up environment, fetches data +
# models from GDrive, runs pipeline configs.
#
# Usage (piped via SSH):
#   ssh -p PORT root@HOST "bash -s" < deploy/setup_and_run.sh
#   ssh -p PORT root@HOST "bash -s -- --config celva" < deploy/setup_and_run.sh
#   ssh -p PORT root@HOST "bash -s -- --config all" < deploy/setup_and_run.sh
#
# Or run directly on the server:
#   bash setup_and_run.sh --config celva
#   bash setup_and_run.sh --config all
#   bash setup_and_run.sh --skip-setup --config efcamdat
#
# Configs: celva, efcamdat, kupa, all
#
# Prerequisites:
#   - GPU server with CUDA, Python 3.10+, pip, git
#   - rclone configured with remote 'i:' (for model weights + results sync)
#
# Target: 32GB GPU, >= 64GB system RAM
#===============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'

GITHUB_REPO="https://github.com/berstearns/gen-gec-errant.git"
REPO_DIR="/workspace/gen-gec-errant"
DATA_DIR="/workspace/data"
FT_MODELS_DIR="/workspace/ft-models"
RESULTS_DIR="/workspace/gec-results"
GDRIVE_DATA="i:/_p/artificial-learners/data/splits"
GDRIVE_MODELS="i:/_p/artificial-learners/models"
GDRIVE_RESULTS="i:/_p/artificial-learners/generation-gec-errant/outputs"

CONFIG_NAME=""
SKIP_SETUP=""

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG_NAME="$2"; shift 2 ;;
        --skip-setup) SKIP_SETUP=1; shift ;;
        *) shift ;;
    esac
done

echo -e "${CYAN}=== gen-gec-errant: Setup & Run ===${NC}"
echo "  Config:  ${CONFIG_NAME:-<none — setup only>}"
echo "  GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  RAM:     $(free -h | awk '/Mem:/{print $2}')"
echo "  Python:  $(python3 --version 2>/dev/null || echo 'N/A')"
echo ""

#===============================================================================
# Phase 1: Clone repo & install dependencies
#===============================================================================
if [[ -z "${SKIP_SETUP:-}" ]]; then
    echo -e "${CYAN}[1/4] Cloning repo & installing dependencies${NC}"

    # Clone or update from GitHub
    if [[ -d "$REPO_DIR/.git" ]]; then
        echo "  Updating existing repo..."
        git -C "$REPO_DIR" pull --ff-only 2>&1 | tail -2
        echo -e "  ${GREEN}✓${NC} Repo updated"
    else
        echo "  Cloning from $GITHUB_REPO..."
        rm -rf "$REPO_DIR"
        git clone "$GITHUB_REPO" "$REPO_DIR" 2>&1 | tail -2
        echo -e "  ${GREEN}✓${NC} Repo cloned"
    fi

    # Install Python deps
    pip install torch transformers errant spacy scipy pandas matplotlib pyyaml 2>&1 | tail -3
    python3 -m spacy download en_core_web_sm 2>&1 | tail -1

    # Install pipeline package in editable mode
    pip install -e "$REPO_DIR" 2>&1 | tail -2

    # Verify
    python3 -c "from gen_gec_errant.pipeline import run_pipeline; print('  Pipeline importable')"
    echo -e "${GREEN}  Dependencies OK${NC}"
    echo ""

    #===========================================================================
    # Phase 2: Fetch data from GDrive
    #===========================================================================
    echo -e "${CYAN}[2/4] Fetching data${NC}"
    mkdir -p "$DATA_DIR"
    for f in norm-CELVA-SP.csv norm-EFCAMDAT-test.csv norm-KUPA-KEYS.csv; do
        if [[ -f "$DATA_DIR/$f" ]]; then
            echo -e "  ${GREEN}✓${NC} $f (cached)"
        else
            rclone copy "$GDRIVE_DATA/$f" "$DATA_DIR/" 2>&1
            echo -e "  ${GREEN}✓${NC} $f (downloaded)"
        fi
    done
    echo ""

    #===========================================================================
    # Phase 3: Fetch fine-tuned model weights from GDrive
    #===========================================================================
    echo -e "${CYAN}[3/4] Fetching fine-tuned model weights${NC}"
    mkdir -p "$FT_MODELS_DIR"

    # GPT-2 fine-tuned
    declare -A GPT2_FT=(
        ["gpt2-small-all-data-resume"]="gpt2/gpt2-small-all-data-resume/final"
        ["gpt2-medium-all-data-resume"]="gpt2/gpt2-medium-all-data-resume/final"
        ["gpt2-large-all-data-20260222-092036"]="gpt2/gpt2-large-all-data-20260222-092036/final"
    )
    for name in "${!GPT2_FT[@]}"; do
        src="${GPT2_FT[$name]}"
        dest="$FT_MODELS_DIR/$src"
        if [[ -f "$dest/model.safetensors" ]]; then
            echo -e "  ${GREEN}✓${NC} $src (cached)"
        else
            mkdir -p "$dest"
            rclone copy "$GDRIVE_MODELS/$src" "$dest/" 2>&1
            echo -e "  ${GREEN}✓${NC} $src"
        fi
    done

    # Pythia fine-tuned
    for size in 70m 160m 410m 1b 1.4b; do
        src="pythia/pythia-${size}-all-data/final"
        dest="$FT_MODELS_DIR/$src"
        if [[ -f "$dest/model.safetensors" ]]; then
            echo -e "  ${GREEN}✓${NC} $src (cached)"
        else
            mkdir -p "$dest"
            rclone copy "$GDRIVE_MODELS/$src" "$dest/" 2>&1
            echo -e "  ${GREEN}✓${NC} $src"
        fi
    done

    # SmolLM2 fine-tuned
    for size in 135m 360m 1.7b; do
        src="smollm2/smollm2-${size}-all-data/final"
        dest="$FT_MODELS_DIR/$src"
        if [[ -f "$dest/model.safetensors" ]]; then
            echo -e "  ${GREEN}✓${NC} $src (cached)"
        else
            mkdir -p "$dest"
            rclone copy "$GDRIVE_MODELS/$src" "$dest/" 2>&1
            echo -e "  ${GREEN}✓${NC} $src"
        fi
    done
    echo ""
fi

#===============================================================================
# Phase 4: Run pipeline
#===============================================================================
if [[ -z "$CONFIG_NAME" ]]; then
    echo -e "${GREEN}Setup complete. Run with --config to start pipeline:${NC}"
    echo "  --config celva      CELVA-SP dataset (24 models)"
    echo "  --config efcamdat   EFCAMDAT-test dataset (24 models)"
    echo "  --config kupa       KUPA-KEYS dataset (24 models)"
    echo "  --config all        All 3 datasets sequentially"
    exit 0
fi

run_config() {
    local config_file="$1"
    local dataset_name="$2"

    echo -e "${CYAN}[4/4] Running pipeline: $dataset_name${NC}"
    echo "  Config: $config_file"
    echo "  Start:  $(date)"
    echo ""

    if python3 -m gen_gec_errant.pipeline --config "$config_file" 2>&1 | tee "/workspace/pipeline-${dataset_name}.log"; then
        echo ""
        echo -e "${GREEN}  ✓ $dataset_name completed${NC}"
    else
        echo ""
        echo -e "${RED}  ✗ $dataset_name FAILED${NC}"
    fi

    # Sync results to GDrive
    local output_dir
    output_dir=$(python3 -c "import yaml; print(yaml.safe_load(open('$config_file')).get('output_dir', ''))" 2>/dev/null)
    if [[ -n "$output_dir" && -d "$output_dir" ]]; then
        echo "  Syncing to GDrive..."
        rclone copy "$output_dir/" "${GDRIVE_RESULTS}/${dataset_name}/" 2>&1
        echo -e "  ${GREEN}✓ Synced to ${GDRIVE_RESULTS}/${dataset_name}/${NC}"
    fi
    echo ""
}

CONFIGS_DIR="$REPO_DIR/configs/pipeline"

case "$CONFIG_NAME" in
    celva)
        run_config "$CONFIGS_DIR/celva-all-models.yaml" "celva-sp"
        ;;
    efcamdat)
        run_config "$CONFIGS_DIR/efcamdat-test-all-models.yaml" "efcamdat-test"
        ;;
    kupa)
        run_config "$CONFIGS_DIR/kupa-keys-all-models.yaml" "kupa-keys"
        ;;
    all)
        run_config "$CONFIGS_DIR/celva-all-models.yaml" "celva-sp"
        run_config "$CONFIGS_DIR/efcamdat-test-all-models.yaml" "efcamdat-test"
        run_config "$CONFIGS_DIR/kupa-keys-all-models.yaml" "kupa-keys"
        ;;
    *)
        echo -e "${RED}Unknown config: $CONFIG_NAME (use: celva, efcamdat, kupa, all)${NC}"
        exit 1
        ;;
esac

echo -e "${GREEN}=== All done. $(date) ===${NC}"
