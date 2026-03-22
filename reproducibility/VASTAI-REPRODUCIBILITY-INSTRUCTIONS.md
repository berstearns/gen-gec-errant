# Vast.ai Reproducibility Execution Instructions

Instructions for an LLM agent to run all 12 reproducibility experiments on a
vast.ai GPU server and sync results to Google Drive via rclone.

## Golden Rule

**ALL changes to the server go through the orchestrator or explicit SSH
commands documented here. Never improvise installs or file moves on the
remote.**

## Overview

12 experiments, each running: **Generate text -> GEC correct -> ERRANT annotate -> Analysis**

Each experiment tests one model on 3 datasets (CELVA-SP, EFCAMDAT-test, KUPA-KEYS) = 36 pipeline runs total.

All experiments are inference-only (no training). Peak VRAM ~ 10 GB (coedit-large GEC model at batch=8). A single 16-24 GB GPU handles everything.

---

## Experiment Registry

| # | Experiment Dir | Model Name | Params | Source | rclone Path (GDrive `i:`) | HF ID (for native) |
|---|---------------|------------|--------|--------|---------------------------|---------------------|
| 1 | paper-reproducibility-gpt2-native-zero-shot | gpt2-small | 124M | HuggingFace | N/A | `gpt2` |
| 2 | paper-reproducibility-ft-gpt2-small | ft-gpt2-small | 124M | rclone | `/_p/artificial-learners/models/2026-02-23-model/gpt2-small-all-data-resume` | N/A |
| 3 | paper-reproducibility-ft-gpt2-medium | ft-gpt2-medium | 355M | rclone | `/_p/artificial-learners/models/gpt2/gpt2-medium-all-data-resume` | N/A |
| 4 | paper-reproducibility-ft-gpt2-large | ft-gpt2-large | 774M | rclone | `/_p/artificial-learners/models/gpt2/gpt2-large-all-data-20260222-092036` | N/A |
| 5 | paper-reproducibility-ft-pythia-70m | ft-pythia-70m | 70M | rclone | `/_p/artificial-learners/models/pythia/pythia-70m-all-data` | N/A |
| 6 | paper-reproducibility-ft-pythia-160m | ft-pythia-160m | 160M | rclone | `/_p/artificial-learners/models/pythia/pythia-160m-all-data` | N/A |
| 7 | paper-reproducibility-ft-pythia-410m | ft-pythia-410m | 410M | rclone | `/_p/artificial-learners/models/pythia/pythia-410m-all-data` | N/A |
| 8 | paper-reproducibility-ft-pythia-1b | ft-pythia-1b | 1B | rclone | `/_p/artificial-learners/models/pythia/pythia-1b-all-data` | N/A |
| 9 | paper-reproducibility-ft-pythia-1.4b | ft-pythia-1.4b | 1.4B | rclone | `/_p/artificial-learners/models/pythia/pythia-1.4b-all-data` | N/A |
| 10 | paper-reproducibility-ft-smollm2-135m | ft-smollm2-135m | 135M | rclone | `/_p/artificial-learners/models/smollm2/smollm2-135m-all-data` | N/A |
| 11 | paper-reproducibility-ft-smollm2-360m | ft-smollm2-360m | 360M | rclone | `/_p/artificial-learners/models/smollm2/smollm2-360m-all-data` | N/A |
| 12 | paper-reproducibility-ft-smollm2-1.7b | ft-smollm2-1.7b | 1.7B | rclone | `/_p/artificial-learners/models/smollm2/smollm2-1.7b-all-data` | N/A |

## Datasets (same for all experiments)

| Name | GDrive Path | Rows |
|------|------------|------|
| norm-CELVA-SP.csv | `i:/phd-experimental-data/cefr-classification/data/splits/norm-CELVA-SP.csv` | 1,742 |
| norm-EFCAMDAT-test.csv | `i:/phd-experimental-data/cefr-classification/data/splits/norm-EFCAMDAT-test.csv` | 20,000 |
| norm-KUPA-KEYS.csv | `i:/phd-experimental-data/cefr-classification/data/splits/norm-KUPA-KEYS.csv` | 1,006 |

---

## Remote Server Layout

```
/workspace/
  gec-pipeline/              # v2 project code (pip installed)
    src/gen_gec_errant/      # Python package
    pyproject.toml
    reproducibility/         # experiment dirs (for reference)
  data/                      # dataset CSVs
    norm-CELVA-SP.csv
    norm-EFCAMDAT-test.csv
    norm-KUPA-KEYS.csv
  models/                    # all model weights
    gpt2/                    # native HF gpt2 (downloaded via transformers)
    ft-gpt2-small/final/     # fine-tuned from rclone
    ft-gpt2-medium/final/
    ft-gpt2-large/final/
    ft-pythia-70m/final/
    ft-pythia-160m/final/
    ft-pythia-410m/final/
    ft-pythia-1b/final/
    ft-pythia-1.4b/final/
    ft-smollm2-135m/final/
    ft-smollm2-360m/final/
    ft-smollm2-1.7b/final/
  results/                   # experiment outputs (one subdir per experiment)
    gpt2-native-zero-shot/
      norm-CELVA-SP/
      norm-EFCAMDAT-test/
      norm-KUPA-KEYS/
      cross_dataset_summary.json
    ft-gpt2-small/
      ...
    ...
  configs/                   # generated YAML pipeline configs
```

---

## GPU Requirements

Inference-only. Models load one at a time (gen model, then GEC model sequentially).

- **Generation models**: 70M - 1.7B params, peak ~7 GB at batch=8
- **GEC model** (coedit-large, 770M): ~9 GB at batch=8
- **Total peak**: ~10 GB (models don't load simultaneously)
- **Recommended GPU**: 16 GB minimum (RTX 4060 Ti, RTX A4000, etc.)
- **Optimal GPU**: 24 GB (RTX 3090, RTX 4090) for batch=16

### Vast.ai Search

```bash
cd ~/p/all-my-tiny-projects/vastai
./vastai-rent --min-gpu-ram 16 --min-duration 168 --sort best-value
```

For budget option: `--min-gpu-ram 16 --max-price 0.30`
For speed: `--gpu-name "RTX 4090" --min-gpu-ram 24`

---

## Execution Plan

### Variables (set these first)

```bash
cd ~/p/all-my-tiny-projects/vastai

# After renting, get the SSH URL:
URL="ssh://root@<IP>:<PORT>"

# Orchestrator shorthand (use the GEC orchestrator for get-data, quick_start, etc.)
O="./gec-errant-analysis/orchestrator.sh --mode ssh --ssh-url $URL"
```

---

### Phase 1: Bootstrap Server (~20 min)

```bash
# 1a. System tools (rclone, pyenv, etc.)
$O quick_start

# 1b. Copy rclone config for GDrive access
$O copy-rclone
```

---

### Phase 2: Deploy Project Code (~5 min)

**IMPORTANT**: The v2 project uses `src/gen_gec_errant/` package layout, NOT
the flat `.py` layout of v1. Do NOT use `copy-code` or `project-setup-gec`
from the orchestrator (they expect v1).

```bash
# 2a. Tar-pipe the v2 project to /workspace/gec-pipeline
LOCAL_PROJECT="$HOME/p/research-sketches/automatic-generation-correction-errortagging-v2"

# Extract SSH args from URL
SSH_HOST=$(echo "$URL" | sed 's|ssh://||' | cut -d: -f1)
SSH_PORT=$(echo "$URL" | sed 's|ssh://||' | cut -d: -f2)

tar -C "$LOCAL_PROJECT" -cf - \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.egg-info' \
    --exclude='outputs' \
    --exclude='data' \
    --exclude='.git' \
    --exclude='models' \
    . | ssh -p "$SSH_PORT" "$SSH_HOST" \
    "mkdir -p /workspace/gec-pipeline && tar -C /workspace/gec-pipeline -xf -"
```

---

### Phase 3: Install Dependencies (~10 min)

Run this via SSH pipe:

```bash
ssh -p "$SSH_PORT" "$SSH_HOST" bash -s <<'SETUP_EOF'
set -e

# pyenv setup
export PYENV_ROOT="/root/.pyenv"
export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
eval "$(pyenv init -)" 2>/dev/null || true

# Ensure Python 3.10
if ! pyenv versions --bare | grep -q "^3.10"; then
    pyenv install 3.10.18
fi
pyenv global 3.10.18

# Install project + deps
cd /workspace/gec-pipeline
pip install --upgrade pip
pip install -e "."

# spaCy model
python3 -m spacy download en_core_web_sm

# Verify
python3 -c "
from gen_gec_errant.pipeline.config import PipelineConfig, load_config_from_yaml
from gen_gec_errant.pipeline.runner import run_pipeline
from gen_gec_errant.data_loader.runner import load_sentences
from gen_gec_errant.generation.runner import generate_continuations
from gen_gec_errant.gec.runner import load_gec_corrector
from gen_gec_errant.annotation.runner import ERRANTAnnotator
from gen_gec_errant.analysis.runner import compute_model_summary
print('All imports verified')
"
echo "SETUP COMPLETE"
SETUP_EOF
```

---

### Phase 4: Download Data (~2 min)

```bash
$O get-data rclone \
  "i:/phd-experimental-data/cefr-classification/data/splits" \
  /workspace/data --max-size 50M
```

Verify (read-only SSH):
```bash
ssh -p "$SSH_PORT" "$SSH_HOST" "ls -la /workspace/data/norm-*.csv"
```

Expected: 3 CSV files (norm-CELVA-SP.csv, norm-EFCAMDAT-test.csv, norm-KUPA-KEYS.csv).

---

### Phase 5: Download Models (~30-60 min)

#### 5a. Native HF model (gpt2 — auto-downloaded by pipeline, but pre-download is faster)

```bash
$O download-hf-model gpt2 /workspace/models/gpt2
```

#### 5b. Fine-tuned models from GDrive (only need `final/` checkpoint, skip optimizer)

```bash
# GPT-2 family
$O get-data rclone \
  "i:/_p/artificial-learners/models/2026-02-23-model/gpt2-small-all-data-resume/final" \
  /workspace/models/ft-gpt2-small/final --max-size 800M

$O get-data rclone \
  "i:/_p/artificial-learners/models/gpt2/gpt2-medium-all-data-resume/final" \
  /workspace/models/ft-gpt2-medium/final --max-size 2G

$O get-data rclone \
  "i:/_p/artificial-learners/models/gpt2/gpt2-large-all-data-20260222-092036/final" \
  /workspace/models/ft-gpt2-large/final --max-size 4G

# Pythia family
$O get-data rclone \
  "i:/_p/artificial-learners/models/pythia/pythia-70m-all-data/final" \
  /workspace/models/ft-pythia-70m/final --max-size 800M

$O get-data rclone \
  "i:/_p/artificial-learners/models/pythia/pythia-160m-all-data/final" \
  /workspace/models/ft-pythia-160m/final --max-size 1G

$O get-data rclone \
  "i:/_p/artificial-learners/models/pythia/pythia-410m-all-data/final" \
  /workspace/models/ft-pythia-410m/final --max-size 2G

$O get-data rclone \
  "i:/_p/artificial-learners/models/pythia/pythia-1b-all-data/final" \
  /workspace/models/ft-pythia-1b/final --max-size 5G

$O get-data rclone \
  "i:/_p/artificial-learners/models/pythia/pythia-1.4b-all-data/final" \
  /workspace/models/ft-pythia-1.4b/final --max-size 7G

# SmolLM2 family
$O get-data rclone \
  "i:/_p/artificial-learners/models/smollm2/smollm2-135m-all-data/final" \
  /workspace/models/ft-smollm2-135m/final --max-size 800M

$O get-data rclone \
  "i:/_p/artificial-learners/models/smollm2/smollm2-360m-all-data/final" \
  /workspace/models/ft-smollm2-360m/final --max-size 2G

$O get-data rclone \
  "i:/_p/artificial-learners/models/smollm2/smollm2-1.7b-all-data/final" \
  /workspace/models/ft-smollm2-1.7b/final --max-size 8G
```

Verify (read-only SSH):
```bash
ssh -p "$SSH_PORT" "$SSH_HOST" "ls /workspace/models/*/final/config.json 2>/dev/null; ls /workspace/models/gpt2/config.json 2>/dev/null"
```

Expected: 12 config.json files (one per model).

---

### Phase 6: Smoke Test (~5 min)

Run a tiny end-to-end test before committing to the full run.

```bash
ssh -p "$SSH_PORT" "$SSH_HOST" bash -s <<'SMOKE_EOF'
set -e
export PYENV_ROOT="/root/.pyenv"
export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
eval "$(pyenv init -)" 2>/dev/null || true

cd /workspace/gec-pipeline

# Create a minimal config for smoke test
cat > /workspace/configs/smoke-test.yaml <<'YAML'
data_loader:
  data_path: /workspace/data/norm-CELVA-SP.csv
  text_column: text
  max_sentences: 3
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
  - name: gpt2-small
    hf_model_id: /workspace/models/gpt2
    model_family: gpt2

batch_size: 4
device: auto
seed: 42
output_dir: /workspace/smoke-test
skip_plots: true
YAML

mkdir -p /workspace/configs
python3 -m gen_gec_errant.pipeline --config /workspace/configs/smoke-test.yaml

echo ""
echo "=== SMOKE TEST RESULT ==="
ls -la /workspace/smoke-test/
[ -f /workspace/smoke-test/raw_results.json ] && echo "SMOKE TEST PASSED" || echo "SMOKE TEST FAILED"
SMOKE_EOF
```

**STOP if smoke test fails. Debug before proceeding.**

---

### Phase 7: Generate Configs for All Experiments

For each experiment, generate a YAML config per dataset. This is done locally
and then copied to the remote, OR generated on the remote directly.

The following script generates all 36 configs (12 experiments x 3 datasets).
Run this ON THE REMOTE:

```bash
ssh -p "$SSH_PORT" "$SSH_HOST" bash -s <<'CONFIG_EOF'
set -e
mkdir -p /workspace/configs

DATASETS="norm-CELVA-SP norm-EFCAMDAT-test norm-KUPA-KEYS"

# Experiment definitions: MODEL_NAME|HF_MODEL_PATH|MODEL_FAMILY|IS_LEARNER_TUNED
EXPERIMENTS=(
  "gpt2-small|/workspace/models/gpt2|gpt2|false"
  "ft-gpt2-small|/workspace/models/ft-gpt2-small/final|gpt2|true"
  "ft-gpt2-medium|/workspace/models/ft-gpt2-medium/final|gpt2|true"
  "ft-gpt2-large|/workspace/models/ft-gpt2-large/final|gpt2|true"
  "ft-pythia-70m|/workspace/models/ft-pythia-70m/final|pythia|true"
  "ft-pythia-160m|/workspace/models/ft-pythia-160m/final|pythia|true"
  "ft-pythia-410m|/workspace/models/ft-pythia-410m/final|pythia|true"
  "ft-pythia-1b|/workspace/models/ft-pythia-1b/final|pythia|true"
  "ft-pythia-1.4b|/workspace/models/ft-pythia-1.4b/final|pythia|true"
  "ft-smollm2-135m|/workspace/models/ft-smollm2-135m/final|smollm2|true"
  "ft-smollm2-360m|/workspace/models/ft-smollm2-360m/final|smollm2|true"
  "ft-smollm2-1.7b|/workspace/models/ft-smollm2-1.7b/final|smollm2|true"
)

for exp in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r MODEL_NAME MODEL_PATH MODEL_FAMILY IS_LEARNER <<< "$exp"

  LEARNER_LINE=""
  if [ "$IS_LEARNER" = "true" ]; then
    LEARNER_LINE="    is_learner_tuned: true"
  fi

  for dataset in $DATASETS; do
    CONFIG_FILE="/workspace/configs/${MODEL_NAME}_${dataset}.yaml"
    OUTPUT_DIR="/workspace/results/${MODEL_NAME}/${dataset}"

    cat > "$CONFIG_FILE" <<YAML
data_loader:
  data_path: /workspace/data/${dataset}.csv
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
  - name: ${MODEL_NAME}
    hf_model_id: ${MODEL_PATH}
    model_family: ${MODEL_FAMILY}
${LEARNER_LINE}

batch_size: 8
device: auto
seed: 42
output_dir: ${OUTPUT_DIR}
skip_plots: false
YAML

    echo "  Created: $CONFIG_FILE -> $OUTPUT_DIR"
  done
done

echo ""
echo "Total configs: $(ls /workspace/configs/*.yaml | wc -l)"
CONFIG_EOF
```

---

### Phase 8: Run All Experiments (detached)

Run the experiments as a single detached batch job. This processes all 36
config files sequentially and generates cross-dataset summaries.

```bash
ssh -p "$SSH_PORT" "$SSH_HOST" bash -s <<'RUN_SCRIPT' > /tmp/run-all-experiments.sh
cat <<'RUNEOF'
#!/bin/bash
set -e

export PYENV_ROOT="/root/.pyenv"
export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
eval "$(pyenv init -)" 2>/dev/null || true

cd /workspace/gec-pipeline

DATASETS="norm-CELVA-SP norm-EFCAMDAT-test norm-KUPA-KEYS"
EXPERIMENTS=(
  "gpt2-small"
  "ft-gpt2-small"
  "ft-gpt2-medium"
  "ft-gpt2-large"
  "ft-pythia-70m"
  "ft-pythia-160m"
  "ft-pythia-410m"
  "ft-pythia-1b"
  "ft-pythia-1.4b"
  "ft-smollm2-135m"
  "ft-smollm2-360m"
  "ft-smollm2-1.7b"
)

TOTAL=${#EXPERIMENTS[@]}
CURRENT=0
START_TIME=$(date +%s)

for exp in "${EXPERIMENTS[@]}"; do
  CURRENT=$((CURRENT + 1))
  echo ""
  echo "###################################################################"
  echo "# EXPERIMENT $CURRENT/$TOTAL: $exp"
  echo "# Started: $(date)"
  echo "###################################################################"

  for dataset in $DATASETS; do
    CONFIG="/workspace/configs/${exp}_${dataset}.yaml"
    OUTPUT="/workspace/results/${exp}/${dataset}"

    # Skip if already completed
    if [ -f "${OUTPUT}/raw_results.json" ]; then
      echo "[SKIP] Already completed: ${exp} / ${dataset}"
      continue
    fi

    echo ""
    echo "[RUN] ${exp} / ${dataset}"
    echo "  Config: $CONFIG"
    echo "  Output: $OUTPUT"
    echo "  Time: $(date '+%H:%M:%S')"

    python3 -m gen_gec_errant.pipeline --config "$CONFIG"

    echo "[DONE] ${exp} / ${dataset} at $(date '+%H:%M:%S')"
  done

  # Cross-dataset summary for this experiment
  echo ""
  echo "[SUMMARY] Generating cross-dataset summary for $exp..."
  python3 -c "
import json
from pathlib import Path

exp_name = '$exp'
datasets = '$DATASETS'.split()
results_base = Path('/workspace/results') / exp_name
summary = {}

for ds in datasets:
    ds_dir = results_base / ds
    ds_summary = {'dataset': ds}

    model_file = ds_dir / f'{exp_name}_summary.json'
    baseline_file = ds_dir / 'learner_baseline_summary.json'

    if model_file.exists():
        data = json.loads(model_file.read_text())
        ds_summary['model'] = {
            'ppl_mean': data.get('ppl_mean'),
            'ppl_median': data.get('ppl_median'),
            'total_errors': data.get('total_errors'),
            'avg_errors_per_sentence': data.get('avg_errors_per_sentence'),
            'error_rate': data.get('error_rate'),
            'top_error_types': data.get('top_10_error_types', [])[:5],
        }

    if baseline_file.exists():
        data = json.loads(baseline_file.read_text())
        ds_summary['learner_baseline'] = {
            'total_errors': data.get('total_errors'),
            'avg_errors_per_sentence': data.get('avg_errors_per_sentence'),
            'error_rate': data.get('error_rate'),
            'top_error_types': data.get('top_10_error_types', [])[:5],
        }

    summary[ds] = ds_summary

out = results_base / 'cross_dataset_summary.json'
out.write_text(json.dumps(summary, indent=2, default=str))
print(f'  Saved: {out}')
"
done

ELAPSED=$(( $(date +%s) - START_TIME ))
echo ""
echo "###################################################################"
echo "# ALL EXPERIMENTS COMPLETE"
echo "# Total time: $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m"
echo "# Results: /workspace/results/"
echo "###################################################################"
ls -d /workspace/results/*/
RUNEOF
RUN_SCRIPT

# Copy the script to remote and run detached
scp -P "$SSH_PORT" /tmp/run-all-experiments.sh "${SSH_HOST}:/workspace/run-all-experiments.sh"
ssh -p "$SSH_PORT" "$SSH_HOST" \
  "chmod +x /workspace/run-all-experiments.sh && \
   nohup /workspace/run-all-experiments.sh > /root/detached-repro-$(date +%Y%m%d-%H%M%S).log 2>&1 & \
   echo \"PID: \$!\" && echo \"Log: /root/detached-repro-*.log\""
```

**Alternative: simpler detached approach using the orchestrator pattern**

If you prefer not to use scp, pipe the script directly:

```bash
ssh -p "$SSH_PORT" "$SSH_HOST" bash <<'DETACH_EOF'
nohup bash -c '
export PYENV_ROOT="/root/.pyenv"
export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
eval "$(pyenv init -)" 2>/dev/null || true
cd /workspace/gec-pipeline

EXPERIMENTS="gpt2-small ft-gpt2-small ft-gpt2-medium ft-gpt2-large ft-pythia-70m ft-pythia-160m ft-pythia-410m ft-pythia-1b ft-pythia-1.4b ft-smollm2-135m ft-smollm2-360m ft-smollm2-1.7b"
DATASETS="norm-CELVA-SP norm-EFCAMDAT-test norm-KUPA-KEYS"

for exp in $EXPERIMENTS; do
  for dataset in $DATASETS; do
    CONFIG="/workspace/configs/${exp}_${dataset}.yaml"
    OUTPUT="/workspace/results/${exp}/${dataset}"
    [ -f "${OUTPUT}/raw_results.json" ] && echo "[SKIP] ${exp}/${dataset}" && continue
    echo "[RUN] ${exp}/${dataset} at $(date)"
    python3 -m gen_gec_errant.pipeline --config "$CONFIG"
    echo "[DONE] ${exp}/${dataset} at $(date)"
  done
done
echo "ALL COMPLETE at $(date)"
' > /root/detached-repro-all.log 2>&1 &
echo "PID: $!"
echo "Log: /root/detached-repro-all.log"
DETACH_EOF
```

---

### Phase 9: Start Result Sync (detached, looping)

Launch alongside the pipeline so results sync incrementally.

```bash
./gec-errant-analysis/orchestrator.sh --detach --mode ssh --ssh-url "$URL" \
  sync-results \
  --source /workspace/results \
  --dest "i:/_p/artificial-learners/generation-gec-errant/$(date +%Y-%m-%d)-reproducibility-12models"
```

This loops every 120 seconds. Runs until killed.

---

### Phase 10: Monitor Progress

```bash
# Tail the experiment log
ssh -p "$SSH_PORT" "$SSH_HOST" "tail -30 /root/detached-repro-all.log"

# Check GPU usage
ssh -p "$SSH_PORT" "$SSH_HOST" "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader"

# Count completed experiments
ssh -p "$SSH_PORT" "$SSH_HOST" "find /workspace/results -name raw_results.json | wc -l"
# Expected: 36 when all done

# List completed
ssh -p "$SSH_PORT" "$SSH_HOST" "find /workspace/results -name raw_results.json -exec dirname {} \; | sort"

# Check sync log
ssh -p "$SSH_PORT" "$SSH_HOST" "tail -10 /root/detached-sync-*.log 2>/dev/null || echo 'No sync log yet'"

# Follow experiment log live
ssh -p "$SSH_PORT" "$SSH_HOST" "tail -f /root/detached-repro-all.log"
```

---

### Phase 11: Verify Completion & Final Sync

After all 36 runs complete:

```bash
# Verify all outputs exist
ssh -p "$SSH_PORT" "$SSH_HOST" bash -s <<'VERIFY_EOF'
EXPERIMENTS="gpt2-small ft-gpt2-small ft-gpt2-medium ft-gpt2-large ft-pythia-70m ft-pythia-160m ft-pythia-410m ft-pythia-1b ft-pythia-1.4b ft-smollm2-135m ft-smollm2-360m ft-smollm2-1.7b"
DATASETS="norm-CELVA-SP norm-EFCAMDAT-test norm-KUPA-KEYS"
PASS=0
FAIL=0

for exp in $EXPERIMENTS; do
  for dataset in $DATASETS; do
    OUTPUT="/workspace/results/${exp}/${dataset}"
    if [ -f "${OUTPUT}/raw_results.json" ]; then
      PASS=$((PASS + 1))
    else
      echo "MISSING: ${exp}/${dataset}"
      FAIL=$((FAIL + 1))
    fi
  done

  # Check cross-dataset summary
  if [ -f "/workspace/results/${exp}/cross_dataset_summary.json" ]; then
    echo "OK: ${exp}/cross_dataset_summary.json"
  else
    echo "MISSING: ${exp}/cross_dataset_summary.json"
  fi
done

echo ""
echo "Pipeline runs: $PASS passed, $FAIL failed (expected: 36 passed)"
VERIFY_EOF

# Force a final sync
ssh -p "$SSH_PORT" "$SSH_HOST" bash -s <<'SYNC_EOF'
export PATH="/root/.pyenv/bin:/root/.pyenv/shims:$PATH"
rclone copy /workspace/results \
  "i:/_p/artificial-learners/generation-gec-errant/$(date +%Y-%m-%d)-reproducibility-12models" \
  --progress
echo "Final sync complete"
SYNC_EOF
```

---

## GDrive Result Structure

After sync, results will be at:

```
i:/_p/artificial-learners/generation-gec-errant/YYYY-MM-DD-reproducibility-12models/
  gpt2-small/
    norm-CELVA-SP/
      raw_results.json
      gpt2-small_summary.json
      learner_baseline_summary.json
      model_comparison.json
      full_results.tsv
      errors_long_format.tsv
      plots/
        perplexity_comparison.png
        error_comparison.png
        error_type_breakdown.png
        ppl_vs_errors_scatter.png
    norm-EFCAMDAT-test/
      ...
    norm-KUPA-KEYS/
      ...
    cross_dataset_summary.json
  ft-gpt2-small/
    ...
  ft-gpt2-medium/
    ...
  (... 12 experiment directories total)
```

---

## Timing Estimates (24 GB GPU)

| Phase | Duration |
|-------|----------|
| Bootstrap + deps | ~30 min |
| Model downloads (all 12) | ~30-60 min |
| Smoke test | ~5 min |
| Small models (gpt2-small, pythia-70m, 160m, smollm2-135m, 360m) x 3 datasets | ~2-4 hours |
| Medium models (gpt2-medium, pythia-410m) x 3 datasets | ~2-3 hours |
| Large models (gpt2-large, pythia-1b, 1.4b, smollm2-1.7b) x 3 datasets | ~4-8 hours |
| **Total** | **~10-18 hours** |

Most time is spent on GEC correction (coedit-large processes each sentence).
EFCAMDAT-test (20k sentences) dominates: ~70% of total time.

---

## Troubleshooting

### OOM on large model
Reduce batch_size in the config YAML from 8 to 4 or 2:
```bash
ssh -p "$SSH_PORT" "$SSH_HOST" \
  "sed -i 's/batch_size: 8/batch_size: 4/' /workspace/configs/ft-smollm2-1.7b_*.yaml"
```

### Pipeline crash mid-run
The run script skips completed experiments (checks for `raw_results.json`).
Just re-run the detached script — it resumes from where it stopped.

### Model download incomplete
Check for `config.json` in the model dir. If missing, re-download:
```bash
$O get-data rclone "i:/_p/artificial-learners/models/..." /workspace/models/... --max-size ...
```

### Sync not running
Check if sync process is alive:
```bash
ssh -p "$SSH_PORT" "$SSH_HOST" "ps aux | grep rclone"
```
If dead, relaunch Phase 9.

### Wrong Python version
Verify pyenv is active:
```bash
ssh -p "$SSH_PORT" "$SSH_HOST" 'export PYENV_ROOT="/root/.pyenv"; export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"; eval "$(pyenv init -)"; python3 --version'
```

---

## Checklist

- [ ] GPU rented (note instance ID, GPU model, SSH URL)
- [ ] `quick_start` completed
- [ ] `copy-rclone` completed
- [ ] v2 project code deployed to `/workspace/gec-pipeline`
- [ ] Dependencies installed (`pip install -e .`)
- [ ] All imports verified
- [ ] Data files present (3 CSVs in `/workspace/data/`)
- [ ] All 12 models downloaded (check `config.json` in each)
- [ ] Smoke test passed
- [ ] All 36 configs generated in `/workspace/configs/`
- [ ] Pipeline launched (detached)
- [ ] Sync launched (detached)
- [ ] GPU usage verified via `nvidia-smi`
- [ ] Progress monitored (tail logs, count raw_results.json)
- [ ] All 36 raw_results.json present
- [ ] All 12 cross_dataset_summary.json present
- [ ] Final sync to GDrive complete
- [ ] Instance destroyed after verification
