#!/usr/bin/env python3
"""Generate all enforcer scripts from rule definitions for gen-gec-errant."""
import os, stat, textwrap

ENFORCERS_DIR = os.path.dirname(os.path.abspath(__file__))

# (rule_id, short_name, title, description_of_what_to_validate)
RULES = [
    ("00", "index", "Enforcer Index",
     """Validates that the enforcer framework itself is intact.
Checks:
  - All enforcer scripts exist (one per rule)
  - No enforcer is empty or zero-length"""),

    ("01", "python_environment", "Python Environment",
     """Validates Python runtime and required packages.
Checks:
  - Python >= 3.10
  - torch, transformers, errant, spacy, scipy, pandas, matplotlib, pyyaml importable"""),

    ("02", "spacy_model", "spaCy Model",
     """Validates spaCy model required by ERRANT.
Checks:
  - en_core_web_sm is installed and loadable"""),

    ("03", "pipeline_importable", "Pipeline Importable",
     """Validates that the gen_gec_errant pipeline module is importable.
Checks:
  - python -c 'from gen_gec_errant.pipeline import run_pipeline' succeeds"""),

    ("04", "config_validation", "Config Validation",
     """Validates YAML pipeline configs.
Checks:
  - All YAML configs in configs/pipeline/ parse correctly
  - Required top-level fields present (data_loader, generation, gec, annotation, models)"""),

    ("05", "data_files", "Data Files",
     """Validates that input data files exist.
Checks:
  - data_path referenced in each config points to an existing file"""),

    ("06", "model_weights", "Model Weights",
     """Validates that model checkpoints are available (pre-experiment).
Checks:
  - Local checkpoint directory exists if hf_model_id is a local path
  - HuggingFace model_id is well-formed if remote"""),

    ("07", "gpu_device", "GPU / Device",
     """Validates GPU/CUDA availability for the experiment.
Checks:
  - CUDA available if device=cuda or device=auto
  - nvidia-smi reachable
  - Sufficient VRAM reported"""),

    ("08", "gec_backend", "GEC Backend",
     """Validates that the GEC model is accessible.
Checks:
  - grammarly/coedit-large (or configured model) is cached or downloadable"""),

    ("09", "disk_space", "Disk Space",
     """Validates sufficient disk space for outputs.
Checks:
  - >= 2GB free on output directory partition"""),

    ("10", "reproducibility", "Reproducibility",
     """Validates reproducibility requirements.
Checks:
  - Git hash available for the project
  - seed is set in pipeline configs
  - pyproject.toml exists"""),

    ("11", "code_quality", "Code Quality",
     """Validates minimum code quality.
Checks:
  - All .py files in src/ pass syntax check
  - No import crashes on core modules"""),

    ("12", "output_completeness", "Output Completeness",
     """Validates that pipeline outputs are complete (post-experiment).
Checks:
  - raw_results.json exists in output_dir
  - Summary JSON file exists
  - plots/ directory exists (unless skip_plots)"""),

    ("13", "results_validity", "Results Validity",
     """Validates output file contents (post-experiment).
Checks:
  - JSON files parse correctly
  - Summary contains required keys (mean_ppl, error_rate, etc.)
  - TSV/CSV files have expected columns"""),

    ("14", "error_annotations", "Error Annotations",
     """Validates ERRANT annotation output (post-experiment).
Checks:
  - At least 1 error annotation found in results
  - error_type_counts is non-empty
  - Region classification (prompt vs generation) present"""),

    ("15", "statistical_results", "Statistical Results",
     """Validates statistical analysis output (post-experiment).
Checks:
  - model_comparison.json exists if >1 model
  - Contains pairwise tests"""),

    ("16", "cloud_sync", "Cloud Sync",
     """Validates rclone cloud sync readiness.
Checks:
  - rclone is installed
  - Remote 'i:' is configured and reachable"""),

    ("17", "experiment_log", "Experiment Log",
     """Validates pipeline logging (post-experiment).
Checks:
  - Pipeline produced stdout/stderr log
  - Log contains stage completion markers"""),

    ("18", "gdrive_output", "GDrive Output Completeness",
     """Validates all expected outputs exist on GDrive (post-run).
Checks:
  - rclone can reach i: remote
  - 3 dataset directories exist (celva-sp, efcamdat-test, kupa-keys)
  - Each has 5 core files (raw_results.json, prompts.json, model_comparison.json, full_results.tsv, errors_long_format.tsv)
  - Each has 25 model summary JSONs ({model}_summary.json for 24 models + learner-baseline)
  - Each has 5 plots (perplexity_comparison, error_comparison, error_type_breakdown, ppl_vs_errors_scatter, combined_metric)
  - Grand total: 3 datasets x 35 files = 105 files"""),
]

TEMPLATE = '''\
#!/bin/bash
#===============================================================================
# enforcer_{rule_id}_{short_name}.sh — Rule {rule_id}: {title}
#===============================================================================
#
{description_commented}
#===============================================================================

set -uo pipefail

PROJECT_DIR="${{PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"
CONFIGS_DIR="${{CONFIGS_DIR:-$PROJECT_DIR/configs/pipeline}}"
DATA_DIR="${{DATA_DIR:-$PROJECT_DIR/data}}"
OUTPUT_DIR="${{OUTPUT_DIR:-$PROJECT_DIR/outputs}}"
MODEL_NAME="${{MODEL_NAME:-}}"

PASS=0
WARN=0
FAIL=0

log_pass() {{ echo "  ✓ $1"; PASS=$((PASS + 1)); }}
log_warn() {{ echo "  ⚠ $1"; WARN=$((WARN + 1)); }}
log_fail() {{ echo "  ✗ $1"; FAIL=$((FAIL + 1)); }}

echo "=== Rule {rule_id}: {title} ==="
echo ""

{checks}

echo ""
echo "--- Rule {rule_id} Summary: $PASS passed, $WARN warnings, $FAIL failed ---"
[[ $FAIL -eq 0 ]]
'''


def make_checks(rule_id, short_name, title, description):
    """Generate rule-specific check code."""
    checks_map = {
        "00": '''\
# Check all enforcer scripts exist
ENFORCER_DIR="$(dirname "$0")"
EXPECTED_COUNT={expected_count}
ACTUAL_COUNT=$(ls "$ENFORCER_DIR"/enforcer_*.sh 2>/dev/null | wc -l)
if [[ "$ACTUAL_COUNT" -ge "$EXPECTED_COUNT" ]]; then
    log_pass "All $ACTUAL_COUNT enforcer scripts exist"
else
    log_fail "Only $ACTUAL_COUNT/$EXPECTED_COUNT enforcer scripts found"
fi

# Check no enforcer is empty
EMPTY=0
for f in "$ENFORCER_DIR"/enforcer_*.sh; do
    if [[ ! -s "$f" ]]; then
        log_fail "Empty enforcer: $(basename "$f")"
        EMPTY=$((EMPTY + 1))
    fi
done
if [[ $EMPTY -eq 0 ]]; then
    log_pass "All enforcer scripts are non-empty"
fi
'''.format(expected_count=len(RULES)),

        "01": '''\
# Check Python version >= 3.10
PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
if [[ -n "$PY_VERSION" ]]; then
    MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
    MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
    if [[ "$MAJOR" -ge 3 && "$MINOR" -ge 10 ]]; then
        log_pass "Python $PY_VERSION >= 3.10"
    else
        log_fail "Python $PY_VERSION < 3.10"
    fi
else
    log_fail "python3 not found"
fi

# Check required packages
PACKAGES=(torch transformers errant spacy scipy pandas matplotlib yaml)
for pkg in "${PACKAGES[@]}"; do
    if python3 -c "import $pkg" 2>/dev/null; then
        log_pass "Package importable: $pkg"
    else
        log_fail "Package not importable: $pkg"
    fi
done
''',

        "02": '''\
# Check en_core_web_sm is installed
if python3 -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null; then
    log_pass "spaCy model en_core_web_sm loaded successfully"
else
    log_fail "spaCy model en_core_web_sm not installed (run: python -m spacy download en_core_web_sm)"
fi
''',

        "03": '''\
# Check pipeline is importable
if python3 -c "from gen_gec_errant.pipeline import run_pipeline" 2>/dev/null; then
    log_pass "gen_gec_errant.pipeline.run_pipeline importable"
else
    log_fail "Cannot import gen_gec_errant.pipeline.run_pipeline (check PYTHONPATH or installation)"
fi

# Check individual stages importable
STAGES=(data_loader generation gec annotation analysis)
for stage in "${STAGES[@]}"; do
    if python3 -c "import gen_gec_errant.$stage" 2>/dev/null; then
        log_pass "Stage importable: gen_gec_errant.$stage"
    else
        log_fail "Stage not importable: gen_gec_errant.$stage"
    fi
done
''',

        "04": '''\
# Check all YAML configs parse correctly
CONFIGS_FOUND=0
CONFIGS_VALID=0
REQUIRED_FIELDS="data_loader generation gec annotation models"

for cfg in "$CONFIGS_DIR"/*.yaml "$CONFIGS_DIR"/per-model/*.yaml; do
    [[ -f "$cfg" ]] || continue
    CONFIGS_FOUND=$((CONFIGS_FOUND + 1))
    if python3 -c "import yaml; yaml.safe_load(open('$cfg'))" 2>/dev/null; then
        CONFIGS_VALID=$((CONFIGS_VALID + 1))
    else
        log_fail "Invalid YAML: $(basename "$cfg")"
    fi
done

if [[ $CONFIGS_FOUND -eq 0 ]]; then
    log_fail "No YAML configs found in $CONFIGS_DIR"
elif [[ $CONFIGS_FOUND -eq $CONFIGS_VALID ]]; then
    log_pass "All $CONFIGS_FOUND configs parse correctly"
fi

# Check required fields in a sample config
SAMPLE_CFG=$(ls "$CONFIGS_DIR"/*.yaml 2>/dev/null | head -1)
if [[ -n "$SAMPLE_CFG" && -f "$SAMPLE_CFG" ]]; then
    for field in $REQUIRED_FIELDS; do
        if python3 -c "import yaml; d=yaml.safe_load(open('$SAMPLE_CFG')); assert '$field' in d" 2>/dev/null; then
            log_pass "Required field present: $field (in $(basename "$SAMPLE_CFG"))"
        else
            log_warn "Missing field: $field (in $(basename "$SAMPLE_CFG"))"
        fi
    done
fi
''',

        "05": '''\
# Check data files referenced in configs exist
DATA_PATHS_CHECKED=0
DATA_PATHS_OK=0

for cfg in "$CONFIGS_DIR"/*.yaml "$CONFIGS_DIR"/per-model/*.yaml; do
    [[ -f "$cfg" ]] || continue
    DATA_PATH=$(python3 -c "
import yaml
d = yaml.safe_load(open('$cfg'))
dl = d.get('data_loader', {})
print(dl.get('data_path', ''))
" 2>/dev/null)
    [[ -z "$DATA_PATH" ]] && continue

    DATA_PATHS_CHECKED=$((DATA_PATHS_CHECKED + 1))

    # Resolve relative paths against project dir
    if [[ "$DATA_PATH" != /* ]]; then
        DATA_PATH="$PROJECT_DIR/$DATA_PATH"
    fi

    if [[ -f "$DATA_PATH" ]]; then
        DATA_PATHS_OK=$((DATA_PATHS_OK + 1))
    else
        log_warn "Data file missing: $DATA_PATH (from $(basename "$cfg"))"
    fi
done

if [[ $DATA_PATHS_CHECKED -eq 0 ]]; then
    log_warn "No data_path fields found in configs"
elif [[ $DATA_PATHS_OK -eq $DATA_PATHS_CHECKED ]]; then
    log_pass "All $DATA_PATHS_OK data files exist"
else
    log_warn "$DATA_PATHS_OK/$DATA_PATHS_CHECKED data files found"
fi
''',

        "06": '''\
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
''',

        "07": '''\
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
''',

        "08": '''\
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
''',

        "09": '''\
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
''',

        "10": '''\
# Check git hash available
if git -C "$PROJECT_DIR" rev-parse HEAD >/dev/null 2>&1; then
    HASH=$(git -C "$PROJECT_DIR" rev-parse --short HEAD)
    log_pass "Git hash available: $HASH"
else
    log_warn "Not a git repository (reproducibility reduced)"
fi

# Check for uncommitted changes
if git -C "$PROJECT_DIR" diff --quiet 2>/dev/null; then
    log_pass "No uncommitted changes"
else
    log_warn "Uncommitted changes exist — results may not be reproducible"
fi

# Check seed is set in configs
if grep -rq "seed:" "$CONFIGS_DIR"/*.yaml "$CONFIGS_DIR"/per-model/*.yaml 2>/dev/null; then
    log_pass "Random seed configured in pipeline configs"
else
    log_warn "No explicit seed in pipeline configs"
fi

# Check pyproject.toml exists
if [[ -f "$PROJECT_DIR/pyproject.toml" ]]; then
    log_pass "pyproject.toml exists"
else
    log_fail "No pyproject.toml found"
fi
''',

        "11": '''\
# Check Python files have no syntax errors
SRC_DIR="$PROJECT_DIR/src"
SYNTAX_ERRORS=0
FILE_COUNT=0

if [[ -d "$SRC_DIR" ]]; then
    while IFS= read -r pyfile; do
        FILE_COUNT=$((FILE_COUNT + 1))
        if ! python3 -c "import py_compile; py_compile.compile('$pyfile', doraise=True)" 2>/dev/null; then
            log_fail "Syntax error: ${pyfile#$PROJECT_DIR/}"
            SYNTAX_ERRORS=$((SYNTAX_ERRORS + 1))
        fi
    done < <(find "$SRC_DIR" -name "*.py" 2>/dev/null)

    if [[ $SYNTAX_ERRORS -eq 0 && $FILE_COUNT -gt 0 ]]; then
        log_pass "All $FILE_COUNT Python files pass syntax check"
    fi
else
    log_fail "Source directory not found: $SRC_DIR"
fi

# Check core module imports
if python3 -c "from gen_gec_errant._config_utils import load_yaml_config" 2>/dev/null; then
    log_pass "Core config utils importable"
else
    log_warn "Cannot import config utils (may need pip install -e .)"
fi
''',

        "12": '''\
# Check output completeness (post-experiment)
if [[ -z "$OUTPUT_DIR" || ! -d "$OUTPUT_DIR" ]]; then
    log_fail "Output directory does not exist: ${OUTPUT_DIR:-<not set>}"
else
    # Check raw_results.json
    if [[ -f "$OUTPUT_DIR/raw_results.json" ]]; then
        log_pass "raw_results.json exists"
    else
        log_fail "raw_results.json missing in $OUTPUT_DIR"
    fi

    # Check summary JSON(s)
    SUMMARIES=$(ls "$OUTPUT_DIR"/*summary*.json 2>/dev/null | wc -l)
    if [[ $SUMMARIES -gt 0 ]]; then
        log_pass "$SUMMARIES summary JSON file(s) found"
    else
        log_fail "No summary JSON files in $OUTPUT_DIR"
    fi

    # Check plots directory
    if [[ -d "$OUTPUT_DIR/plots" ]]; then
        PLOT_COUNT=$(ls "$OUTPUT_DIR/plots/"*.png "$OUTPUT_DIR/plots/"*.pdf "$OUTPUT_DIR/plots/"*.svg 2>/dev/null | wc -l)
        log_pass "plots/ directory exists ($PLOT_COUNT plot files)"
    else
        log_warn "plots/ directory missing (may have used skip_plots=true)"
    fi
fi
''',

        "13": '''\
# Check results validity (post-experiment)
if [[ -z "$OUTPUT_DIR" || ! -d "$OUTPUT_DIR" ]]; then
    log_fail "Output directory does not exist: ${OUTPUT_DIR:-<not set>}"
else
    # Validate JSON files parse
    JSON_COUNT=0
    JSON_OK=0
    for jf in "$OUTPUT_DIR"/*.json; do
        [[ -f "$jf" ]] || continue
        JSON_COUNT=$((JSON_COUNT + 1))
        if python3 -c "import json; json.load(open('$jf'))" 2>/dev/null; then
            JSON_OK=$((JSON_OK + 1))
        else
            log_fail "Invalid JSON: $(basename "$jf")"
        fi
    done
    if [[ $JSON_COUNT -gt 0 && $JSON_OK -eq $JSON_COUNT ]]; then
        log_pass "All $JSON_COUNT JSON files parse correctly"
    fi

    # Check summary has required keys
    SUMMARY=$(ls "$OUTPUT_DIR"/*summary*.json 2>/dev/null | head -1)
    if [[ -n "$SUMMARY" && -f "$SUMMARY" ]]; then
        python3 -c "
import json, sys
d = json.load(open('$SUMMARY'))
# Accept either flat dict or nested per-model
if isinstance(d, dict):
    # If it has model sub-keys, check first model
    vals = d
    for k, v in d.items():
        if isinstance(v, dict):
            vals = v
            break
    required = ['mean_ppl', 'error_rate']
    found = [k for k in required if k in vals or any(k in str(kk) for kk in vals)]
    missing = [k for k in required if k not in vals and not any(k in str(kk) for kk in vals)]
    if found:
        print('PASS:Summary has keys: ' + ', '.join(found))
    if missing:
        print('WARN:Summary missing keys: ' + ', '.join(missing))
" 2>&1 | while IFS= read -r line; do
            case "$line" in
                PASS:*) log_pass "${line#PASS:}" ;;
                WARN:*) log_warn "${line#WARN:}" ;;
                FAIL:*) log_fail "${line#FAIL:}" ;;
            esac
        done
    fi

    # Check CSV/TSV files have columns
    for tf in "$OUTPUT_DIR"/*.tsv "$OUTPUT_DIR"/*.csv; do
        [[ -f "$tf" ]] || continue
        COLS=$(head -1 "$tf" | tr '\\t' '\\n' | wc -l)
        if [[ $COLS -gt 1 ]]; then
            log_pass "$(basename "$tf") has $COLS columns"
        else
            log_warn "$(basename "$tf") has only $COLS column"
        fi
    done
fi
''',

        "14": '''\
# Check error annotations (post-experiment)
if [[ -z "$OUTPUT_DIR" || ! -d "$OUTPUT_DIR" ]]; then
    log_fail "Output directory does not exist: ${OUTPUT_DIR:-<not set>}"
else
    # Check raw_results.json for annotations
    if [[ -f "$OUTPUT_DIR/raw_results.json" ]]; then
        python3 -c "
import json, sys
data = json.load(open('$OUTPUT_DIR/raw_results.json'))

# Look for error annotations in results
results = data if isinstance(data, list) else data.get('results', data.get('sentences', []))
if isinstance(data, dict):
    # Try to find any key with annotation data
    for k, v in data.items():
        if isinstance(v, list) and len(v) > 0:
            results = v
            break

anno_count = 0
has_types = False
has_region = False

for r in (results if isinstance(results, list) else []):
    if isinstance(r, dict):
        errs = r.get('errors', r.get('annotations', []))
        if isinstance(errs, list):
            anno_count += len(errs)
        if r.get('error_type_counts') or r.get('error_types'):
            has_types = True
        if 'region' in str(r) or 'prompt_errors' in str(r) or 'gen_errors' in str(r):
            has_region = True

if anno_count > 0:
    print(f'PASS:Found {anno_count} error annotations')
else:
    print('WARN:No error annotations found in raw_results.json')

if has_types:
    print('PASS:Error type counts present')

if has_region:
    print('PASS:Region classification present')
" 2>&1 | while IFS= read -r line; do
            case "$line" in
                PASS:*) log_pass "${line#PASS:}" ;;
                WARN:*) log_warn "${line#WARN:}" ;;
                FAIL:*) log_fail "${line#FAIL:}" ;;
            esac
        done
    else
        log_fail "raw_results.json not found — cannot check annotations"
    fi
fi
''',

        "15": '''\
# Check statistical results (post-experiment)
if [[ -z "$OUTPUT_DIR" || ! -d "$OUTPUT_DIR" ]]; then
    log_fail "Output directory does not exist: ${OUTPUT_DIR:-<not set>}"
else
    # Check for model comparison results
    if [[ -f "$OUTPUT_DIR/model_comparison.json" ]]; then
        log_pass "model_comparison.json exists"

        # Check it contains pairwise tests
        python3 -c "
import json
d = json.load(open('$OUTPUT_DIR/model_comparison.json'))
if 'pairwise' in str(d).lower() or 'wilcoxon' in str(d).lower() or 'p_value' in str(d).lower():
    print('PASS:Contains statistical test results')
else:
    print('WARN:No pairwise test results found')
" 2>&1 | while IFS= read -r line; do
            case "$line" in
                PASS:*) log_pass "${line#PASS:}" ;;
                WARN:*) log_warn "${line#WARN:}" ;;
            esac
        done
    else
        # Check if there was only 1 model (comparison not applicable)
        MODEL_COUNT=$(python3 -c "
import json
d = json.load(open('$OUTPUT_DIR/raw_results.json'))
models = set()
results = d if isinstance(d, list) else d.get('results', [])
for r in (results if isinstance(results, list) else []):
    if isinstance(r, dict) and 'model' in r:
        models.add(r['model'])
print(len(models))
" 2>/dev/null)
        if [[ "${MODEL_COUNT:-0}" -le 1 ]]; then
            log_pass "Single model — model_comparison.json not required"
        else
            log_warn "model_comparison.json missing for multi-model run"
        fi
    fi
fi
''',

        "16": '''\
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
''',

        "17": '''\
# Check experiment log (post-experiment)
if [[ -z "$OUTPUT_DIR" || ! -d "$OUTPUT_DIR" ]]; then
    log_fail "Output directory does not exist: ${OUTPUT_DIR:-<not set>}"
else
    # Check for pipeline log file
    LOG_FILE=$(ls "$OUTPUT_DIR"/pipeline*.log "$OUTPUT_DIR"/*.log 2>/dev/null | head -1)
    if [[ -n "$LOG_FILE" ]]; then
        log_pass "Pipeline log found: $(basename "$LOG_FILE")"

        # Check for stage completion markers
        for marker in "data_loader" "generation" "gec" "annotation" "analysis"; do
            if grep -qi "$marker" "$LOG_FILE" 2>/dev/null; then
                log_pass "Log contains stage marker: $marker"
            else
                log_warn "Log missing stage marker: $marker"
            fi
        done
    else
        log_warn "No pipeline log file found in $OUTPUT_DIR"
    fi
fi
''',

        "18": '''\
# Check GDrive output completeness (post-run)
# Reads configs/expected_outputs.yaml:
#   - 3 dataset directories
#   - 25 models (24 + learner-baseline) -> {model}_summary.json each
#   - 5 core files + 5 plots per dataset = 35 files per dataset
#   - Grand total: 3 x 35 = 105 files
EXPECTED_OUTPUTS="$PROJECT_DIR/configs/expected_outputs.yaml"

if [[ ! -f "$EXPECTED_OUTPUTS" ]]; then
    log_fail "Expected outputs manifest not found: $EXPECTED_OUTPUTS"
else
    log_pass "Expected outputs manifest found"

    # Verify rclone can reach remote
    if ! command -v rclone >/dev/null 2>&1; then
        log_fail "rclone not installed"
    elif ! rclone lsd "i:" --max-depth 0 >/dev/null 2>&1; then
        log_fail "rclone remote 'i:' not reachable"
    else
        log_pass "rclone remote 'i:' reachable"

        # Check each dataset directory for all expected files
        python3 -c "
import yaml, subprocess, sys

with open('$EXPECTED_OUTPUTS') as f:
    spec = yaml.safe_load(f)

base = spec['gdrive_base']
datasets = spec['datasets']
models = spec['models']
req_files = spec.get('required_files', [])
req_plots = spec.get('required_plots', [])

# Build full expected file list per dataset
model_summaries = [f'{m}_summary.json' for m in models]
all_expected = req_files + model_summaries + req_plots
files_per_ds = len(all_expected)

total_ok = 0
total_expected = len(datasets) * files_per_ds

for ds in datasets:
    ds_path = f'{base}/{ds}'
    print(f'INFO:')
    print(f'INFO:--- {ds} ({ds_path}) ---')

    try:
        out = subprocess.run(
            ['rclone', 'lsf', '-R', ds_path + '/'],
            capture_output=True, text=True, timeout=30
        )
    except Exception as e:
        print(f'FAIL:{ds}: cannot list directory ({e})')
        continue

    if out.returncode != 0 or not out.stdout.strip():
        print(f'FAIL:{ds}: directory empty or absent')
        continue

    remote_files = set(out.stdout.strip().splitlines())

    # Check core data files
    for f in req_files:
        if f in remote_files:
            print(f'PASS:{ds}/{f}')
            total_ok += 1
        else:
            print(f'FAIL:{ds}/{f}: MISSING')

    # Check per-model summary JSONs
    summaries_ok = 0
    summaries_missing = []
    for sf in model_summaries:
        if sf in remote_files:
            summaries_ok += 1
            total_ok += 1
        else:
            summaries_missing.append(sf)
    if summaries_ok == len(model_summaries):
        print(f'PASS:{ds}: all {summaries_ok} model summaries present')
    else:
        for sf in summaries_missing:
            print(f'FAIL:{ds}/{sf}: MISSING')
        print(f'WARN:{ds}: {summaries_ok}/{len(model_summaries)} model summaries')

    # Check plots
    plots_ok = 0
    plots_missing = []
    for pf in req_plots:
        if pf in remote_files:
            plots_ok += 1
            total_ok += 1
        else:
            plots_missing.append(pf)
    if plots_ok == len(req_plots):
        print(f'PASS:{ds}: all {plots_ok} plots present')
    else:
        for pf in plots_missing:
            print(f'FAIL:{ds}/{pf}: MISSING')

print(f'INFO:')
print(f'INFO:=== GRAND TOTAL: {total_ok}/{total_expected} files present across {len(datasets)} datasets ===')
print(f'INFO:  Per dataset: {len(req_files)} core + {len(model_summaries)} summaries + {len(req_plots)} plots = {files_per_ds}')
" 2>&1 | while IFS= read -r line; do
            case "$line" in
                PASS:*) log_pass "${line#PASS:}" ;;
                WARN:*) log_warn "${line#WARN:}" ;;
                FAIL:*) log_fail "${line#FAIL:}" ;;
                INFO:*) echo "  ${line#INFO:}" ;;
            esac
        done
    fi
fi
''',
    }

    return checks_map.get(rule_id, f'''\
# Generic validation for rule {rule_id}
log_warn "Detailed checks not yet implemented for rule {rule_id}: {title}"
''')


# ── Generate individual enforcer scripts ────────────────────────────────
for rule_id, short_name, title, description in RULES:
    desc_commented = "\n".join(
        f"# {line}" if line.strip() else "#"
        for line in description.strip().splitlines()
    )
    checks = make_checks(rule_id, short_name, title, description)

    content = TEMPLATE.format(
        rule_id=rule_id,
        short_name=short_name,
        title=title,
        description_commented=desc_commented,
        checks=checks,
    )

    filename = f"enforcer_{rule_id}_{short_name}.sh"
    filepath = os.path.join(ENFORCERS_DIR, filename)
    with open(filepath, 'w') as f:
        f.write(content)
    os.chmod(filepath, os.stat(filepath).st_mode | stat.S_IEXEC)

print(f"Generated {len(RULES)} enforcer scripts in {ENFORCERS_DIR}")


# ── Generate the master runner ──────────────────────────────────────────
master = '''\
#!/bin/bash
#===============================================================================
# run_all_enforcers.sh — Run all gen-gec-errant enforcement checks
#===============================================================================
# Usage:
#   bash deploy/enforcers/run_all_enforcers.sh                # run all
#   bash deploy/enforcers/run_all_enforcers.sh 1 2 3          # run specific rules
#   bash deploy/enforcers/run_all_enforcers.sh --pre-flight   # pre-flight only
#   bash deploy/enforcers/run_all_enforcers.sh --pre-experiment  # pre-experiment only
#   bash deploy/enforcers/run_all_enforcers.sh --post-experiment # post-experiment only
#
# Environment variables:
#   PROJECT_DIR   — pipeline root (default: auto-detect from script location)
#   CONFIGS_DIR   — configs directory
#   OUTPUT_DIR    — output directory (for post-experiment checks)
#   MODEL_NAME    — model name (for pre-experiment checks)
#===============================================================================

set -uo pipefail

ENFORCER_DIR="$(cd "$(dirname "$0")" && pwd)"
TOTAL_PASS=0
TOTAL_WARN=0
TOTAL_FAIL=0
RULES_RUN=0

PRE_FLIGHT_ENFORCERS=(1 2 3 4 5 10 11)
PRE_EXPERIMENT_ENFORCERS=(6 7 8 9)
POST_EXPERIMENT_ENFORCERS=(12 13 14 15 16 17 18)

run_enforcer() {
    local script="$1"
    local output
    output=$(bash "$script" 2>&1)
    echo "$output"
    echo ""

    local p w f
    p=$(echo "$output" | grep -c "✓" || true)
    w=$(echo "$output" | grep -c "⚠" || true)
    f=$(echo "$output" | grep -c "✗" || true)
    TOTAL_PASS=$((TOTAL_PASS + p))
    TOTAL_WARN=$((TOTAL_WARN + w))
    TOTAL_FAIL=$((TOTAL_FAIL + f))
    RULES_RUN=$((RULES_RUN + 1))
}

run_rules() {
    for rule_num in "$@"; do
        local padded
        padded=$(printf "%02d" "$rule_num" 2>/dev/null || echo "$rule_num")
        for script in "$ENFORCER_DIR"/enforcer_${padded}_*.sh; do
            if [[ -f "$script" ]]; then
                run_enforcer "$script"
            else
                echo "No enforcer found for rule $rule_num"
            fi
        done
    done
}

# Parse arguments
if [[ $# -gt 0 ]]; then
    case "$1" in
        --pre-flight)
            echo "=== PRE-FLIGHT CHECKS ==="
            echo ""
            run_rules "${PRE_FLIGHT_ENFORCERS[@]}"
            ;;
        --pre-experiment)
            echo "=== PRE-EXPERIMENT CHECKS ==="
            echo ""
            run_rules "${PRE_EXPERIMENT_ENFORCERS[@]}"
            ;;
        --post-experiment)
            echo "=== POST-EXPERIMENT CHECKS ==="
            echo ""
            run_rules "${POST_EXPERIMENT_ENFORCERS[@]}"
            ;;
        *)
            # Run specific rules by number
            run_rules "$@"
            ;;
    esac
else
    # Run all enforcers (skip 00 index, run it last)
    for script in "$ENFORCER_DIR"/enforcer_*.sh; do
        [[ "$(basename "$script")" == "enforcer_00_index.sh" ]] && continue
        run_enforcer "$script"
    done
    # Run index last (it checks all others exist)
    if [[ -f "$ENFORCER_DIR/enforcer_00_index.sh" ]]; then
        run_enforcer "$ENFORCER_DIR/enforcer_00_index.sh"
    fi
fi

echo "================================================================"
echo "=== ENFORCEMENT SUMMARY ==="
echo "  Rules checked: $RULES_RUN"
echo "  Passed: $TOTAL_PASS"
echo "  Warnings: $TOTAL_WARN"
echo "  Failed: $TOTAL_FAIL"
echo "================================================================"

[[ $TOTAL_FAIL -eq 0 ]]
'''

master_path = os.path.join(ENFORCERS_DIR, "run_all_enforcers.sh")
with open(master_path, 'w') as f:
    f.write(master)
os.chmod(master_path, os.stat(master_path).st_mode | stat.S_IEXEC)
print(f"Generated master runner: {master_path}")
