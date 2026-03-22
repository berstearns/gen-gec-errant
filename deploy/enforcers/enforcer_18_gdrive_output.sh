#!/bin/bash
#===============================================================================
# enforcer_18_gdrive_output.sh — Rule 18: GDrive Output Completeness
#===============================================================================
#
# Validates all expected outputs exist on GDrive (post-run).
# Checks:
#   - rclone can reach i: remote
#   - 3 dataset directories exist (celva-sp, efcamdat-test, kupa-keys)
#   - Each has 5 core files (raw_results.json, prompts.json, model_comparison.json, full_results.tsv, errors_long_format.tsv)
#   - Each has 25 model summary JSONs ({model}_summary.json for 24 models + learner-baseline)
#   - Each has 5 plots (perplexity_comparison, error_comparison, error_type_breakdown, ppl_vs_errors_scatter, combined_metric)
#   - Grand total: 3 datasets x 35 files = 105 files
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

echo "=== Rule 18: GDrive Output Completeness ==="
echo ""

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


echo ""
echo "--- Rule 18 Summary: $PASS passed, $WARN warnings, $FAIL failed ---"
[[ $FAIL -eq 0 ]]
