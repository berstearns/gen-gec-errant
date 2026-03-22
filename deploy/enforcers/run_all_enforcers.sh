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
