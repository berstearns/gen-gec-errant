#!/usr/bin/env python3
"""
End-to-end GPT-2 Native Generation Error Profile Experiment.

Runs the full generate → GEC → ERRANT → analysis pipeline using
pre-trained GPT-2 on multiple learner datasets.

DOES NOT MODIFY ANYTHING IN src/.
All outputs go into the experiment/ subdirectory.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
REPRO_DIR = Path(__file__).resolve().parents[1]   # paper-reproducibility-.../
PROJECT_ROOT = REPRO_DIR.parents[1]                # automatic-generation-...-v2/
EXP_DIR = REPRO_DIR / "experiment"
DATA_SRC = Path(
    "/home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits"
)

# Datasets to process
DATASETS = {
    "norm-CELVA-SP": {
        "src": DATA_SRC / "norm-CELVA-SP.csv",
        "description": "Spanish L1 learner English (primary)",
    },
    "norm-EFCAMDAT-test": {
        "src": DATA_SRC / "norm-EFCAMDAT-test.csv",
        "description": "In-domain learner English",
    },
    "norm-KUPA-KEYS": {
        "src": DATA_SRC / "norm-KUPA-KEYS.csv",
        "description": "Cross-corpus learner English",
    },
}

# Sentence limits for CPU runs (set to None for full data)
LIMITS = {
    "norm-CELVA-SP":       50,
    "norm-EFCAMDAT-test":  50,
    "norm-KUPA-KEYS":      50,
}

# ── Generation & GEC configuration ────────────────────────────────────────
MODEL_NAME = "gpt2-small"
HF_MODEL_ID = "gpt2"
MODEL_FAMILY = "gpt2"
GEC_MODEL = "grammarly/coedit-large"
GEC_METHOD = "dedicated"

GENERATION_PARAMS = {
    "max_new_tokens": 50,
    "min_new_tokens": 10,
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 0.95,
    "do_sample": True,
    "repetition_penalty": 1.2,
}

DATA_LOADER_PARAMS = {
    "text_column": "text",
    "min_words": 10,
    "max_words": 500,
    "prompt_ratio": 0.5,
    "min_prompt_words": 5,
}

BATCH_SIZE = 2
DEVICE = "auto"
SEED = 42


def write_config(dataset_name: str, data_path: Path, output_dir: Path, max_sentences):
    """Write a YAML pipeline config for one dataset."""
    config_dir = EXP_DIR / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{dataset_name}.yaml"

    max_sent_line = f"  max_sentences: {max_sentences}" if max_sentences else "  max_sentences: null"

    yaml_content = f"""data_loader:
  data_path: {data_path}
  text_column: {DATA_LOADER_PARAMS['text_column']}
{max_sent_line}
  min_words: {DATA_LOADER_PARAMS['min_words']}
  max_words: {DATA_LOADER_PARAMS['max_words']}
  prompt_ratio: {DATA_LOADER_PARAMS['prompt_ratio']}
  min_prompt_words: {DATA_LOADER_PARAMS['min_prompt_words']}

generation:
  max_new_tokens: {GENERATION_PARAMS['max_new_tokens']}
  min_new_tokens: {GENERATION_PARAMS['min_new_tokens']}
  temperature: {GENERATION_PARAMS['temperature']}
  top_k: {GENERATION_PARAMS['top_k']}
  top_p: {GENERATION_PARAMS['top_p']}
  do_sample: {str(GENERATION_PARAMS['do_sample']).lower()}
  repetition_penalty: {GENERATION_PARAMS['repetition_penalty']}

gec:
  method: {GEC_METHOD}
  model_id: {GEC_MODEL}
  batch_size: 4
  device: {DEVICE}

annotation:
  lang: en

analysis:
  skip_plots: false
  top_n_error_types: 10

models:
  - name: {MODEL_NAME}
    hf_model_id: {HF_MODEL_ID}
    model_family: {MODEL_FAMILY}

batch_size: {BATCH_SIZE}
device: {DEVICE}
seed: {SEED}
output_dir: {output_dir}
skip_plots: false
"""
    config_path.write_text(yaml_content)
    print(f"  Wrote config: {config_path}")
    return config_path


def run_cmd(cmd, desc="", cwd=None, timeout=7200):
    """Run a shell command, print output, raise on failure."""
    print(f"\n{'='*70}")
    print(f"  {desc}")
    print(f"  CMD: {' '.join(str(c) for c in cmd)}")
    print(f"{'='*70}")
    t0 = time.time()
    result = subprocess.run(
        [str(c) for c in cmd],
        cwd=cwd or str(PROJECT_ROOT),
        capture_output=True, text=True, timeout=timeout,
    )
    elapsed = time.time() - t0
    if result.stdout:
        lines = result.stdout.strip().split("\n")
        if len(lines) > 40:
            print(f"  ... ({len(lines)-40} lines omitted)")
        for line in lines[-40:]:
            print(f"  {line}")
    if result.returncode != 0:
        print(f"  STDERR (last 2000 chars): {result.stderr[-2000:]}")
        raise RuntimeError(f"Command failed (rc={result.returncode}): {desc}")
    print(f"  Done in {elapsed:.1f}s")
    return result


def step0_setup():
    """Verify data and create experiment directory structure."""
    print("\n" + "#"*70)
    print("# STEP 0: Setup experiment directory")
    print("#"*70)

    EXP_DIR.mkdir(parents=True, exist_ok=True)

    for name, info in DATASETS.items():
        if not info["src"].exists():
            raise FileNotFoundError(f"Missing source data: {info['src']}")
        print(f"  OK: {info['src'].name} ({info['description']})")

    print("  Setup complete.")


def step1_run_pipeline(dataset_name: str):
    """Run the full gen-gec-errant pipeline for one dataset."""
    info = DATASETS[dataset_name]
    output_dir = EXP_DIR / dataset_name
    limit = LIMITS.get(dataset_name)

    # Check if already completed
    raw_results = output_dir / "raw_results.json"
    if raw_results.exists():
        print(f"  Pipeline already completed: {dataset_name} ({raw_results})")
        return

    # Write config
    config_path = write_config(
        dataset_name=dataset_name,
        data_path=info["src"],
        output_dir=output_dir,
        max_sentences=limit,
    )

    # Run pipeline
    cmd = [
        sys.executable, "-m", "gen_gec_errant.pipeline",
        "--config", str(config_path),
    ]

    run_cmd(cmd, desc=f"Pipeline: {dataset_name} (limit={limit})")


def step2_cross_dataset_summary():
    """Aggregate results across all datasets into a single summary."""
    print("\n" + "#"*70)
    print("# STEP 2: Cross-dataset summary")
    print("#"*70)

    summary = {}
    for dataset_name in DATASETS:
        output_dir = EXP_DIR / dataset_name

        # Load model summary
        model_summary_path = output_dir / f"{MODEL_NAME}_summary.json"
        baseline_summary_path = output_dir / "learner_baseline_summary.json"

        dataset_summary = {"dataset": dataset_name}

        if model_summary_path.exists():
            with open(model_summary_path) as f:
                model_data = json.load(f)
            dataset_summary["gpt2_native"] = {
                "ppl_mean": model_data.get("ppl_mean"),
                "ppl_median": model_data.get("ppl_median"),
                "total_errors": model_data.get("total_errors"),
                "avg_errors_per_sentence": model_data.get("avg_errors_per_sentence"),
                "error_rate": model_data.get("error_rate"),
                "top_error_types": model_data.get("top_10_error_types", [])[:5],
            }
        else:
            print(f"  Warning: No model summary for {dataset_name}")

        if baseline_summary_path.exists():
            with open(baseline_summary_path) as f:
                baseline_data = json.load(f)
            dataset_summary["learner_baseline"] = {
                "total_errors": baseline_data.get("total_errors"),
                "avg_errors_per_sentence": baseline_data.get("avg_errors_per_sentence"),
                "error_rate": baseline_data.get("error_rate"),
                "top_error_types": baseline_data.get("top_10_error_types", [])[:5],
            }

        summary[dataset_name] = dataset_summary

    summary_path = EXP_DIR / "cross_dataset_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"  Saved cross-dataset summary: {summary_path}")

    # Print summary table
    print("\n  Cross-Dataset Summary:")
    print(f"  {'Dataset':<25} {'PPL Mean':>10} {'Errors/Sent':>12} {'Error Rate':>12}")
    print(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*12}")
    for name, data in summary.items():
        gpt2 = data.get("gpt2_native", {})
        ppl = gpt2.get("ppl_mean", "N/A")
        avg_err = gpt2.get("avg_errors_per_sentence", "N/A")
        err_rate = gpt2.get("error_rate", "N/A")
        ppl_str = f"{ppl:.2f}" if isinstance(ppl, (int, float)) else str(ppl)
        avg_str = f"{avg_err:.2f}" if isinstance(avg_err, (int, float)) else str(avg_err)
        rate_str = f"{err_rate:.3f}" if isinstance(err_rate, (int, float)) else str(err_rate)
        print(f"  {name:<25} {ppl_str:>10} {avg_str:>12} {rate_str:>12}")


def main():
    t_start = time.time()
    print("="*70)
    print(" GPT-2 Native Generation Error Profile Experiment")
    print(f" Project root: {PROJECT_ROOT}")
    print(f" Experiment dir: {EXP_DIR}")
    print(f" Data source: {DATA_SRC}")
    print(f" Model: {HF_MODEL_ID} ({MODEL_NAME})")
    print(f" GEC: {GEC_MODEL}")
    print(f" Datasets: {', '.join(DATASETS.keys())}")
    print("="*70)

    # ── Step 0: Setup ──
    step0_setup()

    # ── Step 1: Run pipeline per dataset ──
    print("\n" + "#"*70)
    print("# STEP 1: Run generate → GEC → ERRANT → analysis pipeline")
    print("#"*70)

    for dataset_name in DATASETS:
        print(f"\n--- Processing: {dataset_name} ---")
        step1_run_pipeline(dataset_name)

    # ── Step 2: Cross-dataset summary ──
    step2_cross_dataset_summary()

    elapsed = time.time() - t_start
    print("\n" + "="*70)
    print(f" EXPERIMENT COMPLETE in {elapsed/60:.1f} minutes")
    print(f" Results: {EXP_DIR}")
    print(f" Per-dataset outputs:")
    for name in DATASETS:
        print(f"   {EXP_DIR / name}/")
    print(f" Summary: {EXP_DIR / 'cross_dataset_summary.json'}")
    print("="*70)


if __name__ == "__main__":
    main()
