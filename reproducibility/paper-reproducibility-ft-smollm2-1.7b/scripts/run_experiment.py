#!/usr/bin/env python3
"""
End-to-end SmolLM2 1.7B fine-tuned on EFCAMDAT all-data Error Profile Experiment.

Runs the full generate -> GEC -> ERRANT -> analysis pipeline using
ft-smollm2-1.7b (fine-tuned on EFCAMDAT all-data) on multiple learner datasets.

DOES NOT MODIFY ANYTHING IN src/.
All outputs go into the experiment/ subdirectory.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

# -- Paths -----------------------------------------------------------------
REPRO_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPRO_DIR.parents[1]
EXP_DIR = REPRO_DIR / "experiment"
DATA_SRC = Path(
    "/home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits"
)

# -- Model -----------------------------------------------------------------
MODEL_NAME = "ft-smollm2-1.7b"
HF_MODEL_ID = "/home/b/p/my-data/i/_p/artificial-learners/models/smollm2/smollm2-1.7b-all-data/final"
BASE_HF_ID = "HuggingFaceTB/SmolLM2-1.7B"
MODEL_FAMILY = "smollm2"
MODEL_PARAMS = "1.7B"
RCLONE_DIR = "i:/_p/artificial-learners/models/smollm2/smollm2-1.7b-all-data"
LOCAL_DIR = Path("/home/b/p/my-data/i/_p/artificial-learners/models/smollm2/smollm2-1.7b-all-data")
IS_LEARNER_TUNED = True

# -- Datasets --------------------------------------------------------------
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

# -- Pipeline config -------------------------------------------------------
GEC_MODEL = "grammarly/coedit-large"
GEC_METHOD = "dedicated"
BATCH_SIZE = 2
DEVICE = "auto"
SEED = 42


def run_cmd(cmd, desc="", cwd=None, timeout=7200):
    """Run a shell command, print output, raise on failure."""
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  {desc}")
    print(f"  CMD: {' '.join(str(c) for c in cmd)}")
    print(sep)
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


def step0_download_model():
    """Download model from Google Drive via rclone if not present locally."""
    print("\n" + "#" * 70)
    print("# STEP 0: Ensure model weights are available")
    print("#" * 70)

    final_dir = LOCAL_DIR / "final"
    config_json = final_dir / "config.json"

    if config_json.exists():
        print(f"  Model already available: {final_dir}")
        return

    print(f"  Downloading from: {RCLONE_DIR}/final")
    print(f"  Downloading to:   {final_dir}")

    final_dir.mkdir(parents=True, exist_ok=True)
    rclone_src = f"{RCLONE_DIR}/final"
    run_cmd(
        ["rclone", "copy", rclone_src, str(final_dir), "--progress"],
        desc=f"Download {MODEL_NAME} weights via rclone",
        cwd=str(Path.home()),
        timeout=3600,
    )

    if not config_json.exists():
        raise FileNotFoundError(
            f"Download failed: {config_json} not found after rclone copy"
        )
    print(f"  Model downloaded successfully: {final_dir}")


def step1_setup():
    """Verify data and create experiment directory structure."""
    print("\n" + "#" * 70)
    print("# STEP 1: Setup experiment directory")
    print("#" * 70)

    EXP_DIR.mkdir(parents=True, exist_ok=True)

    for name, info in DATASETS.items():
        src = info["src"]
        if not src.exists():
            raise FileNotFoundError(f"Missing source data: {src}")
        print(f"  OK: {src.name} ({info['description']})")

    print("  Setup complete.")


def write_config(dataset_name, data_path, output_dir, max_sentences):
    """Write a YAML pipeline config for one dataset."""
    config_dir = EXP_DIR / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{dataset_name}.yaml"

    if max_sentences:
        max_sent_line = f"  max_sentences: {max_sentences}"
    else:
        max_sent_line = "  max_sentences: null"

    learner_flag = str(IS_LEARNER_TUNED).lower()

    yaml_content = (
        f"data_loader:\n"
        f"  data_path: {data_path}\n"
        f"  text_column: text\n"
        f"{max_sent_line}\n"
        f"  min_words: 10\n"
        f"  max_words: 500\n"
        f"  prompt_ratio: 0.5\n"
        f"  min_prompt_words: 5\n"
        f"\n"
        f"generation:\n"
        f"  max_new_tokens: 50\n"
        f"  min_new_tokens: 10\n"
        f"  temperature: 1.0\n"
        f"  top_k: 50\n"
        f"  top_p: 0.95\n"
        f"  do_sample: true\n"
        f"  repetition_penalty: 1.2\n"
        f"\n"
        f"gec:\n"
        f"  method: {GEC_METHOD}\n"
        f"  model_id: {GEC_MODEL}\n"
        f"  batch_size: 4\n"
        f"  device: {DEVICE}\n"
        f"\n"
        f"annotation:\n"
        f"  lang: en\n"
        f"\n"
        f"analysis:\n"
        f"  skip_plots: false\n"
        f"  top_n_error_types: 10\n"
        f"\n"
        f"models:\n"
        f"  - name: {MODEL_NAME}\n"
        f"    hf_model_id: {HF_MODEL_ID}\n"
        f"    model_family: {MODEL_FAMILY}\n"
        f"    is_learner_tuned: {learner_flag}\n"
        f"\n"
        f"batch_size: {BATCH_SIZE}\n"
        f"device: {DEVICE}\n"
        f"seed: {SEED}\n"
        f"output_dir: {output_dir}\n"
        f"skip_plots: false\n"
    )
    config_path.write_text(yaml_content)
    print(f"  Wrote config: {config_path}")
    return config_path


def step2_run_pipeline(dataset_name):
    """Run the full gen-gec-errant pipeline for one dataset."""
    info = DATASETS[dataset_name]
    output_dir = EXP_DIR / dataset_name
    limit = LIMITS.get(dataset_name)

    raw_results = output_dir / "raw_results.json"
    if raw_results.exists():
        print(f"  Pipeline already completed: {dataset_name} ({raw_results})")
        return

    config_path = write_config(
        dataset_name=dataset_name,
        data_path=info["src"],
        output_dir=output_dir,
        max_sentences=limit,
    )

    cmd = [
        sys.executable, "-m", "gen_gec_errant.pipeline",
        "--config", str(config_path),
    ]
    run_cmd(cmd, desc=f"Pipeline: {dataset_name} (limit={limit})")


def step3_cross_dataset_summary():
    """Aggregate results across all datasets."""
    print("\n" + "#" * 70)
    print("# STEP 3: Cross-dataset summary")
    print("#" * 70)

    summary = {}
    for dataset_name in DATASETS:
        output_dir = EXP_DIR / dataset_name
        dataset_summary = {"dataset": dataset_name}

        model_summary_path = output_dir / f"{MODEL_NAME}_summary.json"
        baseline_summary_path = output_dir / "learner_baseline_summary.json"

        if model_summary_path.exists():
            with open(model_summary_path) as f:
                data = json.load(f)
            dataset_summary["model"] = {
                "ppl_mean": data.get("ppl_mean"),
                "ppl_median": data.get("ppl_median"),
                "total_errors": data.get("total_errors"),
                "avg_errors_per_sentence": data.get("avg_errors_per_sentence"),
                "error_rate": data.get("error_rate"),
                "top_error_types": data.get("top_10_error_types", [])[:5],
            }
        else:
            print(f"  Warning: No model summary for {dataset_name}")

        if baseline_summary_path.exists():
            with open(baseline_summary_path) as f:
                data = json.load(f)
            dataset_summary["learner_baseline"] = {
                "total_errors": data.get("total_errors"),
                "avg_errors_per_sentence": data.get("avg_errors_per_sentence"),
                "error_rate": data.get("error_rate"),
                "top_error_types": data.get("top_10_error_types", [])[:5],
            }

        summary[dataset_name] = dataset_summary

    summary_path = EXP_DIR / "cross_dataset_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"  Saved: {summary_path}")
    print(f"\n  Cross-Dataset Summary for {MODEL_NAME}:")
    header = f"  {'Dataset':<25} {'PPL Mean':>10} {'Errors/Sent':>12} {'Error Rate':>12}"
    print(header)
    print(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*12}")
    for name, data in summary.items():
        m = data.get("model", {})
        ppl = m.get("ppl_mean", "N/A")
        avg_err = m.get("avg_errors_per_sentence", "N/A")
        err_rate = m.get("error_rate", "N/A")
        ppl_s = f"{ppl:.2f}" if isinstance(ppl, (int, float)) else str(ppl)
        avg_s = f"{avg_err:.2f}" if isinstance(avg_err, (int, float)) else str(avg_err)
        rate_s = f"{err_rate:.3f}" if isinstance(err_rate, (int, float)) else str(err_rate)
        print(f"  {name:<25} {ppl_s:>10} {avg_s:>12} {rate_s:>12}")


def main():
    t_start = time.time()
    sep = "=" * 70
    print(sep)
    print(f" {MODEL_NAME} Error Profile Experiment")
    print(f" Model: {MODEL_NAME} ({MODEL_PARAMS})")
    print(f" Base: {BASE_HF_ID}")
    print(f" Local: {HF_MODEL_ID}")
    print(f" rclone: {RCLONE_DIR}")
    print(f" Experiment dir: {EXP_DIR}")
    print(sep)

    step0_download_model()
    step1_setup()

    print("\n" + "#" * 70)
    print("# STEP 2: Run generate -> GEC -> ERRANT -> analysis pipeline")
    print("#" * 70)
    for dataset_name in DATASETS:
        print(f"\n--- Processing: {dataset_name} ---")
        step2_run_pipeline(dataset_name)

    step3_cross_dataset_summary()

    elapsed = time.time() - t_start
    print("\n" + sep)
    print(f" EXPERIMENT COMPLETE in {elapsed/60:.1f} minutes")
    print(f" Results: {EXP_DIR}")
    for name in DATASETS:
        print(f"   {EXP_DIR / name}/")
    summary_file = EXP_DIR / "cross_dataset_summary.json"
    print(f" Summary: {summary_file}")
    print(sep)


if __name__ == "__main__":
    main()
