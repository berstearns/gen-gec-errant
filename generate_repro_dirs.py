#!/usr/bin/env python3
"""
Generate paper-reproducibility directories for all trained NWP artificial learner models.

Creates one self-contained directory per model with:
  - plan/overview.md, plan/steps.md
  - scripts/run_experiment.py
  - IO.md, commands.md

Models: 3 GPT-2 + 5 Pythia + 3 SmolLM2 = 11 fine-tuned models on EFCAMDAT all-data.

Usage:
    python generate_repro_dirs.py
"""

import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_MODELS_BASE = Path("/home/b/p/my-data/i/_p/artificial-learners/models")

# ── Model Registry ────────────────────────────────────────────────────────
# Each model maps to its rclone source, local path, base HF ID, and family.
# The `final/` subdirectory of each contains the ready-to-use HF checkpoint.

MODELS = {
    # ── GPT-2 ──
    "ft-gpt2-small": {
        "params": "124M",
        "base_hf_id": "gpt2",
        "model_family": "gpt2",
        "rclone_dir": "i:/_p/artificial-learners/models/2026-02-23-model/gpt2-small-all-data-resume",
        "local_dir": LOCAL_MODELS_BASE / "2026-02-23-model" / "gpt2-small-all-data-resume",
        "description": "GPT-2 Small (124M) fine-tuned on EFCAMDAT all-data",
    },
    "ft-gpt2-medium": {
        "params": "355M",
        "base_hf_id": "gpt2-medium",
        "model_family": "gpt2",
        "rclone_dir": "i:/_p/artificial-learners/models/gpt2/gpt2-medium-all-data-resume",
        "local_dir": LOCAL_MODELS_BASE / "2026-02-23-model" / "gpt2-medium-all-data-resume",
        "description": "GPT-2 Medium (355M) fine-tuned on EFCAMDAT all-data",
    },
    "ft-gpt2-large": {
        "params": "774M",
        "base_hf_id": "gpt2-large",
        "model_family": "gpt2",
        "rclone_dir": "i:/_p/artificial-learners/models/gpt2/gpt2-large-all-data-20260222-092036",
        "local_dir": LOCAL_MODELS_BASE / "2026-02-20-model" / "gpt2-large-all-data-20260222-092036",
        "description": "GPT-2 Large (774M) fine-tuned on EFCAMDAT all-data",
    },
    # ── Pythia ──
    "ft-pythia-70m": {
        "params": "70M",
        "base_hf_id": "EleutherAI/pythia-70m",
        "model_family": "pythia",
        "rclone_dir": "i:/_p/artificial-learners/models/pythia/pythia-70m-all-data",
        "local_dir": LOCAL_MODELS_BASE / "pythia" / "pythia-70m-all-data",
        "description": "Pythia 70M fine-tuned on EFCAMDAT all-data",
    },
    "ft-pythia-160m": {
        "params": "160M",
        "base_hf_id": "EleutherAI/pythia-160m",
        "model_family": "pythia",
        "rclone_dir": "i:/_p/artificial-learners/models/pythia/pythia-160m-all-data",
        "local_dir": LOCAL_MODELS_BASE / "pythia" / "pythia-160m-all-data",
        "description": "Pythia 160M fine-tuned on EFCAMDAT all-data",
    },
    "ft-pythia-410m": {
        "params": "410M",
        "base_hf_id": "EleutherAI/pythia-410m",
        "model_family": "pythia",
        "rclone_dir": "i:/_p/artificial-learners/models/pythia/pythia-410m-all-data",
        "local_dir": LOCAL_MODELS_BASE / "pythia" / "pythia-410m-all-data",
        "description": "Pythia 410M fine-tuned on EFCAMDAT all-data",
    },
    "ft-pythia-1b": {
        "params": "1B",
        "base_hf_id": "EleutherAI/pythia-1b",
        "model_family": "pythia",
        "rclone_dir": "i:/_p/artificial-learners/models/pythia/pythia-1b-all-data",
        "local_dir": LOCAL_MODELS_BASE / "pythia" / "pythia-1b-all-data",
        "description": "Pythia 1B fine-tuned on EFCAMDAT all-data",
    },
    "ft-pythia-1.4b": {
        "params": "1.4B",
        "base_hf_id": "EleutherAI/pythia-1.4b",
        "model_family": "pythia",
        "rclone_dir": "i:/_p/artificial-learners/models/pythia/pythia-1.4b-all-data",
        "local_dir": LOCAL_MODELS_BASE / "pythia" / "pythia-1.4b-all-data",
        "description": "Pythia 1.4B fine-tuned on EFCAMDAT all-data",
    },
    # ── SmolLM2 ──
    "ft-smollm2-135m": {
        "params": "135M",
        "base_hf_id": "HuggingFaceTB/SmolLM2-135M",
        "model_family": "smollm2",
        "rclone_dir": "i:/_p/artificial-learners/models/smollm2/smollm2-135m-all-data",
        "local_dir": LOCAL_MODELS_BASE / "smollm2" / "smollm2-135m-all-data",
        "description": "SmolLM2 135M fine-tuned on EFCAMDAT all-data",
    },
    "ft-smollm2-360m": {
        "params": "360M",
        "base_hf_id": "HuggingFaceTB/SmolLM2-360M",
        "model_family": "smollm2",
        "rclone_dir": "i:/_p/artificial-learners/models/smollm2/smollm2-360m-all-data",
        "local_dir": LOCAL_MODELS_BASE / "smollm2" / "smollm2-360m-all-data",
        "description": "SmolLM2 360M fine-tuned on EFCAMDAT all-data",
    },
    "ft-smollm2-1.7b": {
        "params": "1.7B",
        "base_hf_id": "HuggingFaceTB/SmolLM2-1.7B",
        "model_family": "smollm2",
        "rclone_dir": "i:/_p/artificial-learners/models/smollm2/smollm2-1.7b-all-data",
        "local_dir": LOCAL_MODELS_BASE / "smollm2" / "smollm2-1.7b-all-data",
        "description": "SmolLM2 1.7B fine-tuned on EFCAMDAT all-data",
    },
}


def gen_overview(model_name, info):
    return f"""\
# {info['description']} — Error Profile Experiment

## Goal

Run the full generate-GEC-ERRANT-analysis pipeline using **{model_name}**
({info['description']}) on learner corpus data.
This model is a **fine-tuned artificial learner**: pre-trained {info['base_hf_id']}
further trained on EFCAMDAT all-data (English learner texts across all CEFR levels).

Compare its error profile against the native (pre-trained) baseline and learner
reference continuations to evaluate how well it reproduces learner-like errors.

## Pipeline

1. Download model weights from Google Drive (via rclone) if not locally available
2. Load learner text data, split into prompts + reference continuations
3. Generate continuations with {model_name} (fine-tuned on EFCAMDAT)
4. Run grammatical error correction (GEC) on generated text via coedit-large
5. Annotate errors with ERRANT (automatic error type classification)
6. Analyze: per-model summaries, CSV exports, visualizations

## Datasets

Source: `/home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits/`

| Dataset | File | Rows | Description |
|---|---|---|---|
| CELVA-SP | norm-CELVA-SP.csv | 1,742 | Spanish L1 learner English (primary) |
| EFCAMDAT-test | norm-EFCAMDAT-test.csv | 20,000 | In-domain learner English |
| KUPA-KEYS | norm-KUPA-KEYS.csv | 1,006 | Cross-corpus learner English |

## Model

- **{model_name}** ({info['params']} parameters)
- Base: `{info['base_hf_id']}` (pre-trained)
- Fine-tuned on: EFCAMDAT all-data (full learner corpus, all CEFR levels)
- Architecture family: {info['model_family']}
- rclone source: `{info['rclone_dir']}`
- Local path: `{info['local_dir']}/final`

## Constraints

- Uses the `gen_gec_errant` package from this repository (does NOT modify src/)
- All outputs go into `experiment/` subdirectory
- Reproducible via fixed seed (42)
- Model weights downloaded from Google Drive via rclone (only `final/` checkpoint)
"""


def gen_steps(model_name, info):
    return f"""\
# Execution Steps

## Step 0: Download model weights
- Check if `{info['local_dir']}/final` exists locally
- If not: download from `{info['rclone_dir']}/final` via rclone
- Verify that `config.json` and `model.safetensors` (or `model.bin`) are present

## Step 1: Setup experiment directory
- Create `experiment/` subdirectory structure
- Verify source data CSVs exist
- Write per-dataset YAML configs into `experiment/configs/`

## Step 2: Run pipeline per dataset
- For each dataset (CELVA-SP, EFCAMDAT-test, KUPA-KEYS):
  - Load CSV, filter by word count, split into (prompt, reference) pairs
  - Generate continuations with {model_name} from local checkpoint
  - Run GEC correction with coedit-large
  - Annotate errors with ERRANT
  - Compute summaries, export CSVs and plots

## Step 3: Cross-dataset summary
- Aggregate results across all datasets
- Produce a single summary comparing error profiles by dataset
"""


def gen_io(model_name, info):
    return f"""\
# Input/Output Documentation

## Design Decisions

### Model Weights

Model: **{model_name}** ({info['params']})
- rclone source: `{info['rclone_dir']}/final`
- Local path: `{info['local_dir']}/final`
- Only the `final/` checkpoint is downloaded (not intermediate checkpoints)
- The `final/` directory contains standard HuggingFace format:
  `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`,
  `generation_config.json`

### Sentence Limits for CPU Runs

| Dataset | Full rows | Limited to | Reason |
|---|---|---|---|
| norm-CELVA-SP | 1,742 | 50 | Primary dataset, medium size |
| norm-EFCAMDAT-test | 20,000 | 50 | Large; full run needs GPU for speed |
| norm-KUPA-KEYS | 1,006 | 50 | Small, but limit for quick validation |

To reproduce full paper results: set all limits to `None` in `scripts/run_experiment.py`.

### Data Splitting

Each sentence is split at `prompt_ratio=0.5`:
- First half = **prompt** (given to the model as context)
- Second half = **reference continuation** (the learner's actual text)
- The model generates its own continuation from the prompt

### GEC Model

Default: `grammarly/coedit-large` (T5-large, ~770M params).

### Generation Parameters

- `temperature=1.0`, `top_k=50`, `top_p=0.95` (nucleus sampling)
- `repetition_penalty=1.2`, `max_new_tokens=50`, `min_new_tokens=10`

---

## Step 0: Download Model

**Input:** rclone remote `{info['rclone_dir']}/final`
**Output:** `{info['local_dir']}/final/` containing HuggingFace model files

**Command:**
```bash
rclone copy "{info['rclone_dir']}/final" "{info['local_dir']}/final" --progress
```

---

## Steps 1-2: Pipeline Execution (per dataset)

Runs `python -m gen_gec_errant.pipeline --config <yaml>` which internally executes
data loading, generation, GEC, ERRANT annotation, and analysis.

**Output per dataset:** `experiment/{{dataset_name}}/`
- `prompts.json` — Input data
- `raw_results.json` — Complete pipeline output
- `{model_name}_summary.json` — Per-model metrics
- `learner_baseline_summary.json` — Reference learner metrics
- `model_comparison.json` — Cross-model comparison
- `full_results.tsv` — 1 row per sentence
- `errors_long_format.tsv` — 1 row per error
- `plots/` — Visualizations

---

## Step 3: Cross-Dataset Summary

**Output:** `experiment/cross_dataset_summary.json`

---

## Directory Structure After Complete Run

```
reproducibility/paper-reproducibility-{model_name}/
├── plan/
│   ├── overview.md
│   └── steps.md
├── scripts/
│   └── run_experiment.py
├── experiment/
│   ├── configs/
│   │   ├── norm-CELVA-SP.yaml
│   │   ├── norm-EFCAMDAT-test.yaml
│   │   └── norm-KUPA-KEYS.yaml
│   ├── norm-CELVA-SP/
│   │   ├── prompts.json
│   │   ├── raw_results.json
│   │   ├── {model_name}_summary.json
│   │   ├── learner_baseline_summary.json
│   │   ├── model_comparison.json
│   │   ├── full_results.tsv
│   │   ├── errors_long_format.tsv
│   │   └── plots/
│   ├── norm-EFCAMDAT-test/
│   │   └── (same structure)
│   ├── norm-KUPA-KEYS/
│   │   └── (same structure)
│   └── cross_dataset_summary.json
├── IO.md
└── commands.md
```
"""


def gen_commands(model_name, info):
    return f"""\
# Tangible Commands

All commands run from repo root: `cd /home/b/p/research-sketches/automatic-generation-correction-errortagging-v2`

## Prerequisites

- Python 3.10+ with the project's `.venv` activated
- rclone configured with `i:` remote (Google Drive)
- Required packages: torch, transformers, errant, spacy, pandas, scipy, matplotlib, pyyaml
- spaCy model: `python -m spacy download en_core_web_sm`
- GEC model (coedit-large) is auto-downloaded from HuggingFace Hub (~3GB)
- Package installed: `pip install -e .`

## Step 0: Download model weights

```bash
rclone copy "{info['rclone_dir']}/final" \\
    "{info['local_dir']}/final" \\
    --progress
```

Verify:
```bash
ls {info['local_dir']}/final/config.json
```

## One-command run (recommended)

```bash
.venv/bin/python reproducibility/paper-reproducibility-{model_name}/scripts/run_experiment.py
```

This downloads the model (if needed), then runs all datasets automatically.

## Manual step-by-step

### Run pipeline for a single dataset

```bash
.venv/bin/python -m gen_gec_errant.pipeline \\
    --config reproducibility/paper-reproducibility-{model_name}/experiment/configs/norm-CELVA-SP.yaml
```

### Run with GPU and larger batches

```bash
.venv/bin/python reproducibility/paper-reproducibility-{model_name}/scripts/run_experiment.py
# Or override per-dataset:
.venv/bin/python -m gen_gec_errant.pipeline \\
    --config reproducibility/paper-reproducibility-{model_name}/experiment/configs/norm-CELVA-SP.yaml \\
    --device cuda \\
    --batch_size 8 \\
    data_loader.max_sentences=500
```

## Remove limits for full paper reproduction

In `scripts/run_experiment.py`, set all values in `LIMITS` to `None`.
"""


def gen_run_experiment(model_name, info):
    # Use placeholder tokens to avoid nested f-string escaping issues
    template = r'''#!/usr/bin/env python3
"""
End-to-end __DESCRIPTION__ Error Profile Experiment.

Runs the full generate -> GEC -> ERRANT -> analysis pipeline using
__MODEL_NAME__ (fine-tuned on EFCAMDAT all-data) on multiple learner datasets.

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
MODEL_NAME = "__MODEL_NAME__"
HF_MODEL_ID = "__LOCAL_DIR__/final"
BASE_HF_ID = "__BASE_HF_ID__"
MODEL_FAMILY = "__MODEL_FAMILY__"
MODEL_PARAMS = "__PARAMS__"
RCLONE_DIR = "__RCLONE_DIR__"
LOCAL_DIR = Path("__LOCAL_DIR__")
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
'''

    return (
        template
        .replace("__MODEL_NAME__", model_name)
        .replace("__DESCRIPTION__", info["description"])
        .replace("__LOCAL_DIR__", str(info["local_dir"]))
        .replace("__BASE_HF_ID__", info["base_hf_id"])
        .replace("__MODEL_FAMILY__", info["model_family"])
        .replace("__PARAMS__", info["params"])
        .replace("__RCLONE_DIR__", info["rclone_dir"])
    )


def create_repro_dir(model_name, info):
    """Create a complete paper-reproducibility directory for one model."""
    repro_dir = PROJECT_ROOT / "reproducibility" / f"paper-reproducibility-{model_name}"

    # Create directories
    (repro_dir / "plan").mkdir(parents=True, exist_ok=True)
    (repro_dir / "scripts").mkdir(parents=True, exist_ok=True)

    # Write files
    (repro_dir / "plan" / "overview.md").write_text(gen_overview(model_name, info))
    (repro_dir / "plan" / "steps.md").write_text(gen_steps(model_name, info))
    (repro_dir / "IO.md").write_text(gen_io(model_name, info))
    (repro_dir / "commands.md").write_text(gen_commands(model_name, info))

    script_path = repro_dir / "scripts" / "run_experiment.py"
    script_path.write_text(gen_run_experiment(model_name, info))
    script_path.chmod(0o755)

    return repro_dir


def main():
    print("=" * 70)
    print(" Generating paper-reproducibility directories")
    print(f" Project root: {PROJECT_ROOT}")
    print(f" Models: {len(MODELS)}")
    print("=" * 70)

    for model_name, info in MODELS.items():
        repro_dir = create_repro_dir(model_name, info)
        print(f"  Created: {repro_dir.name}/")

    print()
    print(f"Done. {len(MODELS)} directories created.")
    print()
    print("To run a single model:")
    print("  .venv/bin/python reproducibility/paper-reproducibility-ft-gpt2-small/scripts/run_experiment.py")
    print()
    print("To run all models:")
    print("  for d in reproducibility/paper-reproducibility-ft-*/; do")
    print('    .venv/bin/python "$d/scripts/run_experiment.py"')
    print("  done")


if __name__ == "__main__":
    main()
