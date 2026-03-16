# Input/Output Documentation

## Design Decisions

### Model Weights

Model: **ft-smollm2-135m** (135M)
- rclone source: `i:/_p/artificial-learners/models/smollm2/smollm2-135m-all-data/final`
- Local path: `/home/b/p/my-data/i/_p/artificial-learners/models/smollm2/smollm2-135m-all-data/final`
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

**Input:** rclone remote `i:/_p/artificial-learners/models/smollm2/smollm2-135m-all-data/final`
**Output:** `/home/b/p/my-data/i/_p/artificial-learners/models/smollm2/smollm2-135m-all-data/final/` containing HuggingFace model files

**Command:**
```bash
rclone copy "i:/_p/artificial-learners/models/smollm2/smollm2-135m-all-data/final" "/home/b/p/my-data/i/_p/artificial-learners/models/smollm2/smollm2-135m-all-data/final" --progress
```

---

## Steps 1-2: Pipeline Execution (per dataset)

Runs `python -m gen_gec_errant.pipeline --config <yaml>` which internally executes
data loading, generation, GEC, ERRANT annotation, and analysis.

**Output per dataset:** `experiment/{dataset_name}/`
- `prompts.json` — Input data
- `raw_results.json` — Complete pipeline output
- `ft-smollm2-135m_summary.json` — Per-model metrics
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
reproducibility/paper-reproducibility-ft-smollm2-135m/
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
│   │   ├── ft-smollm2-135m_summary.json
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
