# Input/Output Documentation for Each Step

## Design Decisions

### Sentence Limits for CPU Runs

Full generation + GEC on CPU is slow (GPT-2 generation + coedit-large correction).
For quick end-to-end pipeline validation, the following limits are applied via `max_sentences`:

| Dataset | Full rows | Limited to | Reason |
|---|---|---|---|
| norm-CELVA-SP | 1,742 | 50 | Primary dataset, medium size |
| norm-EFCAMDAT-test | 20,000 | 50 | Large; full run needs GPU for speed |
| norm-KUPA-KEYS | 1,006 | 50 | Small, but limit for quick validation |

To reproduce full paper results: set all limits to `None` in `scripts/run_experiment.py`.

### Data Splitting

Each sentence is split at `prompt_ratio=0.5`:
- First half = **prompt** (given to GPT-2 as context)
- Second half = **reference continuation** (the learner's actual text)
- GPT-2 generates its own continuation from the prompt
- Comparison: GPT-2 continuation errors vs. learner continuation errors

### Word Count Filtering

Sentences are filtered by word count: `min_words=10`, `max_words=500`.
This removes very short fragments and excessively long essays.

### GEC Model

Default: `grammarly/coedit-large` (T5-large, ~770M params).
This is a dedicated grammar correction model, not an LLM-based approach.
It provides consistent, focused grammatical corrections.

### Generation Parameters

- `temperature=1.0`: Standard sampling (not greedy)
- `top_k=50`, `top_p=0.95`: Nucleus sampling
- `repetition_penalty=1.2`: Avoid degenerate repetition
- `max_new_tokens=50`: Generate up to 50 new tokens
- `min_new_tokens=10`: At least 10 tokens per continuation

### Python Environment

Uses the project's `.venv` virtual environment.
The `sys.executable` in `run_experiment.py` ensures child processes use the same interpreter.

---

## Step 0: Setup

**Input:**
- Source CSVs from `/home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits/`
  - `norm-CELVA-SP.csv` (1,742 rows, schema: writing_id,l1,cefr_level,text)
  - `norm-EFCAMDAT-test.csv` (20,000 rows, same schema)
  - `norm-KUPA-KEYS.csv` (1,006 rows, same schema)

**Output:**
- `experiment/configs/{dataset_name}.yaml` (per-dataset pipeline YAML config)
- Verification that source data exists

---

## Steps 1-4: Pipeline Execution (per dataset)

Runs `python -m gen_gec_errant.pipeline --config <yaml>` which internally executes:

### Step 1: Data Loading

**Input:** Source CSV
**Output:** `experiment/{dataset_name}/prompts.json`

prompts.json schema:
```json
[
  {"prompt": "The student was trying to", "reference": "explain the problem clearly", "full": "The student was trying to explain the problem clearly"},
  ...
]
```

### Step 2: Generation

**Input:** Prompts from Step 1 + pre-trained GPT-2 model (auto-downloaded from HuggingFace Hub)
**Output:** In-memory results dict with:
- `continuations`: List[str] - GPT-2 generated continuations
- `full_texts`: List[str] - prompt + continuation
- `perplexities`: List[float] - per-sentence perplexity
- `prompt_boundaries`: List[int] - char offset where prompt ends

Also adds `learner_baseline` pseudo-model with reference continuations.

### Step 3: GEC

**Input:** Generated texts + `grammarly/coedit-large` model (auto-downloaded, ~3GB)
**Output:** Added to results dict:
- `corrected_continuations`: List[str] - GEC-corrected continuations
- `corrected_full_texts`: List[str] - GEC-corrected full texts

### Step 4: ERRANT Annotation

**Input:** Original vs. corrected texts
**Output:** Added to results dict:
- `annotations`: List[SentenceAnnotation] - continuation-level errors
- `full_text_annotations`: List[SentenceAnnotation] - full-text errors with region tags
- `error_summary`: Dict - aggregate error statistics
- `region_error_summary`: Dict - errors split by prompt vs. generation region

SentenceAnnotation contains:
- `original`, `corrected`: The two texts compared
- `errors`: List of ErrorAnnotation (original_tokens, corrected_tokens, error_type, region)
- `num_errors`, `error_type_counts`
- `prompt_error_count`, `generation_error_count`

---

## Step 5: Analysis & Export

**Input:** All results from Steps 1-4

**Output per dataset:** `experiment/{dataset_name}/`

1. **raw_results.json** - Complete pipeline output (all models, all sentences):
   ```json
   {
     "gpt2-small": {
       "continuations": [...],
       "full_texts": [...],
       "corrected_continuations": [...],
       "corrected_full_texts": [...],
       "perplexities": [...],
       "annotations": [...],
       "full_text_annotations": [...],
       "error_summary": {...},
       "region_error_summary": {...}
     },
     "learner_baseline": { ... }
   }
   ```

2. **gpt2-small_summary.json** - Per-model metrics:
   ```json
   {
     "ppl_mean": 45.2,
     "ppl_median": 32.1,
     "total_errors": 42,
     "avg_errors_per_sentence": 2.1,
     "error_rate": 0.84,
     "top_10_error_types": [["M:DET", 12], ["R:VERB:SVA", 8], ...],
     "ppl_x_errors": 95.0
   }
   ```

3. **model_comparison.json** - Cross-model comparison (gpt2-small vs learner_baseline):
   - Pairwise Mann-Whitney U tests for perplexity and error counts
   - Side-by-side metric comparison

4. **full_results.tsv** - Flat CSV, 1 row per sentence:
   - Columns: sentence_id, prompt, reference_continuation, full_original
   - Per model: continuation, full_text, corrected_*, perplexity, num_errors, error_types, etc.

5. **errors_long_format.tsv** - Long format, 1 row per error:
   - sentence_id, model, source, original_text, corrected_text
   - error_original_tokens, error_corrected_tokens, error_type
   - error_operation (M/R/U/T), error_subcategory (DET/VERB/etc.)
   - region (prompt/generation)

6. **plots/** (if skip_plots=False):
   - perplexity_comparison.png
   - error_comparison.png
   - error_type_breakdown.png
   - ppl_vs_errors_scatter.png

---

## Step 6: Cross-Dataset Summary

**Input:** All per-dataset results
**Output:** `experiment/cross_dataset_summary.json`

Aggregates key metrics across datasets:
- Per-dataset: mean PPL, error rate, top error types
- Comparison table for quick reference

---

## Directory Structure After Complete Run

```
reproducibility/paper-reproducibility-gpt2-native-zero-shot/
├── plan/
│   ├── overview.md
│   └── steps.md
├── scripts/
│   └── run_experiment.py
├── experiment/
│   ├── configs/                                # Generated YAML configs
│   │   ├── norm-CELVA-SP.yaml
│   │   ├── norm-EFCAMDAT-test.yaml
│   │   └── norm-KUPA-KEYS.yaml
│   ├── norm-CELVA-SP/                          # Per-dataset outputs
│   │   ├── prompts.json
│   │   ├── raw_results.json
│   │   ├── gpt2-small_summary.json
│   │   ├── learner_baseline_summary.json
│   │   ├── model_comparison.json
│   │   ├── full_results.tsv
│   │   ├── errors_long_format.tsv
│   │   └── plots/
│   │       ├── perplexity_comparison.png
│   │       ├── error_comparison.png
│   │       └── ...
│   ├── norm-EFCAMDAT-test/
│   │   └── (same structure)
│   ├── norm-KUPA-KEYS/
│   │   └── (same structure)
│   └── cross_dataset_summary.json              # Aggregated summary
├── IO.md
└── commands.md
```
