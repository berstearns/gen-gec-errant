# Pipeline Explanation

This document explains each step of the L2 Learner Error Analysis Pipeline as run on the CELVA-SP dataset with vanilla HuggingFace GPT-2.

## Overview

The pipeline takes L2 learner essays, splits them into sentences, generates text continuations using GPT-2, corrects those continuations with a GEC model, then annotates and analyzes the grammatical errors.

```
Essays (CSV) → Sentences → Sample → GPT-2 Generation → GEC Correction → ERRANT Annotation → Analysis + Plots
```

---

## Step 1: Environment Setup

Created a virtual environment using pyenv Python 3.10.18 and installed all dependencies via `uv`. The project requires PyTorch, Transformers, ERRANT (for error annotation), spaCy (for sentence splitting and ERRANT), and various analysis libraries.

A `[tool.setuptools] py-modules` section was added to `pyproject.toml` to fix a flat-layout build error (multiple top-level modules without a package directory).

## Step 2: Preprocess Essays into Sentences

The input CSV (`norm-CELVA-SP.csv`) contains full essays with columns: `writing_id`, `l1`, `cefr_level`, `text`. The preprocessing script (`preprocess_efcamdat.py`):

1. Reads the CSV and auto-detects columns by header name
2. Splits each essay into individual sentences using spaCy (`en_core_web_sm`)
3. Filters sentences by word count (5-60 words by default)
4. Outputs a clean CSV with one sentence per row plus metadata (essay_id, CEFR level, L1, etc.)

- **Input:** `/home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits/norm-CELVA-SP.csv` (1,742 essays, cols: `writing_id`, `l1`, `cefr_level`, `text`)
- **Output:** `data/efcamdat_sentences.csv` (21,493 sentences, cols: `essay_id`, `sentence_idx`, `sentence`, `cefr_level`, `l1_language`, `topic`, `word_count`)

**Result:** 1,742 essays → 21,493 sentences. All L1=French, CEFR distribution: A1(1241), A2(5453), B1(7743), B2(5103), C1(1789), C2(164).

## Step 3: Sample 5,000 Sentences

Randomly samples 5,000 sentences (seed=42) from the full 21,493 to keep the pipeline run manageable. This is the working dataset for generation.

- **Input:** `data/efcamdat_sentences.csv` (21,493 rows)
- **Output:** `data/efcamdat_sentences_5k.csv` (5,000 rows, same columns)

## Step 4: Download GPT-2 Locally

Downloads the HuggingFace `gpt2` model (124M params) and tokenizer to `models/gpt2/` for local reuse. The pipeline also caches models via HuggingFace Hub, so this step ensures you have a local copy independent of the cache.

- **Input:** HuggingFace Hub model ID `gpt2`
- **Output:** `models/gpt2/` (contains `config.json`, `model.safetensors`, `tokenizer.json`, `vocab.json`, `merges.txt`, etc.)

## Step 5: Run the Pipeline (`run_pipeline.py`)

The pipeline script orchestrates 5 sub-steps:

### 5a. Load Data (Step 1 in pipeline)

Reads the sampled sentences, extracts the `sentence` column, and constructs prompts. Each sentence is split at ~50% of its words to create a prompt prefix — the model must generate the continuation.

- **Input:** `data/efcamdat_sentences_5k.csv`
- **Output:** `outputs/prompts.json` (20 prompt objects with `prompt`, `reference`, `source_sentence`, `cefr_level`)

### 5b. Generate Continuations (Step 2 in pipeline)

For each prompt, GPT-2 generates a continuation of up to 50 new tokens using sampling (temperature=1.0, top_k=50, top_p=0.95, repetition_penalty=1.2). Perplexity is computed for each generated text.

- **Input:** `outputs/prompts.json` (20 prompts)
- **Model used:** `gpt2-base` (vanilla HuggingFace GPT-2, 124M params)
- **Device:** CPU (no CUDA available)
- **Intermediate output:** generated text + perplexity scores stored in `outputs/raw_results.json`
- **Result:** 20 sentences processed, mean perplexity = 47.86 (std=14.73)

### 5c. Grammatical Error Correction (Step 3 in pipeline)

Each generated continuation is corrected using `grammarly/coedit-large`, a dedicated GEC model based on T5-large fine-tuned on grammar correction tasks. The prompt format is:

> "Fix the grammar: {generated_text}"

This produces a "corrected" version of each generation.

- **Input:** 20 generated texts from step 5b
- **GEC model:** `grammarly/coedit-large` (downloaded from HuggingFace Hub on first run, ~1.2GB)
- **Intermediate output:** corrected texts appended to each result in `outputs/raw_results.json`

### 5d. ERRANT Error Annotation (Step 4 in pipeline)

ERRANT (ERRor ANnotation Toolkit) aligns the original generated text with the GEC-corrected version and classifies each edit into typed error categories:

- **R:NOUN** — noun replacement (16 occurrences)
- **R:OTHER** — other replacements (15)
- **R:ORTH** — orthography/capitalization (7)
- **U:PREP** — unnecessary preposition (2)
- **R:PREP** — preposition replacement (2)

- **Input:** 20 (original, corrected) text pairs from step 5c
- **Output:** `outputs/errors_long_format.tsv` — 60 individual error rows with columns: `model`, `sentence_idx`, `error_type`, `original_token`, `corrected_token`, `start_offset`, `end_offset`

**Result:** 60 total errors across 20 sentences (error rate: 0.95, avg 3.0 errors/sentence).

### 5e. Analysis & Visualization (Step 5 in pipeline)

Computes summary statistics per model and generates all final outputs.

- **Input:** all intermediate results from steps 5a-5d
- **Outputs (all under `outputs/`):**

| File | Description | Size |
|------|-------------|------|
| `full_results.tsv` | Full results table (20 rows × 14 cols: prompt, generation, correction, perplexity, error count, etc.) | 28K |
| `errors_long_format.tsv` | One row per error annotation (60 rows) | 32K |
| `gpt2-base_summary.json` | Per-model summary (mean PPL, error rate, top errors) | 2.5K |
| `model_comparison.json` | Cross-model comparison table | 217B |
| `raw_results.json` | All raw data (prompts, generations, corrections, errors, perplexity) | 33K |
| `prompts.json` | Input prompts used for generation | 5.4K |
| `plots/perplexity_comparison.png` | Bar chart of mean perplexity per model | 30K |
| `plots/error_comparison.png` | Bar chart of error rate per model | 54K |
| `plots/error_type_breakdown.png` | Stacked bar chart of ERRANT error types | 62K |
| `plots/ppl_vs_errors_scatter.png` | Scatter plot: perplexity vs error count per sentence | 45K |
| `plots/combined_metric.png` | Combined PPL × error rate metric | 38K |

---

# Raw Commands

All commands are run from the project root directory.

```bash
# ── 1. Extract the zip ──
unzip -o artificial-learner-pipeline.zip -x '__MACOSX/*'

# ── 2. Create directories ──
mkdir -p data models/gpt2 outputs

# ── 3. Create venv with Python 3.10.18 ──
uv venv --python ~/.pyenv/versions/3.10.18/bin/python

# ── 4. Install project and all dependencies ──
uv pip install -e .

# ── 5. Install pip (needed for spaCy download) and download spaCy model ──
uv pip install pip
uv run python -m spacy download en_core_web_sm

# ── 6. Preprocess essays into sentences ──
uv run python preprocess_efcamdat.py \
    --input /home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits/norm-CELVA-SP.csv \
    --output data/efcamdat_sentences.csv \
    --text_col text

# ── 7. Sample 5,000 sentences ──
uv run python -c "
import csv, random
random.seed(42)
with open('data/efcamdat_sentences.csv') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
sample = random.sample(rows, min(5000, len(rows)))
with open('data/efcamdat_sentences_5k.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(sample)
print(f'Sampled {len(sample)} sentences')
"

# ── 8. Download GPT-2 locally ──
uv run python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
model_id = 'gpt2'
save_path = 'models/gpt2'
AutoTokenizer.from_pretrained(model_id).save_pretrained(save_path)
AutoModelForCausalLM.from_pretrained(model_id).save_pretrained(save_path)
print(f'Saved to {save_path}')
"

# ── 9. Run the full pipeline ──
uv run python run_pipeline.py \
    --data_path data/efcamdat_sentences_5k.csv \
    --models gpt2-base \
    --max_sentences 20 \
    --output_dir outputs/
```
