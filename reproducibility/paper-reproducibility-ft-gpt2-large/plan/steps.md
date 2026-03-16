# Execution Steps

## Step 0: Download model weights
- Check if `/home/b/p/my-data/i/_p/artificial-learners/models/2026-02-20-model/gpt2-large-all-data-20260222-092036/final` exists locally
- If not: download from `i:/_p/artificial-learners/models/gpt2/gpt2-large-all-data-20260222-092036/final` via rclone
- Verify that `config.json` and `model.safetensors` (or `model.bin`) are present

## Step 1: Setup experiment directory
- Create `experiment/` subdirectory structure
- Verify source data CSVs exist
- Write per-dataset YAML configs into `experiment/configs/`

## Step 2: Run pipeline per dataset
- For each dataset (CELVA-SP, EFCAMDAT-test, KUPA-KEYS):
  - Load CSV, filter by word count, split into (prompt, reference) pairs
  - Generate continuations with ft-gpt2-large from local checkpoint
  - Run GEC correction with coedit-large
  - Annotate errors with ERRANT
  - Compute summaries, export CSVs and plots

## Step 3: Cross-dataset summary
- Aggregate results across all datasets
- Produce a single summary comparing error profiles by dataset
