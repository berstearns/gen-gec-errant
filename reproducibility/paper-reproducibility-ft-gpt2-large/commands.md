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
rclone copy "i:/_p/artificial-learners/models/gpt2/gpt2-large-all-data-20260222-092036/final" \
    "/home/b/p/my-data/i/_p/artificial-learners/models/2026-02-20-model/gpt2-large-all-data-20260222-092036/final" \
    --progress
```

Verify:
```bash
ls /home/b/p/my-data/i/_p/artificial-learners/models/2026-02-20-model/gpt2-large-all-data-20260222-092036/final/config.json
```

## One-command run (recommended)

```bash
.venv/bin/python reproducibility/paper-reproducibility-ft-gpt2-large/scripts/run_experiment.py
```

This downloads the model (if needed), then runs all datasets automatically.

## Manual step-by-step

### Run pipeline for a single dataset

```bash
.venv/bin/python -m gen_gec_errant.pipeline \
    --config reproducibility/paper-reproducibility-ft-gpt2-large/experiment/configs/norm-CELVA-SP.yaml
```

### Run with GPU and larger batches

```bash
.venv/bin/python reproducibility/paper-reproducibility-ft-gpt2-large/scripts/run_experiment.py
# Or override per-dataset:
.venv/bin/python -m gen_gec_errant.pipeline \
    --config reproducibility/paper-reproducibility-ft-gpt2-large/experiment/configs/norm-CELVA-SP.yaml \
    --device cuda \
    --batch_size 8 \
    data_loader.max_sentences=500
```

## Remove limits for full paper reproduction

In `scripts/run_experiment.py`, set all values in `LIMITS` to `None`.
