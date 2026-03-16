# Tangible Commands

All commands run from repo root: `cd /home/b/p/research-sketches/automatic-generation-correction-errortagging-v2`

## Prerequisites

- Python 3.10+ with the project's `.venv` activated
- Required packages: torch, transformers, errant, spacy, pandas, scipy, matplotlib, pyyaml
- spaCy model: `python -m spacy download en_core_web_sm`
- GPT-2 model is auto-downloaded from HuggingFace Hub (~550MB)
- GEC model (coedit-large) is auto-downloaded from HuggingFace Hub (~3GB)
- Package installed: `pip install -e .`

## One-command run (recommended)

```bash
.venv/bin/python reproducibility/paper-reproducibility-gpt2-native-zero-shot/scripts/run_experiment.py
```

This orchestrates all steps automatically with error handling and resumable runs.

## Manual step-by-step commands

### Run pipeline for a single dataset (CELVA-SP)

```bash
.venv/bin/python -m gen_gec_errant.pipeline \
    --config reproducibility/paper-reproducibility-gpt2-native-zero-shot/experiment/configs/norm-CELVA-SP.yaml
```

### Run pipeline for all datasets

```bash
for DATASET in norm-CELVA-SP norm-EFCAMDAT-test norm-KUPA-KEYS; do
    .venv/bin/python -m gen_gec_errant.pipeline \
        --config reproducibility/paper-reproducibility-gpt2-native-zero-shot/experiment/configs/${DATASET}.yaml
done
```

### Run with custom overrides

```bash
.venv/bin/python -m gen_gec_errant.pipeline \
    --config reproducibility/paper-reproducibility-gpt2-native-zero-shot/experiment/configs/norm-CELVA-SP.yaml \
    --device cuda \
    --batch_size 8 \
    data_loader.max_sentences=100
```

### Skip generation (re-run GEC + annotation + analysis on existing results)

```bash
.venv/bin/python -m gen_gec_errant.pipeline \
    --config reproducibility/paper-reproducibility-gpt2-native-zero-shot/experiment/configs/norm-CELVA-SP.yaml \
    --skip_generation
```

### Skip GEC (re-run annotation + analysis only)

```bash
.venv/bin/python -m gen_gec_errant.pipeline \
    --config reproducibility/paper-reproducibility-gpt2-native-zero-shot/experiment/configs/norm-CELVA-SP.yaml \
    --skip_gec
```

## Remove limits for full paper reproduction

In `scripts/run_experiment.py`, set all values in `LIMITS` to `None`:

```python
LIMITS = {
    "norm-CELVA-SP":       None,  # was 50
    "norm-EFCAMDAT-test":  None,  # was 50
    "norm-KUPA-KEYS":      None,  # was 50
}
```

Warning: Full generation + GEC on CPU is very slow. Use `--device cuda` if a GPU is available.
Estimated times (CPU): ~30min per 100 sentences. Full EFCAMDAT-test (20k) would take many hours.
With GPU: ~2-5min per 100 sentences depending on hardware.
