# GPT-2 Native Generation Error Profile Experiment

## Goal

Run the full generate-GEC-ERRANT-analysis pipeline using pre-trained GPT-2
(native, no fine-tuning) on learner corpus data. Evaluate the error profile
of GPT-2's text continuations across multiple datasets to establish a
**native model baseline** for comparison with fine-tuned artificial learner models.

## Pipeline

1. Load learner text data, split into prompts + reference continuations
2. Generate continuations with pre-trained GPT-2 (HuggingFace `gpt2`)
3. Run grammatical error correction (GEC) on generated text via `coedit-large`
4. Annotate errors with ERRANT (automatic error type classification)
5. Analyze: per-model summaries, CSV exports, visualizations

## Datasets

Source: `/home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits/`

| Dataset | File | Rows | Description |
|---|---|---|---|
| CELVA-SP | norm-CELVA-SP.csv | 1,742 | Spanish L1 learner English (primary) |
| EFCAMDAT-test | norm-EFCAMDAT-test.csv | 20,000 | In-domain learner English |
| KUPA-KEYS | norm-KUPA-KEYS.csv | 1,006 | Cross-corpus learner English |

All CSVs: schema `writing_id, l1, cefr_level, text`

## Model

- **gpt2** (117M parameters, pre-trained on WebText)
- Auto-downloaded from HuggingFace Hub (~550MB)
- No fine-tuning, no learner-specific training
- This is the "native" baseline: a model that has seen only standard English

## Constraints

- Uses the `gen_gec_errant` package from this repository (does NOT modify src/)
- All outputs go into `experiment/` subdirectory
- Reproducible via fixed seed (42)
- CPU-compatible with sentence limits; full reproduction needs GPU for speed
