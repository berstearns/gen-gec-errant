# Pythia 160M fine-tuned on EFCAMDAT all-data — Error Profile Experiment

## Goal

Run the full generate-GEC-ERRANT-analysis pipeline using **ft-pythia-160m**
(Pythia 160M fine-tuned on EFCAMDAT all-data) on learner corpus data.
This model is a **fine-tuned artificial learner**: pre-trained EleutherAI/pythia-160m
further trained on EFCAMDAT all-data (English learner texts across all CEFR levels).

Compare its error profile against the native (pre-trained) baseline and learner
reference continuations to evaluate how well it reproduces learner-like errors.

## Pipeline

1. Download model weights from Google Drive (via rclone) if not locally available
2. Load learner text data, split into prompts + reference continuations
3. Generate continuations with ft-pythia-160m (fine-tuned on EFCAMDAT)
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

- **ft-pythia-160m** (160M parameters)
- Base: `EleutherAI/pythia-160m` (pre-trained)
- Fine-tuned on: EFCAMDAT all-data (full learner corpus, all CEFR levels)
- Architecture family: pythia
- rclone source: `i:/_p/artificial-learners/models/pythia/pythia-160m-all-data`
- Local path: `/home/b/p/my-data/i/_p/artificial-learners/models/pythia/pythia-160m-all-data/final`

## Constraints

- Uses the `gen_gec_errant` package from this repository (does NOT modify src/)
- All outputs go into `experiment/` subdirectory
- Reproducible via fixed seed (42)
- Model weights downloaded from Google Drive via rclone (only `final/` checkpoint)
