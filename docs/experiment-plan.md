# Experiment Plan: Generation-GEC-Annotation Pipeline

## Overview

This document tracks the full experimental setup for the gen-gec-errant pipeline:
fine-tuned "artificial learner" models generate text from learner prompts, a GEC
system corrects the output, and ERRANT annotates the errors. The goal is to
measure how error profiles differ across model families, sizes, and fine-tuning
conditions — and how sensitive those profiles are to the choice of GEC model.

---

## 1. Trained Artificial-Learner Models (NWP on EFCAMDAT)

All models are fine-tuned on **norm-EFCAMDAT-ALL-CONCAT.csv** (full normalized
EFCAMDAT learner corpus, ~53k grouped samples) using the `axolotl-artificial-learners`
training pipeline with HuggingFace Trainer (causal LM objective).

Training repos:
- `/home/b/p/all-my-tiny-projects/vastai/training-artificial-learners-axolotl` (orchestrator + deploy scripts)
- `/home/b/p/al-eval/axolotl-artificial-learners` (configs + training code)

Remote model storage: `rclone i:/_p/artificial-learners/models/`

### 1.1 Pythia Models (EleutherAI) — 15 epochs, BF16, block_size=2048

| Model | Params | LR | Eff. Batch | Grad Ckpt | Eval PPL | Status | rclone Path |
|-------|--------|----|-----------|-----------|----------|--------|-------------|
| pythia-70m | 70M | 5e-5 | 40 | No | — | **TODO** | `pythia/pythia-70m-all-data/` |
| pythia-160m | 160M | 5e-5 | 32 | No | — | **TODO** | `pythia/pythia-160m-all-data/` |
| pythia-410m | 410M | 3e-5 | 12 | No | 12.25 | **Done** | `pythia/pythia-410m-all-data/` |
| pythia-1b | 1B | 2e-5 | 32 | Yes | 12.27 | **Done** | `pythia/pythia-1b-all-data/` |
| pythia-1.4b | 1.4B | 1e-5 | 28 | Yes | — | **TODO** | `pythia/pythia-1.4b-all-data/` |

Training results (completed):
- **pythia-410m**: train_loss=2.50, eval_loss=2.51, eval_ppl=12.25 (15 ep, 18k s)
- **pythia-1b**: train_loss=2.51, eval_loss=2.51, eval_ppl=12.27 (15 ep, 36k s)

Note: pythia-410m and pythia-1b converge to nearly identical eval PPL — worth
investigating whether the 410m is already at the corpus entropy floor.

### 1.2 GPT-2 Models (OpenAI) — 10 epochs, FP16, block_size=1024

| Model | Params | LR | Eff. Batch | Eval PPL | Status | rclone Path |
|-------|--------|----|-----------|----------|--------|-------------|
| gpt2-small | 117M | 5e-5 | 32 | — | **Done** | `2026-02-23-model/gpt2-small-all-data-resume/` |
| gpt2-medium | 345M | 5e-5 | 32 | 25.70 | **Done** | `gpt2/gpt2-medium-all-data-resume/` |
| gpt2-large | 774M | 3e-5 | 32 | 18.36 | **Done** | `gpt2/gpt2-large-all-data-20260222-092036/` |
| gpt2-xl | 1.5B | 1e-5 | 32 | — | **TODO** | — |

Training results (completed):
- **gpt2-medium**: train_loss=2.31, eval_loss=3.25, eval_ppl=25.70 (10 ep)
- **gpt2-large**: train_loss=3.00, eval_loss=2.91, eval_ppl=18.36 (10 ep)

### 1.3 Legacy GPT-2 Models (CEFR-stratified, older runs)

Stored as `.pt` files in `rclone i:/_p/artificial-learners/models/`:

| File | Subset |
|------|--------|
| `gpt2-small-a1-full-efcamdat.pt` | CEFR A1 |
| `gpt2-small-a2-full-efcamdat.pt` | CEFR A2 |
| `gpt2-small-b1-full-efcamdat.pt` | CEFR B1 |
| `gpt2-small-b2-full-efcamdat.pt` | CEFR B2 |
| `gpt2-small-c1-full-efcamdat.pt` | CEFR C1 |
| `gpt2-small-all-full-efcamdat.pt` | All data |

Also in `selected_models/`:
- `gpt2-efcamdat-remainder-{a1,a2,b1,b2,c1}-AL/` (older CEFR-split runs)
- `bert-full-efcamdat/` (BERT encoder, for contrastive/classification)

### 1.4 Models Still To Train

These model families have configs generated but no fine-tuned checkpoints yet:

| Model | Params | Config exists | Priority |
|-------|--------|---------------|----------|
| SmolLM2-135M | 135M | Yes | Medium |
| SmolLM2-360M | 360M | Yes | Medium |
| SmolLM2-1.7B | 1.7B | Yes | Medium |
| Mamba-130M | 130M | Yes | Low (architecture comparison) |
| Mamba-370M | 370M | Yes | Low |
| Mamba-790M | 790M | Yes | Low |
| OLMo-2-1B | 1B | Yes | Low |
| Qwen2.5-0.5B | 0.5B | Yes | Low |
| Qwen2.5-1.5B | 1.5B | Yes | Low |
| LLaMA-3.2-1B | 1B | Yes | Low |
| TinyLlama-1.1B | 1.1B | Yes | Low |
| RWKV6-1.6B | 1.6B | Yes | Low (linear attention) |
| Gemma-2-2B | 2B | Yes | Low |

---

## 2. Pipeline Experiments to Run

The gen-gec-errant pipeline runs in 5 stages:

```
Data Loading → Generation → GEC Correction → ERRANT Annotation → Analysis
```

### 2.1 Experiment Matrix: Generation Models

Each generation model is run through the full pipeline. Two classes of models:

**A. Pretrained baselines (no fine-tuning):**

| Config | Model | Purpose |
|--------|-------|---------|
| `per-model/gpt2-small.yaml` | `gpt2` | Baseline — generic English |
| `per-model/gpt2-medium.yaml` | `gpt2-medium` | Baseline — scaling |
| `per-model/gpt2-large.yaml` | `gpt2-large` | Baseline — scaling |
| `per-model/gpt2-xl.yaml` | `gpt2-xl` | Baseline — scaling |
| `per-model/pythia-70m.yaml` | `EleutherAI/pythia-70m` | Baseline |
| `per-model/pythia-160m.yaml` | `EleutherAI/pythia-160m` | Baseline |
| `per-model/pythia-410m.yaml` | `EleutherAI/pythia-410m` | Baseline |
| `per-model/pythia-1b.yaml` | `EleutherAI/pythia-1b` | Baseline |
| `per-model/pythia-1.4b.yaml` | `EleutherAI/pythia-1.4b` | Baseline |
| `per-model/smollm2-135m.yaml` | `HuggingFaceTB/SmolLM2-135M` | Baseline |
| `per-model/smollm2-360m.yaml` | `HuggingFaceTB/SmolLM2-360M` | Baseline |
| `per-model/smollm2-1.7b.yaml` | `HuggingFaceTB/SmolLM2-1.7B` | Baseline |
| + 13 other model configs | Various | Architecture comparison |

**B. Fine-tuned on EFCAMDAT (artificial learners):**

| Config | Model | Checkpoint |
|--------|-------|------------|
| `per-model/ft-gpt2-small.yaml` | ft-gpt2-small | `2026-02-23-model/gpt2-small-all-data-resume/final` |
| `per-model/ft-gpt2-medium.yaml` | ft-gpt2-medium | `gpt2/gpt2-medium-all-data-resume/final` |
| `per-model/ft-gpt2-large.yaml` | ft-gpt2-large | `gpt2/gpt2-large-all-data-20260222-092036/final` |
| **TODO** | ft-pythia-410m | `pythia/pythia-410m-all-data/final` |
| **TODO** | ft-pythia-1b | `pythia/pythia-1b-all-data/final` |
| **TODO** | ft-pythia-70m | _pending training_ |
| **TODO** | ft-pythia-160m | _pending training_ |
| **TODO** | ft-pythia-1.4b | _pending training_ |

### 2.2 Experiment Priority: Pythia Focus

The Pythia family is the primary focus because it offers:
- 5 sizes (70M–1.4B) with identical architecture and training data (The Pile)
- Clean scaling analysis (same tokenizer, same pretraining corpus)
- Smaller sizes feasible on consumer GPUs

**Priority pipeline runs (Pythia fine-tuned vs pretrained):**

| Run | Generation Model | GEC Model | Dataset | Notes |
|-----|-----------------|-----------|---------|-------|
| 1 | pythia-410m (pretrained) | coedit-large | norm-CELVA-SP | Baseline |
| 2 | ft-pythia-410m (EFCAMDAT) | coedit-large | norm-CELVA-SP | Learner model |
| 3 | pythia-1b (pretrained) | coedit-large | norm-CELVA-SP | Baseline |
| 4 | ft-pythia-1b (EFCAMDAT) | coedit-large | norm-CELVA-SP | Learner model |
| 5 | pythia-70m (pretrained) | coedit-large | norm-CELVA-SP | Smallest baseline |
| 6 | ft-pythia-70m (EFCAMDAT) | coedit-large | norm-CELVA-SP | Smallest learner |
| 7–10 | Repeat above for pythia-160m, pythia-1.4b | | | Full scaling curve |

---

## 3. GEC Model Ablation Plan

The GEC step is critical — it determines what counts as an "error". Different GEC
models may have different correction biases, affecting error type distributions
and error counts.

### 3.1 Current GEC Models in Pipeline

| Method | Model ID | Type | Notes |
|--------|----------|------|-------|
| `dedicated` | `grammarly/coedit-large` | Seq2Seq (T5-based) | Current default. Instruction-tuned for editing tasks. |
| `llm` | `google/gemma-2-2b-it` | Causal LM | LLM-based correction via prompt. Heavier but more flexible. |

### 3.2 GEC Models to Test for Ablation

**Phase 1 — Initial ablation on norm-CELVA-SP** (small dataset, quick iteration):

| Model ID | Type | Method | Rationale |
|----------|------|--------|-----------|
| `grammarly/coedit-large` | Seq2Seq (T5-large) | `dedicated` | **Current default**. Strong general editing model. |
| `grammarly/coedit-xl` | Seq2Seq (T5-xl) | `dedicated` | Larger variant — test if scaling GEC changes error profiles. |
| `grammarly/coedit-xxl` | Seq2Seq (T5-xxl) | `dedicated` | Largest coedit — upper bound for this family. |
| `pszemraj/flan-t5-large-grammar-synthesis` | Seq2Seq (Flan-T5) | `dedicated` | Alternative T5-based GEC, different training data. |
| `vennify/t5-base-grammar-correction` | Seq2Seq (T5-base) | `dedicated` | Smaller T5 baseline — test sensitivity to GEC model size. |
| `Unbabel/gec-t5_small` | Seq2Seq (T5-small) | `dedicated` | Smallest seq2seq — lower bound. |
| `google/gemma-2-2b-it` | Causal LM | `llm` | **Already configured**. LLM-based alternative. |
| `Qwen/Qwen2.5-3B-Instruct` | Causal LM | `llm` | Different LLM family for GEC. |
| `meta-llama/Llama-3.2-3B-Instruct` | Causal LM | `llm` | Another LLM for comparison. |

**Phase 2 — Extend to more datasets** (after selecting best GEC configuration):

| Dataset | Path | Description |
|---------|------|-------------|
| norm-CELVA-SP | `norm-CELVA-SP.csv` | Spanish L1 learners (initial ablation) |
| norm-EFCAMDAT subsets | `cleaned_efcamdat_proficiency_{A1..C2}.txt` | CEFR-stratified subsets |
| norm-EFCAMDAT L1 subsets | `cleaned_efcamdat_l1_{lang}.txt` | L1-stratified subsets |
| Full EFCAMDAT | `norm-EFCAMDAT-ALL-CONCAT.csv` | Complete corpus |
| Other learner corpora | _TBD_ | e.g., FCE, NUCLE, W&I+LOCNESS, Lang-8 |

### 3.3 Ablation Experiment Design

The ablation tests how much the **choice of GEC model** affects the error profiles
detected by ERRANT. For a fixed generation model, vary the GEC model:

```
Generation: ft-pythia-410m (fixed)
    └── GEC: coedit-large       → ERRANT → error profile A
    └── GEC: coedit-xl          → ERRANT → error profile B
    └── GEC: coedit-xxl         → ERRANT → error profile C
    └── GEC: flan-t5-grammar    → ERRANT → error profile D
    └── GEC: t5-base-grammar    → ERRANT → error profile E
    └── GEC: gemma-2-2b-it      → ERRANT → error profile F
    └── GEC: qwen2.5-3b-inst    → ERRANT → error profile G
    └── GEC: llama-3.2-3b-inst  → ERRANT → error profile H
```

Metrics to compare across GEC models:
- Total error count (per sentence and per model)
- Error type distribution (ERRANT categories: M:DET, R:VERB:SVA, etc.)
- Precision-like measure: % of corrections that are genuine errors vs. style changes
- Agreement between GEC models (pairwise overlap of detected errors)
- Prompt-region vs. generation-region error breakdown

### 3.4 Creating New GEC Configs

To add a new GEC model, create a YAML in `configs/gec/`:

```yaml
# configs/gec/coedit-xl.yaml
method: dedicated
model_id: grammarly/coedit-xl
batch_size: 4
device: auto
```

```yaml
# configs/gec/llama3-gec.yaml
method: llm
model_id: meta-llama/Llama-3.2-3B-Instruct
batch_size: 4
device: auto
prompt_template: >
  Correct any grammatical errors in the following sentence.
  Only fix grammar — do not change meaning, vocabulary, or style.
  If the sentence is already correct, return it unchanged.

  Sentence: {sentence}

  Corrected sentence:
```

Then reference it in a pipeline config by overriding the `gec` section, or use
CLI overrides:

```bash
python -m gen_gec_errant.pipeline \
    --config configs/pipeline/per-model/pythia-410m.yaml \
    --gec.model_id grammarly/coedit-xl
```

---

## 4. Datasets

### 4.1 Primary Evaluation Dataset

**norm-CELVA-SP** — Spanish L1 English learners
- Path: `/home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits/norm-CELVA-SP.csv`
- Column: `text`
- Used in all per-model pipeline configs (`max_sentences: 10` for dev, remove for full)

### 4.2 Training Data (for fine-tuning generation models)

**norm-EFCAMDAT-ALL-CONCAT.csv** — Full EFCAMDAT corpus
- Path (remote): `/workspace/splits/norm-EFCAMDAT-ALL-CONCAT.csv`
- Path (rclone): `i:/phd-experimental-data/cefr-classification/data/splits/`
- ~53k grouped samples, ~83k lines
- Subsets available by CEFR level (A1–C2) and L1 (10 languages)

### 4.3 Future Evaluation Datasets

| Dataset | Description | Status |
|---------|-------------|--------|
| norm-CELVA-SP | Spanish L1 learners | **Active** |
| EFCAMDAT (by CEFR) | A1–C2 proficiency splits | Available |
| EFCAMDAT (by L1) | 10 L1 backgrounds | Available |
| FCE | Cambridge First Certificate | To acquire |
| NUCLE | NUS Corpus of Learner English | To acquire |
| W&I+LOCNESS | Write & Improve + LOCNESS | To acquire |
| Lang-8 | Lang-8 Learner Corpus | To acquire |

---

## 5. Saved Model Inventory (rclone)

Full path: `rclone i:/_p/artificial-learners/models/`

```
models/
├── gpt2-small-{a1,a2,b1,b2,c1,all}-full-efcamdat.pt   # Legacy .pt checkpoints
├── selected_models/
│   ├── bert-full-efcamdat/                               # BERT encoder
│   └── gpt2-efcamdat-remainder-{a1..c1}-AL/             # Older CEFR-split GPT-2
├── 2026-02-20-model/
│   └── gpt2-large-all-data-20260222-092036/              # GPT-2 Large (10ep, ppl=18.36)
├── 2026-02-23-model/
│   ├── gpt2-small-all-data-resume/                       # GPT-2 Small (10ep)
│   └── gpt2-medium-all-data-20260221-232834/             # GPT-2 Medium (10ep, ppl=25.70)
├── gpt2/
│   ├── gpt2-medium-all-data-resume/                      # GPT-2 Medium resumed
│   └── gpt2-large-all-data-20260222-092036/              # GPT-2 Large
└── pythia/
    ├── pythia-410m-all-data/                              # 15 checkpoints + final (ppl=12.25)
    └── pythia-1b-all-data/                                # 15 checkpoints + final (ppl=12.27)
```

### Local mirror

Fine-tuned models are mirrored locally at:
`/home/b/p/my-data/i/_p/artificial-learners/models/`

This is the path used in `per-model/ft-gpt2-*.yaml` configs.

---

## 6. Running the Experiments

### 6.1 Single model run

```bash
python -m gen_gec_errant.pipeline --config configs/pipeline/per-model/pythia-410m.yaml
```

### 6.2 All models via orchestrator script

```bash
bash scripts/run_pipeline.sh \
    --configs-dir configs/pipeline/per-model \
    --output-root outputs/full-run \
    --device auto \
    --batch-size 2 \
    --max-sentences 0   # 0 = all sentences
```

### 6.3 GEC ablation (single generation model, multiple GEC models)

```bash
for gec_model in grammarly/coedit-large grammarly/coedit-xl grammarly/coedit-xxl; do
    slug=$(echo "$gec_model" | tr '/' '-')
    python -m gen_gec_errant.pipeline \
        --config configs/pipeline/per-model/pythia-410m.yaml \
        --gec.model_id "$gec_model" \
        --output_dir "outputs/gec-ablation/ft-pythia-410m--${slug}"
done
```

### 6.4 Training new models (Vast.ai)

```bash
# From the orchestrator repo:
cd /home/b/p/all-my-tiny-projects/vastai/training-artificial-learners-axolotl

# Bootstrap + train Pythia models:
./orchestrator.sh --mode ssh --ssh-url "$URL" quick_start
./orchestrator.sh copy-rclone
./orchestrator.sh project-setup-axolotl
./orchestrator.sh get-data rclone "i:/phd-experimental-data/cefr-classification/data/splits/" /workspace/splits/
./orchestrator.sh --detach train-nwp --config configs/nwp/generated/csv/pythia-70m-all-data-32gb-15ep.yaml

# Sync results back:
./orchestrator.sh --detach sync-models \
    --source /workspace/training-outputs-32gb-15ep \
    --dest "i:/_p/artificial-learners/models/pythia"
```

---

## 7. TODO / Next Steps

### Training
- [ ] Train pythia-70m on EFCAMDAT (15 epochs)
- [ ] Train pythia-160m on EFCAMDAT (15 epochs)
- [ ] Train pythia-1.4b on EFCAMDAT (15 epochs, needs >32GB GPU)
- [ ] Train gpt2-xl on EFCAMDAT (needs 8-bit Adam or 48GB GPU)
- [ ] Consider training SmolLM2 variants for architecture comparison

### Pipeline Configs
- [ ] Create `ft-pythia-410m.yaml` and `ft-pythia-1b.yaml` per-model configs pointing to rclone checkpoints
- [ ] Create per-model configs for remaining fine-tuned Pythia models as they complete
- [ ] Increase `max_sentences` beyond 10 for full evaluation runs

### GEC Ablation
- [ ] Create GEC config YAMLs for each ablation model (coedit-xl, coedit-xxl, flan-t5, etc.)
- [ ] Run GEC ablation on norm-CELVA-SP with ft-pythia-410m as fixed generation model
- [ ] Analyze error profile sensitivity to GEC model choice
- [ ] Select best GEC configuration for main experiments

### Evaluation
- [ ] Full pipeline run: all Pythia sizes (pretrained + fine-tuned) on norm-CELVA-SP
- [ ] Full pipeline run: all GPT-2 sizes (pretrained + fine-tuned) on norm-CELVA-SP
- [ ] Cross-dataset validation on EFCAMDAT subsets
- [ ] Acquire and test on FCE, NUCLE, W&I+LOCNESS corpora
