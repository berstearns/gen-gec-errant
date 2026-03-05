# L2 Learner Error Analysis Pipeline

End-to-end pipeline for comparing grammatical error patterns between base language models and models continued-pretrained on L2 learner corpora (EFCAMDAT).

## Pipeline Architecture

```
EFCAMDAT Sentences
        │
        ▼
┌─────────────────────┐
│  Prompt Construction │   Split sentence → prompt prefix + reference
└─────────┬───────────┘
          │
     ┌────┴────┐
     ▼         ▼
┌─────────┐ ┌──────────────┐
│ GPT-2   │ │ GPT-2        │   (+ Pythia-70m, 160m, 410m, ...)
│ (base)  │ │ (learner .pt)│
└────┬────┘ └──────┬───────┘
     │              │
     ▼              ▼
  Generated      Generated
  Continuations  Continuations
     │              │
     └──────┬───────┘
            ▼
  ┌──────────────────┐
  │  GEC Correction  │   CoEdit-Large or Gemma-2-2B-IT
  └────────┬─────────┘
           ▼
  ┌──────────────────┐
  │  ERRANT Annotate │   Original vs Corrected → typed errors
  └────────┬─────────┘
           ▼
  ┌──────────────────┐
  │    Analysis       │   PPL, error counts, type breakdown, stat tests
  └──────────────────┘
```

## Setup (uv)

```bash
# 1. Create venv and install all dependencies
uv venv
source .venv/bin/activate
uv pip install -e .

# 2. Download spaCy model (required by ERRANT)
uv run python -m spacy download en_core_web_sm
```

## Preprocessing: EFCAMDAT CSV → Sentences

Your `cleaned_efcamdat.csv` contains full essays. The pipeline needs individual
sentences. Run the preprocessor first:

```bash
# Basic — all essays, all levels
uv run python preprocess_efcamdat.py \
    --input data/cleaned_efcamdat.csv \
    --output data/efcamdat_sentences.csv

# With filters — only A1/A2, only Arabic L1, max 5000 essays
uv run python preprocess_efcamdat.py \
    --input data/cleaned_efcamdat.csv \
    --output data/efcamdat_sentences_a1a2_arabic.csv \
    --cefr_filter A1,A2 \
    --l1_filter Arabic \
    --max_essays 5000

# If auto-detection fails, specify columns manually
uv run python preprocess_efcamdat.py \
    --input data/cleaned_efcamdat.csv \
    --output data/efcamdat_sentences.csv \
    --text_col 23 \
    --corrected_col 24
```

The preprocessor outputs a clean CSV with columns:
`essay_id, sentence_idx, sentence, cefr_level, l1_language, topic, word_count`

## Usage

### Basic: GPT-2 base vs learner-tuned
```bash
uv run python run_pipeline.py \
    --data_path data/efcamdat_sentences.csv \
    --learner_checkpoint /path/to/your/learner_model.pt \
    --models gpt2-base gpt2-learner \
    --max_sentences 500 \
    --output_dir results/gpt2_comparison
```

### Full: Multiple model families and sizes
```bash
uv run python run_pipeline.py \
    --data_path data/efcamdat_sentences.csv \
    --learner_checkpoint /path/to/your/learner_model.pt \
    --models gpt2-base gpt2-medium gpt2-learner pythia-70m pythia-160m pythia-410m \
    --max_sentences 1000 \
    --output_dir results/full_comparison \
    --gec_method dedicated \
    --batch_size 16
```

### Using Gemma for GEC instead of CoEdit
```bash
uv run python run_pipeline.py \
    --data_path data/efcamdat_sentences.csv \
    --learner_checkpoint /path/to/your/learner_model.pt \
    --models gpt2-base gpt2-learner \
    --gec_method llm \
    --gec_model google/gemma-2-2b-it \
    --output_dir results/gemma_gec
```

### Resume from existing generations (skip generation, re-run analysis)
```bash
uv run python run_pipeline.py \
    --data_path data/efcamdat_sentences.csv \
    --output_dir results/gpt2_comparison \
    --skip_generation
```

## Available Models

| Key           | HuggingFace ID              | Notes                          |
|---------------|-----------------------------|---------------------------------|
| gpt2-base     | gpt2                        | 124M params                    |
| gpt2-medium   | gpt2-medium                 | 355M params                    |
| gpt2-large    | gpt2-large                  | 774M params                    |
| gpt2-learner  | gpt2 + your .pt checkpoint  | Requires --learner_checkpoint  |
| pythia-70m    | EleutherAI/pythia-70m       | 70M params                     |
| pythia-160m   | EleutherAI/pythia-160m      | 160M params                    |
| pythia-410m   | EleutherAI/pythia-410m      | 410M params                    |
| pythia-1b     | EleutherAI/pythia-1b        | 1B params                      |

## Checkpoint Loading

The pipeline supports several .pt checkpoint formats:

```python
# Any of these will be auto-detected:
{"model_state_dict": state_dict}   # PyTorch Lightning style
{"state_dict": state_dict}         # Alternative
{"model": state_dict}              # Another variant
state_dict                         # Raw state dict
```

If your checkpoint has a different structure, edit `generation.py > load_model()`.

## Outputs

```
results/
├── full_results.csv                 # ⭐ Main CSV: all data per sentence, all models side-by-side
├── errors_long_format.csv           # ⭐ One row per error (for pivot tables / R analysis)
├── gpt2-base_summary.json           # Per-model metrics
├── gpt2-learner_summary.json
├── pythia-70m_summary.json
├── model_comparison.json            # Cross-model comparison + stat tests
├── raw_results.json                 # All generations, corrections, annotations
├── prompts.json                     # Input prompts and references
└── plots/
    ├── perplexity_comparison.png    # Bar chart: mean PPL per model
    ├── error_comparison.png         # Bar chart: error rate + avg errors
    ├── error_type_breakdown.png     # Grouped bar: top 10 ERRANT error types
    ├── ppl_vs_errors_scatter.png    # Scatter: PPL vs error count per sentence
    └── combined_metric.png          # Bar chart: PPL × Errors combined metric
```

## Analysis Metrics

### Per-model
- **Perplexity**: mean, median, std, quartiles
- **Error count**: total, avg per sentence, rate (fraction with ≥1 error)
- **Error types**: Full ERRANT breakdown (M:DET, R:VERB:SVA, U:PREP, etc.)
- **Combined**: PPL × avg_errors (single summary score; higher = worse)

### Cross-model
- **Mann-Whitney U tests**: Non-parametric tests comparing PPL and error distributions between model pairs
- **Effect sizes**: Visible in the plots and summary tables

## Key ERRANT Error Types

| Code        | Meaning                       | Common in L2? |
|-------------|-------------------------------|:-------------:|
| M:DET       | Missing determiner            |      ✓✓       |
| R:PREP      | Wrong preposition             |      ✓✓       |
| R:VERB:SVA  | Subject-verb agreement error  |      ✓✓       |
| U:DET       | Unnecessary determiner        |      ✓        |
| M:PREP      | Missing preposition           |      ✓        |
| R:VERB:TENSE| Wrong verb tense             |      ✓        |
| R:SPELL     | Spelling error                |      ✓        |
| R:NOUN:NUM  | Noun number error             |      ✓        |

## Extending the Pipeline

### Adding a new model family
Edit `MODEL_REGISTRY` in `run_pipeline.py`:
```python
MODEL_REGISTRY["llama-tiny"] = ModelConfig(
    name="llama-tiny",
    hf_model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    model_family="llama",
)
```

### Adding learner-tuned variants for Pythia
If you fine-tune Pythia on EFCAMDAT and save a checkpoint:
```python
MODEL_REGISTRY["pythia-70m-learner"] = ModelConfig(
    name="pythia-70m-learner",
    hf_model_id="EleutherAI/pythia-70m",
    model_family="pythia",
    is_learner_tuned=True,
    checkpoint_path="/path/to/pythia_learner.pt",
)
```

### Custom GEC prompt
Edit `config.py > GECConfig.gec_prompt_template`.

## Tips

1. **Start small**: Use `--max_sentences 100` for debugging, then scale up.
2. **GPU memory**: Process one model at a time (already handled — models are loaded/freed sequentially).
3. **CoEdit vs Gemma for GEC**: CoEdit is faster and more consistent; Gemma catches more nuanced errors but may overcorrect. Consider running both and comparing.
4. **Reproducibility**: Set `--seed` and keep `temperature=1.0` with `do_sample=True` for controlled randomness, or use `temperature=0` for deterministic output.
