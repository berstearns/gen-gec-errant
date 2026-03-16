# Execution Steps

## Step 0: Setup experiment directory
- Create `experiment/` subdirectory structure
- Verify source data CSVs exist
- Write per-dataset YAML configs into `experiment/configs/`

## Step 1: Load data & generate prompts
- For each dataset: load CSV, filter by word count (10-500 words)
- Split each sentence at 50% into (prompt, reference_continuation)
- Save `prompts.json` in each dataset's output directory

## Step 2: Generate text with GPT-2 native
- Load pre-trained GPT-2 from HuggingFace (`gpt2`)
- Generate continuations from each prompt:
  - `max_new_tokens=50`, `min_new_tokens=10`
  - `temperature=1.0`, `top_k=50`, `top_p=0.95`
  - `repetition_penalty=1.2`, `do_sample=True`
- Compute perplexity for each full text (prompt + continuation)
- Add `learner_baseline` pseudo-model (reference continuations)

## Step 3: Grammatical Error Correction (GEC)
- Correct all generated continuations using `grammarly/coedit-large` (T5-based)
- Also correct full texts (prompt + continuation)
- GEC input format: "Fix grammatical errors in this sentence: [text]"
- Batch size: 4

## Step 4: ERRANT error annotation
- Compare original text vs. GEC-corrected text using ERRANT
- Classify each edit by error type (M:DET, R:VERB:SVA, U:PRON, etc.)
- For full-text annotations: classify errors by region (prompt vs. generation)
- Compute error counts and type distributions per sentence

## Step 5: Analysis & export
- Compute per-model summaries (perplexity stats, error stats, top 10 error types)
- Export `raw_results.json` (complete data)
- Export `full_results.tsv` (1 row per sentence, flat CSV)
- Export `errors_long_format.tsv` (1 row per error, for R/statistical analysis)
- Generate plots (perplexity comparison, error breakdown, etc.)
- Compare gpt2-small vs. learner_baseline error profiles

## Step 6: Cross-dataset summary
- Aggregate results across all datasets
- Produce a single summary comparing error profiles by dataset
