# Evaluating Artificial Learner Error Distributions via Automatic ERRANT Error Tagging

## AutoResearchClaw Topic

> Do language models fine-tuned on L2 learner corpora (EFCAMDAT) produce error distributions that converge toward the empirical error profile of the original human learner text, as measured by automatic ERRANT error tagging? Compare three conditions — (1) original human learner text as ground-truth reference, (2) artificial learner models fine-tuned on EFCAMDAT, and (3) native pretrained baselines — across model families (GPT-2, Pythia, SmolLM2), scales (70M–1.7B parameters), and learner L1 backgrounds. Evaluate whether the AL error-type distributions approximate the human learner ground truth, form developmental clines across CEFR proficiency levels, and capture L1-transfer effects, using a generation–correction–annotation pipeline with statistical divergence measures (Jensen–Shannon divergence, Spearman rank correlation) against the empirical learner error profiles.

**Domains:** `["computational-linguistics", "second-language-acquisition", "cognitive-modeling", "grammatical-error-correction"]`

---

## 1. Research Motivation and Background

### 1.1 The Gap

Prior cognitive modeling work tests whether neural LMs capture human language processing via surprisal-based reading-time predictions (Levy 2008), syntactic acceptability judgements (Warstadt et al. 2020, BLiMP), and neural alignment with brain recordings (Schrimpf et al. 2021). These are *comprehension-side* diagnostics. There is no established framework for testing whether LMs capture *production-side* patterns characteristic of second-language learners — specifically, whether fine-tuning on L2 text produces structured interlanguage error grammars rather than uniform degradation.

### 1.2 SLA Theory Predictions

Second-language acquisition research has established that L2 learners produce systematic, predictable error patterns:

- **Natural Order Hypothesis** (Goldschneider & DeKeyser 2001): English morphemes are acquired in a stable, predictable order. Verbal inflection (-s, -ed, -ing) and articles are late-acquired and most error-prone.
- **Processability Theory** (Pienemann 1998): Structures requiring cross-phrasal information exchange (e.g., subject-verb agreement) are more error-prone at lower proficiency.
- **Interlanguage** (Selinker 1972): The learner's evolving grammar is shaped by L1 transfer, input frequency, and processing constraints — producing systematic (not random) errors.
- **L1 Transfer in Determiners** (Ionin et al. 2004): Learners from L1s lacking articles (Arabic, Chinese, Japanese, Korean) show elevated determiner errors.

### 1.3 Core Hypothesis

If a language model fine-tuned on L2 learner text acquires an interlanguage-like error grammar, its error profile should:
1. **Converge toward** the error distribution of the original human learner text it was trained on
2. Show **selective elevation** in SLA-identified vulnerable categories (determiners, verbal morphology, prepositions, subject-verb agreement) — NOT uniform degradation across all error types
3. **Diverge from** the error profile of its native pretrained baseline

---

## 2. Research Questions

1. **RQ1 (Three-Way Contrast):** How do the error distributions of (a) original human learner text, (b) artificial learner models, and (c) native pretrained baselines compare? Does the AL model's error profile fall between the native baseline and human learner text, or does it converge toward the human learner distribution?
2. **RQ2 (Distributional Alignment):** How closely does the AL model's ERRANT error-type distribution match the original learner text's distribution (measured by JSD and Spearman rank correlation)? Are the additional errors concentrated in the same acquisition-relevant categories (M:DET, R:VERB:SVA, M:VERB:FORM, U:DET, M:PRON) that dominate human learner output?
3. **RQ3 (Scale):** How does model scale (70M–1.7B parameters) interact with convergence toward the human learner error profile? Do larger models produce distributions closer to the empirical learner ground truth?
4. **RQ4 (Architecture):** Do different model families (GPT-2, Pythia, SmolLM2) converge on the same human learner error profile or acquire qualitatively different approximations?
5. **RQ5 (Developmental Cline):** Do error-type distributions shift monotonically across CEFR proficiency levels (A1→C1), mirroring known acquisition sequences — and does this hold for both human learner text and AL models?
6. **RQ6 (L1 Transfer):** Do L1-stratified fine-tuned models capture L1-specific transfer effects visible in the corresponding L1-stratified human learner data (e.g., elevated M:DET for L1-Arabic/Chinese speakers)?

---

## 3. Pipeline Architecture

The experiment uses a 5-stage automated pipeline (`gen_gec_errant`) implemented in Python.

### 3.1 Stage 1 — Data Loading

- **Input:** Learner sentences from EFCAMDAT (EF-Cambridge Open Language Database)
- **Processing:** Each sentence is split at approximately 50% word boundary into a **prompt prefix** and **reference continuation**
- **Constraints:** min_words=10, min_prompt_words=3, prompt_ratio=0.5
- **Output:** List of `{prompt, reference, full}` dictionaries

### 3.2 Stage 2 — Text Generation

- **Input:** Prompt prefixes from Stage 1
- **Models:** HuggingFace `AutoModelForCausalLM` (pretrained or fine-tuned checkpoints)
- **Hyperparameters:**
  - `max_new_tokens`: 50
  - `temperature`: 1.0
  - `top_k`: 50, `top_p`: 0.95 (nucleus sampling)
  - `repetition_penalty`: 1.2
  - `do_sample`: true
- **Output:** Generated continuations, full texts (prompt + continuation), per-sentence perplexity
- **Perplexity:** Computed via cross-entropy loss on full text tokens

### 3.3 Stage 3 — Grammatical Error Correction (GEC)

- **Primary GEC model:** `grammarly/coedit-large` (T5-large, ~770M params, seq2seq)
  - Beam search with `num_beams=4`
  - Input format: `"Fix grammatical errors in this sentence: {text}"`
- **Alternative GEC model:** `google/gemma-2-2b-it` (instruction-tuned causal LM, greedy decoding)
  - Used for robustness comparison
- **Output:** Corrected continuations, corrected full texts

### 3.4 Stage 4 — ERRANT Error Annotation

- **Tool:** ERRANT v3+ (Error Annotation Recognition and Correction Evaluation Tool) with spaCy backend
- **Processing:**
  1. Parse original and corrected text with spaCy
  2. Compute token-level alignment (edits)
  3. Classify each edit by ERRANT error taxonomy
  4. Convert to character offsets
  5. Classify each error as "prompt" or "generation" region based on character boundary
- **Error Taxonomy (hierarchical):**
  - **Operations:** M (Missing), R (Replacement), U (Unnecessary)
  - **Categories:** DET, VERB (SVA, TENSE, FORM), NOUN, ADJ, ADV, PREP, PUNCT, SPELL, PRON, ORTH, CONJ, WO, OTHER
  - **Examples:** M:DET (missing determiner), R:VERB:SVA (subject-verb agreement replacement), U:PUNCT (unnecessary punctuation)
- **Output per sentence:**
  ```
  ErrorAnnotation: {original_tokens, corrected_tokens, error_type, start_offset, end_offset, char_start, char_end, region}
  SentenceAnnotation: {original, corrected, errors[], num_errors, error_type_counts, prompt_error_count, generation_error_count, prompt_error_type_counts, generation_error_type_counts}
  ```

### 3.5 Stage 5 — Statistical Analysis and Visualization

- **Per-model metrics:** Mean/median/std perplexity, total errors, avg errors per sentence, error rate, sentences with errors, error type distribution (sorted by frequency), top-10 error types
- **Combined metric:** `ppl_x_errors = mean_ppl * avg_errors_per_sentence`
- **Statistical tests:** Mann-Whitney U (two-sided) for pairwise PPL and error count distributions
- **Distributional measures:** Jensen-Shannon divergence, Spearman rank correlation between model error vectors and empirical learner text error distributions
- **Region analysis:** Prompt-region errors vs. generation-region errors (separating pre-existing from model-introduced errors)
- **Outputs:**
  - `{model}_summary.json` — per-model metrics
  - `model_comparison.json` — cross-model statistical comparisons with p-values
  - `full_results.tsv` — 1 row per sentence, all models as columns
  - `errors_long_format.tsv` — 1 row per error (for pivot tables)
  - `plots/` — 5 PNG visualizations (perplexity comparison, error comparison, error type breakdown, PPL vs errors scatter, combined metric)

---

## 4. Three-Way Comparison Design

The central experimental design is a **three-way comparison** applied to every condition:

### 4.1 Condition A — Original Human Learner Text (Ground-Truth Reference)

The original learner sentences from the evaluation datasets, passed directly through the GEC + ERRANT pipeline (Stages 3–4) without any generation step. This produces the **empirical human learner error distribution** — the target profile that AL models should converge toward.

- Same sentences used as prompts for AL and native models
- The reference continuation (the actual words the learner wrote) is the text that gets corrected and annotated
- This is the gold-standard error profile: it reflects the real distribution of learner errors
- Enabled via `include_learner_baseline: true` in pipeline config

### 4.2 Condition B — Artificial Learner Models (Fine-Tuned on EFCAMDAT)

All 11 fine-tuned models generate continuations from the same prompts, which are then corrected and annotated. The core question: **does the AL error distribution approximate Condition A?**

| Model | Parameters | Family | Training | Precision | Epochs |
|-------|-----------|--------|----------|-----------|--------|
| ft-gpt2-small | 124M | GPT-2 | EFCAMDAT | FP16 | 10 |
| ft-gpt2-medium | 355M | GPT-2 | EFCAMDAT | FP16 | 10 |
| ft-gpt2-large | 774M | GPT-2 | EFCAMDAT | FP16 | 10 |
| ft-pythia-70m | 70M | Pythia | EFCAMDAT | BF16 | 15 |
| ft-pythia-160m | 160M | Pythia | EFCAMDAT | BF16 | 15 |
| ft-pythia-410m | 410M | Pythia | EFCAMDAT | BF16 | 15 |
| ft-pythia-1b | 1B | Pythia | EFCAMDAT | BF16 | 15 |
| ft-pythia-1.4b | 1.4B | Pythia | EFCAMDAT | BF16 | 15 |
| ft-smollm2-135m | 135M | SmolLM2 | EFCAMDAT | BF16 | 15 |
| ft-smollm2-360m | 360M | SmolLM2 | EFCAMDAT | BF16 | 15 |
| ft-smollm2-1.7b | 1.7B | SmolLM2 | EFCAMDAT | BF16 | 15 |

### 4.3 Condition C — Native Pretrained Baselines (Not Fine-Tuned)

Each fine-tuned model is compared against its own pretrained checkpoint. These models share identical architecture and initialization — the ONLY difference is continued training on EFCAMDAT. Native baselines establish the error profile of a model that has never seen learner text.

- `gpt2` (124M), `gpt2-medium` (355M), `gpt2-large` (774M)
- `EleutherAI/pythia-{70m,160m,410m,1b,1.4b}`
- `HuggingFaceTB/SmolLM2-{135M,360M,1.7B}`

### 4.4 What the Three-Way Comparison Reveals

| Comparison | Question Answered |
|-----------|-------------------|
| A vs. C (Learner text vs. Native model) | How different are real learner errors from native LM errors? Establishes the divergence that AL models need to bridge. |
| A vs. B (Learner text vs. AL model) | How well does the AL model approximate real learner errors? The core metric of success: JSD(AL ‖ Learner) should be small. |
| B vs. C (AL model vs. Native model) | Does fine-tuning on learner data actually shift the error distribution? Confirms that the AL model has moved away from native patterns. |
| Relative: JSD(B‖A) < JSD(C‖A) | The AL model is closer to real learners than its native baseline — the fine-tuning worked. |

---

## 5. Datasets

### 5.1 Training Corpus

**EFCAMDAT** (EF-Cambridge Open Language Database; Geertzen et al. 2013):
- Large-scale corpus of learner English essays
- CEFR proficiency levels A1–C2
- L1 metadata (100+ first languages)
- ~53k sentences used for fine-tuning
- 21,493 total sentences available

### 5.2 Evaluation Datasets

| Dataset | Sentences | Description | L1 |
|---------|-----------|-------------|-----|
| norm-CELVA-SP | 1,742 | Spanish L1 learner sentences | Spanish |
| norm-EFCAMDAT-test | 20,000 | Held-out EFCAMDAT test split | Mixed |
| norm-KUPA-KEYS | 1,006 | Additional learner corpus | Mixed |

### 5.3 Planned Stratifications

- **By CEFR level:** A1, A2, B1, B2, C1 (for developmental cline analysis)
- **By L1:** Arabic, Chinese, French, Japanese, Korean, Spanish (for L1-transfer analysis)

---

## 6. Evaluation Framework

### 6.1 Primary Metrics

| Metric | Description | Purpose |
|--------|-------------|---------|
| Perplexity (PPL) | Per-sentence cross-entropy loss | Measures model's distributional fit to learner text |
| Total errors | Sum of ERRANT-detected errors | Quantifies error volume |
| Avg errors/sentence | Mean errors per generated continuation | Normalized error rate |
| Error rate | Fraction of sentences with >=1 error | Error prevalence |
| Error type distribution | Count vector over ERRANT categories | Error profile fingerprint |
| PPL x errors | `mean_ppl * avg_errors_per_sentence` | Combined quality proxy |

### 6.2 Distributional Alignment Measures (Three-Way)

| Measure | Comparisons | Purpose |
|---------|-------------|---------|
| Jensen-Shannon Divergence | JSD(AL ‖ Learner), JSD(Native ‖ Learner), JSD(AL ‖ Native) | Similarity of error type distributions; AL should minimize JSD to Learner |
| Spearman Rank Correlation | rho(AL, Learner), rho(Native, Learner) | Whether models rank error types in same order as real learners |
| Mann-Whitney U | All pairwise (Learner vs AL, Learner vs Native, AL vs Native) | Statistical significance of PPL and error count differences |
| Convergence Ratio | JSD(AL‖Learner) / JSD(Native‖Learner) | How much of the native→learner gap has the AL model closed (0 = perfect, 1 = no improvement) |

### 6.3 SLA-Relevant Error Categories (Acquisition-Vulnerable)

These ERRANT tags are hypothesized to show **selective elevation** in learner-tuned models AND in the original learner text:

| ERRANT Tag | Linguistic Category | SLA Prediction |
|-----------|-------------------|----------------|
| M:DET | Missing determiner | Late-acquired; elevated for L1 without articles |
| U:DET | Unnecessary determiner | Overgeneralization of article use |
| R:VERB:SVA | Subject-verb agreement | Cross-phrasal processing difficulty |
| M:VERB:FORM | Missing verb form (infinitival *to*, bare verbs) | Morpheme acquisition order |
| R:VERB:TENSE | Verb tense errors | Tense-aspect system development |
| M:PRON | Missing pronoun (subject omission) | Pro-drop L1 transfer |
| R:PREP | Preposition replacement | Collocational errors |

### 6.4 Contrast with Non-SLA Categories

These ERRANT tags should show **no significant elevation** (control categories):

| ERRANT Tag | Why Not SLA-Relevant |
|-----------|---------------------|
| R:PART | Particle errors — not a major L2 developmental marker |
| U:ADV | Unnecessary adverbs — stylistic, not acquisition-driven |
| R:CONJ | Conjunction errors — low frequency in L2 error profiles |

---

## 7. Preliminary Results (Pilot Study)

### 7.1 Pilot Setup

- 20 sentences from EFCAMDAT (French L1, A2-B1)
- Three conditions: Original learner text vs. GPT-2 Base (native) vs. GPT-2 Base fine-tuned on EFCAMDAT (AL)

### 7.2 Aggregate Results

| Metric | Original Learner Text | Native GPT-2 | AL GPT-2 (Fine-Tuned) |
|--------|----------------------|-------------|----------------------|
| Mean PPL | — | 47.86 (σ=14.73) | 41.68 (σ=13.05) |
| Total errors | (empirical reference) | 60 | 103 |
| Avg errors/sentence | (empirical reference) | 3.00 | 5.15 |
| Error rate | (empirical reference) | 95% (19/20) | 100% (20/20) |

### 7.3 Error Type Distribution Shift

| Error Type | Native GPT-2 | AL GPT-2 | Fold Change | SLA-Relevant? |
|-----------|-------------|----------|-------------|---------------|
| R:NOUN | 16 | 15 | 0.94x | No |
| R:OTHER | 15 | 27 | 1.80x | No |
| R:ORTH | 7 | 20 | 2.86x | No |
| **M:VERB:FORM** | **1** | **5** | **5.00x** | **Yes** |
| **R:VERB:SVA** | **2** | **5** | **2.50x** | **Yes** |
| **U:DET** | **1** | **4** | **4.00x** | **Yes** |
| R:VERB | 0 | 4 | inf | Partial |
| M:OTHER | 1 | 3 | 3.00x | No |
| **M:PRON** | **1** | **3** | **3.00x** | **Yes** |
| U:NOUN | 0 | 2 | inf | No |

### 7.4 Key Preliminary Finding

The learner-tuned model shows **selective elevation** in SLA-predicted categories — not uniform degradation:
- Verbal morphology (M:VERB:FORM): 5x increase
- Determiners (U:DET): 4x increase
- Pronoun omission (M:PRON): 3x increase
- Subject-verb agreement (R:VERB:SVA): 2.5x increase

The model simultaneously has **lower perplexity** on learner text (better fit to interlanguage distribution) and **more errors** (reproducing learned error patterns) — it has learned to fluently produce errors.

**Missing from pilot:** The original learner text error distribution was not yet computed in the pilot. The full experiment must include this to answer the central question: does the AL error distribution converge toward the real learner distribution, or does it overshoot/undershoot systematically?

---

## 8. Full Experimental Design

### 8.1 Main Experiment: Three-Way Scale and Architecture Sweep

Run the full pipeline across all 3 conditions on all 3 evaluation datasets:

- **Condition A:** Original learner text (1 per dataset) — GEC + ERRANT only (no generation)
- **Condition B:** 11 fine-tuned AL models — full pipeline (generation + GEC + ERRANT)
- **Condition C:** 11 native pretrained baselines — full pipeline (generation + GEC + ERRANT)

Total: (1 + 11 + 11) × 3 datasets = **69 experimental conditions**

Per condition: error type distribution vectors, PPL statistics (B and C only), regional breakdowns.

Key analyses:
- JSD(AL ‖ Learner) and JSD(Native ‖ Learner) for every model — does fine-tuning close the gap?
- Spearman rho(AL, Learner) — do AL models rank error types in the same order as real learners?
- Convergence ratio per model family and scale
- Per-error-type comparison: which ERRANT categories does the AL model match the learner distribution in, and which does it miss?

### 8.2 Ablation: GEC Backend

Run the full pipeline using both GEC backends:
- `grammarly/coedit-large` (dedicated seq2seq)
- `google/gemma-2-2b-it` (instruction-tuned LLM)

Compare whether ERRANT error distributions — and critically, the three-way JSD rankings — are robust to GEC model choice.

### 8.3 Ablation: CEFR Stratification (Developmental Cline)

For each model family, fine-tune separate models on CEFR-stratified subsets:
- A1-A2 (beginner), B1-B2 (intermediate), C1 (advanced)
- Compare against CEFR-stratified human learner text at each level
- Test whether error-type distributions shift monotonically across levels for both human and AL text
- Expected: More M:DET and R:VERB:SVA errors for lower CEFR, fewer for higher — in both conditions

### 8.4 Ablation: L1-Stratified Fine-Tuning

Fine-tune models on L1-stratified EFCAMDAT subsets:
- Arabic, Chinese, French, Japanese, Korean, Spanish
- Compare each L1-stratified AL model against L1-matched human learner text
- Test whether L1-specific transfer effects emerge:
  - Elevated M:DET for L1-Arabic/Chinese (languages lacking articles)
  - Elevated M:PRON for L1-Spanish/Japanese (pro-drop languages)

### 8.5 Control Experiments

1. **Prompt vs. generation region:** Analyze whether model-introduced errors (generation region) show stronger convergence to the human learner profile than pre-existing errors (prompt region)
2. **Random degradation baseline:** Add noise to native model outputs to verify that random degradation does NOT produce learner-like error distributions (high JSD to learner, unlike AL models)

---

## 9. Expected Contributions

1. **Methodological:** A reusable, open-source generation-correction-annotation pipeline for evaluating language model error profiles against human learner ground-truth distributions using ERRANT's linguistically motivated taxonomy
2. **Empirical:** Large-scale three-way comparison (human learner vs. AL model vs. native baseline) across 3 model families and 8+ scales, measuring distributional convergence via JSD and Spearman correlation
3. **Theoretical:** Connection between distributional learning in LMs and SLA theories of interlanguage — error-tag distributions as a new cognitive diagnostic for language models, validated against real learner data
4. **Applied:** Foundations for error-aware pedagogical NLP systems that model specific learner populations, calibrated against empirical learner error profiles

---

## 10. Codebase Reference

### 10.1 Repository Structure

```
src/gen_gec_errant/
  _types.py             # Core data types (ErrorAnnotation, SentenceAnnotation)
  _config_utils.py      # YAML loading, CLI overrides
  registry.py           # Model/dataset/path registry (11 models, 3 datasets)
  data_loader/          # Stage 1: load + split sentences
  generation/           # Stage 2: generate continuations + compute PPL
  gec/                  # Stage 3: grammatical error correction
  annotation/           # Stage 4: ERRANT error annotation
  analysis/             # Stage 5: statistics, comparisons, plots, CSVs
  pipeline/             # Orchestrator for all 5 stages

configs/
  pipeline/             # Full pipeline configs
    per-model/          # 35+ model-specific YAML configs
  generation/           # Generation-only configs
  gec/                  # GEC model configs
  annotation/           # ERRANT configs
  analysis/             # Analysis configs

reproducibility/        # 11 self-contained experiment directories
  paper-reproducibility-ft-{model}/
    plan/               # Goals, steps
    scripts/            # Automated orchestrator
    experiment/         # Results per dataset
```

### 10.2 Running the Pipeline

```bash
# Single model
python -m gen_gec_errant.pipeline --config configs/pipeline/per-model/ft-gpt2-large.yaml

# Batch run all models
bash scripts/run_pipeline.sh --configs-dir configs/pipeline/per-model --output-root outputs/full-run

# Python API
from gen_gec_errant.pipeline import run_pipeline
from gen_gec_errant.pipeline.config import load_config_from_yaml
config = load_config_from_yaml("configs/pipeline/quickstart.yaml")
summaries, comparison = run_pipeline(config)
```

### 10.3 Key Dependencies

```
torch>=2.0, transformers>=4.35, errant>=3.0, spacy>=3.5
numpy>=1.24, scipy>=1.10, matplotlib>=3.7, pandas>=2.0, pyyaml>=6.0
```

---

## 11. Key References

### SLA and Error Analysis
- Bryant, C., Felice, M., & Briscoe, T. (2017). Automatic Annotation and Evaluation of Error Types for GEC. ACL. [ERRANT]
- Goldschneider, J. M., & DeKeyser, R. M. (2001). Explaining the natural order of L2 morpheme acquisition in English. Language Learning.
- Pienemann, M. (1998). Language Processing and SLA: Processability Theory. John Benjamins.
- Selinker, L. (1972). Interlanguage. International Review of Applied Linguistics.
- Ionin, T., Ko, H., & Wexler, K. (2004). Article semantics in L2 acquisition. Language Acquisition.
- Geertzen, J., Alexopoulou, T., & Korhonen, A. (2013). Automatic linguistic annotation of large scale L2 databases: The EF-Cambridge Open Language Database. SLRF. [EFCAMDAT]

### Cognitive Modeling
- Levy, R. (2008). Expectation-based syntactic comprehension. Cognition.
- Wilcox, E., et al. (2020). On the predictive power of neural LMs for human language comprehension. CogSci.
- Warstadt, A., et al. (2020). BLiMP: A benchmark of linguistic minimal pairs. TACL.
- Schrimpf, M., et al. (2021). Neural architecture of language. PNAS.

### Models and GEC
- Radford, A., et al. (2019). Language models are unsupervised multitask learners. OpenAI. [GPT-2]
- Biderman, S., et al. (2023). Pythia: A suite for analyzing LLMs. ICML.
- Raheja, V., et al. (2023). CoEdIT: Text editing by task-specific instruction tuning. EMNLP.

---

## 12. AutoResearchClaw Configuration

```yaml
project:
  name: "artificial-learner-error-distributions"
  mode: "full-auto"

research:
  topic: >
    Do language models fine-tuned on L2 learner corpora (EFCAMDAT) produce error
    distributions that converge toward the empirical error profile of the original
    human learner text, as measured by automatic ERRANT error tagging? Compare three
    conditions — (1) original human learner text as ground-truth reference,
    (2) artificial learner models fine-tuned on EFCAMDAT, and (3) native pretrained
    baselines — across model families (GPT-2, Pythia, SmolLM2), scales (70M-1.7B
    parameters), and learner L1 backgrounds. Evaluate whether AL error-type
    distributions approximate the human learner ground truth, form developmental
    clines across CEFR proficiency levels, and capture L1-transfer effects, using a
    generation-correction-annotation pipeline with statistical divergence measures
    against empirical learner error profiles.
  domains:
    - "computational-linguistics"
    - "second-language-acquisition"
    - "cognitive-modeling"
    - "grammatical-error-correction"
  daily_paper_count: 15
  quality_threshold: 4.0
  graceful_degradation: true

experiment:
  mode: "sandbox"
  time_budget_sec: 3600
  max_iterations: 23
  metric_key: "jensen_shannon_divergence"
  metric_direction: "minimize"

export:
  target_conference: "cmcl_2026"
  authors: "Anonymous"
```
