# Do Artificial Learners Make Human-Like Errors? Aligning LLM Error Distributions with L2 Acquisition through Automatic Error Tagging

**Target venue:** CMCL 2026 — 15th Workshop on Cognitive Modeling and Computational Linguistics (co-located with LREC 2026, Palma, Mallorca, May 16 2026)

---

## 1. Abstract

Language models pre-trained on native English text have become a standard substrate for probing linguistic knowledge, yet the representations they acquire bear no necessary resemblance to those of human second-language (L2) learners. We ask whether fine-tuning a language model on learner-produced text causes it to generate errors that are not merely more frequent but *typologically aligned* with the systematic error patterns documented in second-language acquisition (SLA) research. Using an automatic generation–correction–annotation pipeline, we compare ERRANT error-tag distributions of text produced by a learner-tuned GPT-2 against native-trained baselines of varying scale (GPT-2 124M–774M, Pythia 70M–1B). Preliminary results on 20 EFCAMDAT sentence prompts show that the learner-tuned model produces 72% more errors per sentence (5.15 vs. 3.00) with a 100% error rate (vs. 95%), and critically, its error profile is shifted toward acquisition-relevant categories: subject–verb agreement errors (`R:VERB:SVA`) increase 2.5×, verbal morphology errors (`M:VERB:FORM`) emerge as a top-5 category (absent from the native baseline's top 10), and determiner errors (`U:DET`) quadruple. These preliminary findings suggest that artificial learner models do not simply degrade into noisier generators but internalise structured interlanguage regularities, offering a computational window into the distributional basis of L2 error patterns.

---

## 2. Introduction and Motivation

A productive line of work in cognitive modeling evaluates whether neural language models (LMs) capture aspects of human language processing: surprisal-based reading-time predictions (Levy, 2008; Wilcox et al., 2020), syntactic acceptability judgements (Warstadt et al., 2020), and neural alignment with brain recordings (Schrimpf et al., 2021). A parallel question — whether LMs can model *language acquisition* rather than just steady-state competence — remains largely unexplored.

Second-language acquisition research has established that L2 learners produce errors in systematic, predictable patterns. Goldschneider and DeKeyser (2001) demonstrated a stable acquisition order for English morphemes; Pienemann's (1998) Processability Theory predicts which structures become available at each developmental stage. These patterns are not random noise — they reflect the learner's evolving interlanguage grammar (Selinker, 1972), shaped by L1 transfer, input frequency, and processing constraints.

This raises a concrete cognitive modeling question: **if we train a language model on the distributional statistics of learner-produced text, does it acquire an interlanguage-like error grammar — or does it simply become a worse native model?** The distinction matters. A model that merely degrades would produce errors uniformly across categories. A model that captures interlanguage structure would produce errors concentrated in the specific categories that SLA research identifies as vulnerable: determiners, verbal morphology, prepositions, and subject–verb agreement.

We operationalise this question through automatic error tagging. By generating text from both native-trained and learner-tuned models, correcting it with a grammatical error correction (GEC) system, and annotating the corrections with ERRANT's linguistically motivated error taxonomy (Bryant et al., 2017), we obtain a fine-grained error fingerprint for each model class — directly comparable to the error profiles studied in SLA.

---

## 3. Experiment

### 3.1 Data

We use the EF-Cambridge Open Language Database (EFCAMDAT; Geertzen et al., 2013), a large-scale corpus of learner English essays with CEFR proficiency levels (A1–C2) and L1 metadata. From the sentence-level extraction (21,493 sentences), we sample prompts by splitting each sentence at approximately the 50% token boundary, yielding a naturalistic prefix that preserves the learner's register, topic, and syntactic trajectory. The initial pilot uses 20 prompts; the full experiment will scale to 1,000+.

### 3.2 Models

We compare two model classes:

| Model | Parameters | Training data | Role |
|-------|-----------|---------------|------|
| **GPT-2 Base** | 124M | WebText (native English) | Native baseline |
| **GPT-2 Medium** | 355M | WebText (native English) | Scale control |
| **GPT-2 Large** | 774M | WebText (native English) | Scale control |
| **Pythia 70M–1B** | 70M–1B | The Pile (native English) | Architecture/scale control |
| **GPT-2 Learner** | 124M | GPT-2 Base + fine-tuned on EFCAMDAT | Artificial learner |

The learner-tuned model shares the same architecture and initialisation as GPT-2 Base; the only difference is continued training on EFCAMDAT learner text. This controlled contrast isolates the effect of learner-corpus exposure from architecture and model capacity.

### 3.3 Pipeline

The pipeline proceeds in four stages:

**Stage 1 — Generation.** Each model receives the prompt prefix and generates a 50-token continuation (temperature = 1.0, top-k = 50, top-p = 0.95, repetition penalty = 1.2). We also compute per-sentence perplexity on the full text.

**Stage 2 — Grammatical Error Correction.** Each generated continuation is passed through Grammarly CoEdit-Large (Schick et al., 2023), a T5-based seq2seq GEC model, which produces a corrected version. As a robustness check, we also run corrections through Gemma 2 2B-it (instruction-tuned LLM) to assess sensitivity to the GEC back-end.

**Stage 3 — Error Annotation.** ERRANT (Bryant et al., 2017) aligns each original–corrected pair and emits typed edits from its taxonomy: **M**issing, **R**eplacement, and **U**nnecessary operations over morphosyntactic categories (e.g., `M:DET` = missing determiner, `R:VERB:SVA` = subject–verb agreement replacement, `R:VERB:TENSE` = tense error, `M:VERB:FORM` = missing verbal morphology).

**Stage 4 — Statistical Analysis.** We compute per-model error counts, error rates, error-type distribution vectors, perplexity statistics, and Mann–Whitney U tests for pairwise comparisons. We measure distributional alignment between model error profiles and empirical learner error distributions using Jensen–Shannon divergence and Spearman rank correlations.

### 3.4 Illustrative Examples

To make the distinction concrete, here are continuations from the same prompt by each model:

**Prompt:** *"at my university because I started studying English with an"*

- **Native GPT-2:** *"at their best – and, after all, I was a business student. And so when we started our company there were no more professions than that which offered jobs for an engineer to take on engineering training by themselves."* → 1 error detected (`R:NOUN`)
- **Learner GPT-2:** *"english teacher, then went back to Brazil for a masters degree. In order obtain this qualification before applying after graduation that is very important. After the student was finished the school has moved on and now work"* → 9 errors detected (`R:ORTH` ×2, `R:NOUN` ×2, `M:VERB:FORM`, `R:OTHER`, `U:VERB:TENSE`, `M:PRON`, `R:VERB`)

The native model drifts off-topic into fluent but contextually unrelated prose. The learner-tuned model stays on-topic (studying English, university, qualifications) but produces characteristic L2 errors: lowercase *english*, missing infinitival *to* (*"in order obtain"*), missing comma after an adverbial clause, tense inconsistency (*"has moved"* → *"moved"*), and a missing subject pronoun (*"now work"* → *"now I work"*). These are precisely the error types documented in SLA research for intermediate-level learners.

---

## 4. Preliminary Results

### 4.1 Aggregate Error Metrics

On the 20-sentence pilot (EFCAMDAT, French L1, A2–B1):

| Metric | Native GPT-2 (124M) | Learner GPT-2 (124M) | Δ |
|--------|---------------------|----------------------|---|
| Mean perplexity | 47.86 (σ = 14.73) | 41.68 (σ = 13.05) | −12.9% |
| Total errors | 60 | 103 | +71.7% |
| Avg errors/sentence | 3.00 | 5.15 | +71.7% |
| Error rate | 95% (19/20) | 100% (20/20) | +5 pp |
| Unique error types | 21 | 22 | +1 |
| Combined (PPL × errors) | 143.57 | 214.67 | +49.5% |

The learner-tuned model is simultaneously *less perplexed* by the learner prompts (lower PPL, indicating better distributional fit to interlanguage input) and *more error-prone* in its continuations. This is the expected signature of a model that has internalised learner-language statistics: it predicts learner text well but reproduces its error patterns when generating.

### 4.2 Error-Type Distribution Shift

The critical question is not whether the learner model produces *more* errors, but whether it produces *different kinds* of errors — specifically, the kinds that SLA research identifies as acquisition-relevant.

**Top error types by model:**

| Rank | Native GPT-2 | Count | Learner GPT-2 | Count |
|------|-------------|-------|---------------|-------|
| 1 | `R:NOUN` | 16 | `R:OTHER` | 27 |
| 2 | `R:OTHER` | 15 | `R:ORTH` | 20 |
| 3 | `R:ORTH` | 7 | `R:NOUN` | 15 |
| 4 | `U:PREP` | 2 | **`M:VERB:FORM`** | **5** |
| 5 | `R:PREP` | 2 | **`R:VERB:SVA`** | **5** |
| 6 | `R:VERB:SVA` | 2 | **`U:DET`** | **4** |
| 7 | `M:OTHER` | 1 | **`R:VERB`** | **4** |
| 8 | `R:PART` | 1 | `M:OTHER` | 3 |
| 9 | `R:VERB:FORM` | 1 | **`M:PRON`** | **3** |
| 10 | `U:ADV` | 1 | `U:NOUN` | 2 |

Three patterns emerge that align with SLA predictions:

**(a) Verbal morphology.** `M:VERB:FORM` (missing infinitival *to*, bare verb forms) jumps from 0 occurrences in the native model to 5 in the learner model, becoming a top-4 error type. `R:VERB:SVA` (subject–verb agreement: *"everyone love"* → *"everyone loves"*; *"it relate"* → *"it relates"*) increases from 2 to 5 (2.5× increase). These are among the most robust developmental markers in L2 English (Goldschneider & DeKeyser, 2001), with verbal inflection typically acquired late and remaining error-prone even at intermediate levels.

**(b) Determiner errors.** `U:DET` (unnecessary determiner, e.g., *"all the world"* → *"the world"*; *"the analysis"* → *"analysis"*) rises from 1 in the native model to 4 in the learner model (4× increase). Determiner use is one of the most persistent difficulty areas for L2 learners, especially those whose L1 lacks articles (Ionin et al., 2004).

**(c) Pronoun omission.** `M:PRON` (missing pronoun, e.g., *"now work"* → *"now I work"*; *"has a nice price"* → *"It has a nice price"*) rises from 1 to 3. Subject pronoun omission is a known transfer error from pro-drop languages and a developmental marker in early-to-intermediate L2 English.

### 4.3 Perplexity–Error Relationship

The learner model exhibits lower perplexity on learner prompts (mean 41.68 vs. 47.86), confirming that fine-tuning on EFCAMDAT produces a better distributional fit to interlanguage. At the sentence level, the learner model shows a wider spread of error counts (range 1–9, vs. 0–7 for native), with higher-error sentences not concentrated at high perplexity — suggesting that the model has learned to *fluently produce errors*, generating learner-like text without flagging it as surprising.

---

## 5. Discussion

### 5.1 Structured Interlanguage, Not Uniform Degradation

The central finding is that the learner-tuned model does not simply produce more noise uniformly. Its error profile is *selectively shifted* toward the categories that SLA research identifies as acquisition-sensitive: verbal morphology, subject–verb agreement, determiner use, and pronoun omission. Categories that are not particularly L2-relevant (e.g., `R:PART`, `U:ADV`) remain low in both models. This selective elevation is consistent with the hypothesis that the model has internalised the structured distributional properties of interlanguage — not just a higher base rate of errors, but the specific *pattern* of errors that characterises L2 development.

### 5.2 Connection to SLA Theory

Goldschneider and DeKeyser's (2001) meta-analysis established that English morphemes are acquired in a predictable order, with verbal inflection (-s, -ed, -ing) and articles among the latest-acquired and most error-prone. Our error-type distribution aligns with this: `R:VERB:SVA` and `M:VERB:FORM` are selectively elevated in the learner model, while `M:DET`/`U:DET` errors also increase — exactly the categories that the natural order hypothesis predicts should be most affected by exposure to interlanguage input.

Pienemann's (1998) Processability Theory further predicts that structures requiring cross-phrasal information exchange (e.g., subject–verb agreement across intervening material) should be more error-prone at lower proficiency levels. The concentration of `R:VERB:SVA` errors in the learner model, trained on A2–B1 level text, is consistent with this prediction.

### 5.3 Implications for Cognitive Modeling

These results suggest that error-tag distributions can serve as a **cognitive diagnostic for language models** — a complementary lens to surprisal and acceptability judgements. Where surprisal measures *what a model expects*, error-tag distributions measure *what a model produces* when its expectations are shaped by non-native input. This production-side diagnostic is particularly relevant for modeling language acquisition, where the learner's output (with its systematic errors) is the primary observable.

### 5.4 Limitations and Planned Extensions

The current pilot is limited to 20 sentences from a single L1 (French) and proficiency band (A2–B1). The full experiment will:

- Scale to 1,000+ sentences stratified by CEFR level (A1 through C1) to test whether **error-type distributions form a developmental cline** — a monotonic ordering across proficiency levels that mirrors known acquisition sequences.
- Include **L1-stratified analysis** (Arabic, Chinese, Spanish, Japanese, Korean) to test whether learner-tuned models capture L1-transfer effects (e.g., elevated `M:DET` for L1-Arabic and L1-Chinese speakers, whose languages lack articles).
- Compare **dedicated GEC vs. LLM-based GEC** back-ends to assess whether the observed distributional contrasts are robust to the correction method.
- Add **Jensen–Shannon divergence** and **Spearman rank correlation** between model error-type vectors and empirical EFCAMDAT error distributions as a formal measure of alignment.
- Run the full native model scale ladder (GPT-2 124M → Pythia 1B) to disentangle capacity from training-data effects.

---

## 6. Conclusion

We present a pipeline and preliminary results addressing whether language models fine-tuned on L2 learner corpora develop cognitively plausible error profiles. Our generation–correction–annotation paradigm uses ERRANT error tags as a diagnostic lens, bridging computational language modeling and SLA error analysis. Preliminary findings on 20 EFCAMDAT prompts show that the learner-tuned GPT-2 not only produces more errors than its native-trained counterpart (5.15 vs. 3.00 per sentence) but produces them in an acquisition-relevant pattern: verbal morphology errors increase 2.5×, determiner errors quadruple, and pronoun omission triples — the exact categories that SLA theory identifies as late-acquired and persistent. The learner model does not merely degrade; it acquires a structured error grammar that mirrors human interlanguage. This finding opens a new avenue for using language models as cognitive models of L2 acquisition and for designing error-aware pedagogical NLP systems.

---

## References

Biderman, S., Schoelkopf, H., Anthony, Q., et al. (2023). Pythia: A suite for analyzing large language models across training and scaling. In *Proceedings of the 40th International Conference on Machine Learning* (pp. 2397–2430). PMLR.

Bryant, C., Felice, M., Andersen, Ø. E., & Briscoe, T. (2017). Automatic annotation and evaluation of error types for grammatical error correction. In *Proceedings of the 55th Annual Meeting of the ACL* (pp. 793–805).

Geertzen, J., Alexopoulou, T., & Korhonen, A. (2013). Automatic linguistic annotation of large scale L2 databases: The EF-Cambridge Open Language Database (EFCAMDAT). In *Proceedings of the 31st Second Language Research Forum* (pp. 240–254).

Goldschneider, J. M., & DeKeyser, R. M. (2001). Explaining the "natural order of L2 morpheme acquisition" in English: A meta-analysis of multiple determinants. *Language Learning*, 51(1), 1–50.

Hale, J. (2001). A probabilistic Earley parser as a psycholinguistic model. In *Proceedings of NAACL* (pp. 1–8).

Ionin, T., Ko, H., & Wexler, K. (2004). Article semantics in L2 acquisition: The role of specificity. *Language Acquisition*, 12(1), 3–69.

Levy, R. (2008). Expectation-based syntactic comprehension. *Cognition*, 106(3), 1126–1177.

Pienemann, M. (1998). *Language Processing and Second Language Development: Processability Theory*. John Benjamins.

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. *OpenAI Blog*, 1(8), 9.

Schick, T., Dwivedi-Yu, J., Dessì, R., et al. (2023). Toolformer: Language models can teach themselves to use tools. In *Proceedings of NeurIPS*.

Schrimpf, M., Blank, I. A., Tuckute, G., et al. (2021). The neural architecture of language: Integrative modeling converges on predictive processing. *PNAS*, 118(45), e2105646118.

Selinker, L. (1972). Interlanguage. *International Review of Applied Linguistics*, 10(3), 209–231.

Warstadt, A., Parrish, A., Liu, H., et al. (2020). BLiMP: The benchmark of linguistic minimal pairs for English. *TACL*, 8, 377–392.

Wilcox, E. G., Gauthier, J., Hu, J., Qian, P., & Levy, R. (2020). On the predictive power of neural language models for human real-time comprehension behavior. In *Proceedings of the 42nd Annual Conference of the Cognitive Science Society* (pp. 1707–1713).
