#!/usr/bin/env python3
"""
Main pipeline runner: L2 Learner Error Analysis.

End-to-end pipeline:
  1. Load EFCAMDAT sentences
  2. For each model (GPT-2 base/learner, Pythia variants):
     a. Generate continuations from sentence prompts
     b. Compute perplexity of generated text
  3. Run GEC on all generated text
  4. Annotate errors with ERRANT
  5. Analyze and compare models
  6. Generate plots and export results

Usage:
    python run_pipeline.py --data_path data/efcamdat.txt \
                           --learner_checkpoint path/to/learner.pt \
                           --output_dir results/ \
                           --max_sentences 500 \
                           --models gpt2-base gpt2-learner pythia-70m
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

# Pipeline modules
from config import PipelineConfig, ModelConfig, GenerationConfig, GECConfig
from data_loader import load_sentences, make_prompts
from generation import get_device, load_model, generate_continuations, compute_perplexity
from gec import load_gec_corrector
from error_annotation import ERRANTAnnotator, summarize_errors
from analysis import (
    compute_model_summary,
    compare_models,
    save_results,
    generate_all_plots,
)
from csv_export import export_csv, export_errors_long_format

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("pipeline")


###############################################################################
# CLI
###############################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="L2 Learner Error Analysis Pipeline")

    # Data
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to input sentences file (.txt, .csv, .tsv)")
    parser.add_argument("--max_sentences", type=int, default=None,
                        help="Max sentences to process (default: all)")

    # Models
    parser.add_argument("--learner_checkpoint", type=str, default=None,
                        help="Path to .pt checkpoint for learner-tuned GPT-2")
    parser.add_argument("--models", nargs="+", default=["gpt2-base", "gpt2-learner"],
                        help="Which models to run. Options: gpt2-base, gpt2-medium, "
                             "gpt2-learner, pythia-70m, pythia-160m, pythia-410m")

    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--prompt_ratio", type=float, default=0.5,
                        help="Fraction of sentence words to use as prompt")

    # GEC
    parser.add_argument("--gec_method", type=str, default="dedicated",
                        choices=["llm", "dedicated"],
                        help="GEC method: 'dedicated' (coedit) or 'llm' (gemma)")
    parser.add_argument("--gec_model", type=str, default=None,
                        help="Override GEC model ID")

    # Output
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)

    # Flags
    parser.add_argument("--skip_generation", action="store_true",
                        help="Skip generation, load from existing outputs")
    parser.add_argument("--skip_gec", action="store_true",
                        help="Skip generation AND GEC, re-run ERRANT + analysis only")
    parser.add_argument("--skip_plots", action="store_true",
                        help="Skip plot generation")

    # Data filtering
    parser.add_argument("--min_words", type=int, default=10,
                        help="Min words per text (default: 10)")
    parser.add_argument("--max_words", type=int, default=10000,
                        help="Max words per text (default: 10000)")

    return parser.parse_args()


###############################################################################
# Model registry
###############################################################################

MODEL_REGISTRY = {
    # Native HF models (local copies on remote)
    "gpt2-base": ModelConfig(name="gpt2-base", hf_model_id="/workspace/models/gpt2", model_family="gpt2"),
    "gpt2-medium": ModelConfig(name="gpt2-medium", hf_model_id="/workspace/models/gpt2-medium", model_family="gpt2"),
    "gpt2-large": ModelConfig(name="gpt2-large", hf_model_id="/workspace/models/gpt2-large", model_family="gpt2"),
    # Artificial learner models (fine-tuned on EFCAMDAT)
    "gpt2-small-all-data": ModelConfig(
        name="gpt2-small-all-data",
        hf_model_id="/workspace/models/gpt2-small-all-data/final",
        model_family="gpt2",
    ),
    "gpt2-medium-all-data": ModelConfig(
        name="gpt2-medium-all-data",
        hf_model_id="/workspace/models/gpt2-medium-all-data/final",
        model_family="gpt2",
    ),
    "gpt2-large-all-data": ModelConfig(
        name="gpt2-large-all-data",
        hf_model_id="/workspace/models/gpt2-large-all-data/final",
        model_family="gpt2",
    ),
    # Legacy entries
    "gpt2-learner": ModelConfig(
        name="gpt2-learner", hf_model_id="gpt2", model_family="gpt2",
        is_learner_tuned=True, checkpoint_path="SET_AT_RUNTIME",
    ),
    "pythia-70m": ModelConfig(name="pythia-70m", hf_model_id="EleutherAI/pythia-70m", model_family="pythia"),
    "pythia-160m": ModelConfig(name="pythia-160m", hf_model_id="EleutherAI/pythia-160m", model_family="pythia"),
    "pythia-410m": ModelConfig(name="pythia-410m", hf_model_id="EleutherAI/pythia-410m", model_family="pythia"),
    "pythia-1b": ModelConfig(name="pythia-1b", hf_model_id="EleutherAI/pythia-1b", model_family="pythia"),
}


###############################################################################
# Pipeline steps
###############################################################################

def step_1_load_data(args) -> list:
    """Load and prepare data."""
    logger.info("=" * 60)
    logger.info("STEP 1: Loading data")
    logger.info("=" * 60)

    sentences = load_sentences(args.data_path, max_sentences=args.max_sentences,
                               min_words=args.min_words, max_words=args.max_words)
    items = make_prompts(sentences, prompt_ratio=args.prompt_ratio)

    logger.info(f"Prepared {len(items)} prompt-reference pairs")
    logger.info(f"  Example prompt:    '{items[0]['prompt']}'")
    logger.info(f"  Example reference: '{items[0]['reference']}'")

    return items


def step_2_generate(args, items, device) -> dict:
    """Generate continuations for all models."""
    logger.info("=" * 60)
    logger.info("STEP 2: Generating text with models")
    logger.info("=" * 60)

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    prompts = [item["prompt"] for item in items]
    all_results = {}  # model_name -> {"continuations": [...], "perplexities": [...]}

    for model_name in args.models:
        if model_name not in MODEL_REGISTRY:
            logger.warning(f"Unknown model: {model_name}, skipping")
            continue

        model_config = MODEL_REGISTRY[model_name]

        # Set learner checkpoint path
        if model_config.is_learner_tuned:
            if not args.learner_checkpoint:
                logger.error("--learner_checkpoint is required for gpt2-learner")
                continue
            model_config.checkpoint_path = args.learner_checkpoint

        t0 = time.time()

        # Load model
        model, tokenizer = load_model(model_config, device)

        # Generate
        logger.info(f"Generating with {model_name}...")
        continuations = generate_continuations(
            model, tokenizer, prompts, gen_config,
            batch_size=args.batch_size, device=device,
        )

        # Compute perplexity on the full generated sentences (prompt + continuation)
        full_texts = [
            f"{prompt} {cont}" for prompt, cont in zip(prompts, continuations)
        ]
        logger.info(f"Computing perplexity for {model_name}...")
        perplexities = compute_perplexity(
            model, tokenizer, full_texts,
            batch_size=args.batch_size, device=device,
        )

        elapsed = time.time() - t0
        logger.info(f"  {model_name}: generated {len(continuations)} sentences in {elapsed:.1f}s")
        logger.info(f"  Mean perplexity: {sum(perplexities)/len(perplexities):.2f}")

        all_results[model_name] = {
            "continuations": continuations,
            "full_texts": full_texts,
            "perplexities": perplexities,
        }

        # Free GPU memory
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    return all_results


def step_3_gec(args, all_results, device) -> dict:
    """Run grammatical error correction on all generated text."""
    logger.info("=" * 60)
    logger.info("STEP 3: Grammatical Error Correction")
    logger.info("=" * 60)

    gec_config = GECConfig()
    if args.gec_model:
        if args.gec_method == "llm":
            gec_config.llm_model_id = args.gec_model
        else:
            gec_config.dedicated_gec_model_id = args.gec_model

    corrector = load_gec_corrector(gec_config, device, method=args.gec_method)

    GEC_BATCH = 32

    for model_name, results in all_results.items():
        logger.info(f"Correcting {model_name} outputs...")
        t0 = time.time()

        # Batch GEC to avoid OOM with padded tensors
        corrected = []
        for i in range(0, len(results["continuations"]), GEC_BATCH):
            batch = results["continuations"][i:i + GEC_BATCH]
            corrected.extend(corrector.correct(batch))
        results["corrected_continuations"] = corrected

        corrected_full = []
        for i in range(0, len(results["full_texts"]), GEC_BATCH):
            batch = results["full_texts"][i:i + GEC_BATCH]
            corrected_full.extend(corrector.correct(batch))
        results["corrected_full_texts"] = corrected_full

        elapsed = time.time() - t0
        logger.info(f"  {model_name}: corrected {len(corrected)} sentences in {elapsed:.1f}s")

    # Free GEC model
    del corrector
    torch.cuda.empty_cache() if device.type == "cuda" else None

    return all_results


def step_4_annotate(all_results) -> dict:
    """Run ERRANT annotation on original vs corrected pairs."""
    logger.info("=" * 60)
    logger.info("STEP 4: ERRANT Error Annotation")
    logger.info("=" * 60)

    annotator = ERRANTAnnotator(lang="en")

    for model_name, results in all_results.items():
        logger.info(f"Annotating errors for {model_name}...")

        # Annotate continuations (generated part only)
        annotations = annotator.annotate_batch(
            results["continuations"],
            results["corrected_continuations"],
        )
        results["annotations"] = annotations

        # Summarize
        summary = summarize_errors(annotations)
        results["error_summary"] = summary

        logger.info(f"  {model_name}: {summary['total_errors']} errors in "
                     f"{summary['total_sentences']} sentences "
                     f"(rate: {summary['error_rate']:.3f})")
        if summary["top_10_error_types"]:
            logger.info(f"  Top errors: {summary['top_10_error_types'][:5]}")

    return all_results


def step_5_analyze(args, all_results, items):
    """Compute summaries, comparisons, and generate plots."""
    logger.info("=" * 60)
    logger.info("STEP 5: Analysis & Visualization")
    logger.info("=" * 60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute per-model summaries
    summaries = []
    for model_name, results in all_results.items():
        summary = compute_model_summary(
            model_name=model_name,
            perplexities=results["perplexities"],
            error_summaries=results["error_summary"],
            annotations=results["annotations"],
        )
        summaries.append(summary)

    # Compare models
    comparison = compare_models(summaries)

    # Print comparison table
    logger.info("\n" + "=" * 80)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 80)
    logger.info(f"{'Model':<20} {'PPL Mean':>10} {'PPL Std':>10} {'Err Rate':>10} "
                f"{'Avg Err':>10} {'PPL×Err':>12}")
    logger.info("-" * 80)
    for s in summaries:
        logger.info(f"{s['model_name']:<20} {s['ppl_mean']:>10.2f} {s['ppl_std']:>10.2f} "
                     f"{s['error_rate']:>10.3f} {s['avg_errors_per_sentence']:>10.3f} "
                     f"{s['ppl_x_errors']:>12.2f}")
    logger.info("=" * 80)

    # Pairwise tests
    for test in comparison.get("pairwise_tests", []):
        logger.info(f"\n  {test['model_a']} vs {test['model_b']}:")
        logger.info(f"    PPL:    U={test['ppl_test_stat']}, p={test['ppl_p_value']}")
        logger.info(f"    Errors: U={test['error_test_stat']}, p={test['error_p_value']}")

    # Save JSON results
    save_results(summaries, comparison, args.output_dir)

    # Save raw data for reproducibility
    raw_output = {}
    for model_name, results in all_results.items():
        raw_output[model_name] = {
            "continuations": results["continuations"],
            "full_texts": results["full_texts"],
            "corrected_continuations": results["corrected_continuations"],
            "corrected_full_texts": results.get("corrected_full_texts", []),
            "perplexities": results["perplexities"],
            "error_summary": results["error_summary"],
            # Annotations serialized
            "annotations": [
                {
                    "original": a.original,
                    "corrected": a.corrected,
                    "num_errors": a.num_errors,
                    "error_types": a.error_type_counts,
                    "errors": [
                        {
                            "orig_tokens": e.original_tokens,
                            "corr_tokens": e.corrected_tokens,
                            "type": e.error_type,
                        }
                        for e in a.errors
                    ],
                }
                for a in results["annotations"]
            ],
        }

    with open(output_dir / "raw_results.json", "w") as f:
        json.dump(raw_output, f, indent=2, default=str)
    logger.info(f"Saved raw results to {output_dir / 'raw_results.json'}")

    # Save prompts and references
    with open(output_dir / "prompts.json", "w") as f:
        json.dump(items, f, indent=2)

    # Generate plots
    if not args.skip_plots:
        try:
            generate_all_plots(summaries, str(output_dir / "plots"))
            logger.info("All plots generated successfully")
        except Exception as e:
            logger.warning(f"Plot generation failed: {e}")

    # Export comprehensive CSV
    logger.info("Exporting CSV files...")
    model_names = list(all_results.keys())

    csv_path = export_csv(
        items=items,
        all_results=all_results,
        model_names=model_names,
        output_path=str(output_dir / "full_results.csv"),
    )
    logger.info(f"Main CSV: {csv_path}")

    errors_csv_path = export_errors_long_format(
        all_results=all_results,
        model_names=model_names,
        output_path=str(output_dir / "errors_long_format.csv"),
    )
    logger.info(f"Errors CSV (long format): {errors_csv_path}")

    return summaries, comparison


###############################################################################
# Main
###############################################################################

def _load_raw_results(output_dir: str, items: list) -> dict:
    """Load raw_results.json and reconstruct missing keys for backwards compat."""
    raw_path = Path(output_dir) / "raw_results.json"
    logger.info(f"Loading existing results from {raw_path}")
    with open(raw_path) as f:
        all_results = json.load(f)

    prompts = [item["prompt"] for item in items]

    for model_name, results in all_results.items():
        # Reconstruct full_texts if missing (old format)
        if "full_texts" not in results:
            results["full_texts"] = [
                f"{p} {c}" for p, c in zip(prompts, results["continuations"])
            ]
            logger.info(f"  {model_name}: reconstructed full_texts from prompts + continuations")

        # Reconstruct corrected_full_texts if missing
        if "corrected_full_texts" not in results:
            results["corrected_full_texts"] = []

    return all_results


def main():
    args = parse_args()

    # --skip_gec implies --skip_generation
    if args.skip_gec:
        args.skip_generation = True

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    # Step 1: Load data
    items = step_1_load_data(args)

    if not args.skip_generation:
        # Step 2: Generate
        all_results = step_2_generate(args, items, device)

        # Step 3: GEC
        all_results = step_3_gec(args, all_results, device)
    else:
        # Load from existing outputs
        all_results = _load_raw_results(args.output_dir, items)

        if not args.skip_gec:
            # Re-run GEC on the loaded continuations (generation done, GEC crashed)
            logger.info("Re-running GEC on loaded results (--skip_generation without --skip_gec)")
            all_results = step_3_gec(args, all_results, device)

    # Step 4: Annotate
    all_results = step_4_annotate(all_results)

    # Step 5: Analyze
    summaries, comparison = step_5_analyze(args, all_results, items)

    logger.info("\nPipeline complete!")
    return summaries, comparison


if __name__ == "__main__":
    main()
