#!/usr/bin/env python3
"""
Resumable pipeline runner: L2 Learner Error Analysis.

Same pipeline as run_pipeline.py but with batch-level checkpointing.
Saves progress after every N sentences so a crash loses at most one batch.

Checkpoint layout inside output_dir/:
  checkpoint/
    state.json              — overall progress tracker
    gen_<model>_batch<N>.json   — generation + perplexity results per batch
    gec_<model>_batch<N>.json   — GEC correction results per batch

On resume, completed batches are skipped and partial results are loaded
from the checkpoint files before continuing.

Usage:
    python run_pipeline_with_resume.py --data_path data/efcamdat.txt \
                                       --output_dir results/ \
                                       --models gpt2-base gpt2-small-all-data \
                                       --checkpoint_every 50
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch

# Pipeline modules (unchanged — no modifications to these files)
from config import PipelineConfig, ModelConfig, GenerationConfig, GECConfig
from data_loader import load_sentences, make_prompts
from generation import get_device, load_model, generate_continuations, compute_perplexity
from gec import load_gec_corrector
from error_annotation import ERRANTAnnotator, SentenceAnnotation, ErrorAnnotation, summarize_errors
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
logger = logging.getLogger("pipeline-resume")


###############################################################################
# Model registry (duplicated from run_pipeline.py to avoid modifying it)
###############################################################################

MODEL_REGISTRY = {
    # Native HF models (downloaded to local folders)
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
# Checkpoint helpers
###############################################################################

def _ckpt_dir(output_dir: str) -> Path:
    d = Path(output_dir) / "checkpoint"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_state(output_dir: str) -> dict:
    path = _ckpt_dir(output_dir) / "state.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_state(output_dir: str, state: dict):
    path = _ckpt_dir(output_dir) / "state.json"
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def _save_batch(output_dir: str, prefix: str, model_name: str, batch_idx: int, data: dict):
    path = _ckpt_dir(output_dir) / f"{prefix}_{model_name}_batch{batch_idx:04d}.json"
    with open(path, "w") as f:
        json.dump(data, f, default=str)


def _load_batch(output_dir: str, prefix: str, model_name: str, batch_idx: int) -> Optional[dict]:
    path = _ckpt_dir(output_dir) / f"{prefix}_{model_name}_batch{batch_idx:04d}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


###############################################################################
# CLI (same args as run_pipeline.py + checkpoint_every)
###############################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="L2 Learner Error Analysis Pipeline (Resumable)")

    # Data
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--max_sentences", type=int, default=None)

    # Models
    parser.add_argument("--learner_checkpoint", type=str, default=None)
    parser.add_argument("--models", nargs="+", default=["gpt2-base", "gpt2-learner"])

    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--prompt_ratio", type=float, default=0.5)

    # GEC
    parser.add_argument("--gec_method", type=str, default="dedicated",
                        choices=["llm", "dedicated"])
    parser.add_argument("--gec_model", type=str, default=None)

    # Output
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)

    # Flags
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--skip_gec", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")

    # Resume-specific
    parser.add_argument("--checkpoint_every", type=int, default=50,
                        help="Save checkpoint every N sentences (default: 50)")

    return parser.parse_args()


###############################################################################
# Resumable step 2: Generate (batched with checkpoints)
###############################################################################

def step_2_generate_resumable(args, items, device) -> dict:
    logger.info("=" * 60)
    logger.info("STEP 2: Generating text with models (resumable)")
    logger.info("=" * 60)

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    prompts = [item["prompt"] for item in items]
    total = len(prompts)
    chunk = args.checkpoint_every
    n_batches = (total + chunk - 1) // chunk

    state = _load_state(args.output_dir)
    all_results = {}

    for model_name in args.models:
        if model_name not in MODEL_REGISTRY:
            logger.warning(f"Unknown model: {model_name}, skipping")
            continue

        gen_key = f"gen_done_{model_name}"
        if state.get(gen_key):
            logger.info(f"  {model_name}: generation already complete, loading from checkpoint")
            continuations = []
            full_texts = []
            perplexities = []
            for bi in range(n_batches):
                bd = _load_batch(args.output_dir, "gen", model_name, bi)
                continuations.extend(bd["continuations"])
                full_texts.extend(bd["full_texts"])
                perplexities.extend(bd["perplexities"])
            all_results[model_name] = {
                "continuations": continuations,
                "full_texts": full_texts,
                "perplexities": perplexities,
            }
            continue

        model_config = MODEL_REGISTRY[model_name]
        if model_config.is_learner_tuned:
            if not args.learner_checkpoint:
                logger.error("--learner_checkpoint is required for gpt2-learner")
                continue
            model_config.checkpoint_path = args.learner_checkpoint

        model, tokenizer = load_model(model_config, device)
        t0 = time.time()

        continuations = []
        full_texts = []
        perplexities = []

        for bi in range(n_batches):
            start = bi * chunk
            end = min(start + chunk, total)

            # Check if this batch is already done
            existing = _load_batch(args.output_dir, "gen", model_name, bi)
            if existing is not None:
                logger.info(f"  {model_name}: batch {bi+1}/{n_batches} [{start}:{end}] loaded from checkpoint")
                continuations.extend(existing["continuations"])
                full_texts.extend(existing["full_texts"])
                perplexities.extend(existing["perplexities"])
                continue

            batch_prompts = prompts[start:end]

            # Generate
            batch_cont = generate_continuations(
                model, tokenizer, batch_prompts, gen_config,
                batch_size=args.batch_size, device=device,
            )

            # Full texts
            batch_full = [
                f"{p} {c}" for p, c in zip(batch_prompts, batch_cont)
            ]

            # Perplexity
            batch_ppl = compute_perplexity(
                model, tokenizer, batch_full,
                batch_size=args.batch_size, device=device,
            )

            # Save batch checkpoint
            _save_batch(args.output_dir, "gen", model_name, bi, {
                "continuations": batch_cont,
                "full_texts": batch_full,
                "perplexities": batch_ppl,
            })

            continuations.extend(batch_cont)
            full_texts.extend(batch_full)
            perplexities.extend(batch_ppl)

            elapsed = time.time() - t0
            logger.info(f"  {model_name}: batch {bi+1}/{n_batches} [{start}:{end}] "
                         f"done ({len(continuations)}/{total} sentences, {elapsed:.0f}s)")

        # Mark model generation complete
        state[gen_key] = True
        _save_state(args.output_dir, state)

        elapsed = time.time() - t0
        mean_ppl = sum(perplexities) / len(perplexities)
        logger.info(f"  {model_name}: generated {len(continuations)} sentences in {elapsed:.1f}s")
        logger.info(f"  Mean perplexity: {mean_ppl:.2f}")

        all_results[model_name] = {
            "continuations": continuations,
            "full_texts": full_texts,
            "perplexities": perplexities,
        }

        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    return all_results


###############################################################################
# Resumable step 3: GEC (batched with checkpoints)
###############################################################################

def step_3_gec_resumable(args, all_results, device) -> dict:
    logger.info("=" * 60)
    logger.info("STEP 3: Grammatical Error Correction (resumable)")
    logger.info("=" * 60)

    chunk = args.checkpoint_every
    state = _load_state(args.output_dir)

    # Check if all models already done
    all_done = all(state.get(f"gec_done_{m}") for m in all_results)
    if all_done:
        logger.info("  GEC already complete for all models, loading from checkpoint")
        for model_name, results in all_results.items():
            total = len(results["continuations"])
            n_batches = (total + chunk - 1) // chunk
            corrected_cont = []
            corrected_full = []
            for bi in range(n_batches):
                bd = _load_batch(args.output_dir, "gec", model_name, bi)
                corrected_cont.extend(bd["corrected_continuations"])
                corrected_full.extend(bd["corrected_full_texts"])
            results["corrected_continuations"] = corrected_cont
            results["corrected_full_texts"] = corrected_full
        return all_results

    gec_config = GECConfig()
    if args.gec_model:
        if args.gec_method == "llm":
            gec_config.llm_model_id = args.gec_model
        else:
            gec_config.dedicated_gec_model_id = args.gec_model

    corrector = load_gec_corrector(gec_config, device, method=args.gec_method)

    for model_name, results in all_results.items():
        gec_key = f"gec_done_{model_name}"
        total = len(results["continuations"])
        n_batches = (total + chunk - 1) // chunk

        if state.get(gec_key):
            logger.info(f"  {model_name}: GEC already complete, loading from checkpoint")
            corrected_cont = []
            corrected_full = []
            for bi in range(n_batches):
                bd = _load_batch(args.output_dir, "gec", model_name, bi)
                corrected_cont.extend(bd["corrected_continuations"])
                corrected_full.extend(bd["corrected_full_texts"])
            results["corrected_continuations"] = corrected_cont
            results["corrected_full_texts"] = corrected_full
            continue

        t0 = time.time()
        corrected_cont = []
        corrected_full = []

        for bi in range(n_batches):
            start = bi * chunk
            end = min(start + chunk, total)

            existing = _load_batch(args.output_dir, "gec", model_name, bi)
            if existing is not None:
                logger.info(f"  {model_name}: GEC batch {bi+1}/{n_batches} [{start}:{end}] loaded from checkpoint")
                corrected_cont.extend(existing["corrected_continuations"])
                corrected_full.extend(existing["corrected_full_texts"])
                continue

            batch_cont = results["continuations"][start:end]
            batch_full = results["full_texts"][start:end]

            cc = corrector.correct(batch_cont)
            cf = corrector.correct(batch_full)

            _save_batch(args.output_dir, "gec", model_name, bi, {
                "corrected_continuations": cc,
                "corrected_full_texts": cf,
            })

            corrected_cont.extend(cc)
            corrected_full.extend(cf)

            elapsed = time.time() - t0
            logger.info(f"  {model_name}: GEC batch {bi+1}/{n_batches} [{start}:{end}] "
                         f"done ({len(corrected_cont)}/{total}, {elapsed:.0f}s)")

        results["corrected_continuations"] = corrected_cont
        results["corrected_full_texts"] = corrected_full

        state[gec_key] = True
        _save_state(args.output_dir, state)

        elapsed = time.time() - t0
        logger.info(f"  {model_name}: corrected {len(corrected_cont)} sentences in {elapsed:.1f}s")

    del corrector
    torch.cuda.empty_cache() if device.type == "cuda" else None

    return all_results


###############################################################################
# Step 4: Annotate (CPU-only, fast — no checkpointing needed but added anyway)
###############################################################################

def step_4_annotate_resumable(args, all_results) -> dict:
    logger.info("=" * 60)
    logger.info("STEP 4: ERRANT Error Annotation")
    logger.info("=" * 60)

    state = _load_state(args.output_dir)
    annotator = ERRANTAnnotator(lang="en")

    for model_name, results in all_results.items():
        logger.info(f"Annotating errors for {model_name}...")

        annotations = annotator.annotate_batch(
            results["continuations"],
            results["corrected_continuations"],
        )
        results["annotations"] = annotations

        summary = summarize_errors(annotations)
        results["error_summary"] = summary

        logger.info(f"  {model_name}: {summary['total_errors']} errors in "
                     f"{summary['total_sentences']} sentences "
                     f"(rate: {summary['error_rate']:.3f})")
        if summary["top_10_error_types"]:
            logger.info(f"  Top errors: {summary['top_10_error_types'][:5]}")

    state["annotate_done"] = True
    _save_state(args.output_dir, state)

    return all_results


###############################################################################
# Step 5: Analyze (identical to run_pipeline.py)
###############################################################################

def step_5_analyze(args, all_results, items):
    logger.info("=" * 60)
    logger.info("STEP 5: Analysis & Visualization")
    logger.info("=" * 60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for model_name, results in all_results.items():
        summary = compute_model_summary(
            model_name=model_name,
            perplexities=results["perplexities"],
            error_summaries=results["error_summary"],
            annotations=results["annotations"],
        )
        summaries.append(summary)

    comparison = compare_models(summaries)

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

    for test in comparison.get("pairwise_tests", []):
        logger.info(f"\n  {test['model_a']} vs {test['model_b']}:")
        logger.info(f"    PPL:    U={test['ppl_test_stat']}, p={test['ppl_p_value']}")
        logger.info(f"    Errors: U={test['error_test_stat']}, p={test['error_p_value']}")

    save_results(summaries, comparison, args.output_dir)

    # Save raw results
    raw_output = {}
    for model_name, results in all_results.items():
        raw_output[model_name] = {
            "continuations": results["continuations"],
            "full_texts": results["full_texts"],
            "corrected_continuations": results["corrected_continuations"],
            "corrected_full_texts": results.get("corrected_full_texts", []),
            "perplexities": results["perplexities"],
            "error_summary": results["error_summary"],
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

    with open(output_dir / "prompts.json", "w") as f:
        json.dump(items, f, indent=2)

    if not args.skip_plots:
        try:
            generate_all_plots(summaries, str(output_dir / "plots"))
            logger.info("All plots generated successfully")
        except Exception as e:
            logger.warning(f"Plot generation failed: {e}")

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

def main():
    args = parse_args()

    if args.skip_gec:
        args.skip_generation = True

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    logger.info(f"Checkpoint every: {args.checkpoint_every} sentences")

    # Step 1: Load data
    logger.info("=" * 60)
    logger.info("STEP 1: Loading data")
    logger.info("=" * 60)

    sentences = load_sentences(args.data_path, max_sentences=args.max_sentences)
    items = make_prompts(sentences, prompt_ratio=args.prompt_ratio)
    logger.info(f"Prepared {len(items)} prompt-reference pairs")

    if not args.skip_generation:
        all_results = step_2_generate_resumable(args, items, device)
        all_results = step_3_gec_resumable(args, all_results, device)
    else:
        # Load from existing raw_results.json (same as original pipeline)
        raw_path = Path(args.output_dir) / "raw_results.json"
        logger.info(f"Loading existing results from {raw_path}")
        with open(raw_path) as f:
            all_results = json.load(f)
        prompts = [item["prompt"] for item in items]
        for model_name, results in all_results.items():
            if "full_texts" not in results:
                results["full_texts"] = [
                    f"{p} {c}" for p, c in zip(prompts, results["continuations"])
                ]
            if "corrected_full_texts" not in results:
                results["corrected_full_texts"] = []

        if not args.skip_gec:
            all_results = step_3_gec_resumable(args, all_results, device)

    # Step 4: Annotate
    all_results = step_4_annotate_resumable(args, all_results)

    # Step 5: Analyze
    summaries, comparison = step_5_analyze(args, all_results, items)

    logger.info("\nPipeline complete!")
    return summaries, comparison


if __name__ == "__main__":
    main()
