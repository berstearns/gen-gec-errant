"""Pipeline runner: orchestrates all stages end-to-end."""

import gc
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from gen_gec_errant.pipeline.config import PipelineConfig, get_model_configs
from gen_gec_errant.data_loader.runner import run_data_loader
from gen_gec_errant.data_loader.config import DataLoaderConfig
from gen_gec_errant.generation.config import GenerationConfig, ModelConfig, GenerationParams
from gen_gec_errant.generation.runner import (
    get_device,
    load_model,
    generate_continuations,
    compute_perplexity,
)
from gen_gec_errant.gec.runner import run_gec, load_gec_corrector
from gen_gec_errant.gec.config import GECConfig
from gen_gec_errant.annotation.runner import (
    run_annotation,
    classify_errors_by_region,
    summarize_errors_by_region,
)
from gen_gec_errant.annotation.config import AnnotationConfig
from gen_gec_errant.analysis.runner import run_analysis
from gen_gec_errant.analysis.config import AnalysisConfig

logger = logging.getLogger(__name__)


def _step_1_load_data(config: PipelineConfig) -> List[dict]:
    """Load and prepare data."""
    logger.info("=" * 60)
    logger.info("STEP 1: Loading data")
    logger.info("=" * 60)

    items = run_data_loader(config.data_loader)

    logger.info("Prepared %d prompt-reference pairs", len(items))
    if items:
        logger.info("  Example prompt:    '%s'", items[0]["prompt"])
        logger.info("  Example reference: '%s'", items[0]["reference"])

    return items


def _step_2_generate(
    config: PipelineConfig,
    items: List[dict],
    device: torch.device,
) -> Dict[str, dict]:
    """Generate continuations for all models."""
    logger.info("=" * 60)
    logger.info("STEP 2: Generating text with models")
    logger.info("=" * 60)

    gen_params = config.generation
    model_configs = get_model_configs(config)
    prompts = [item["prompt"] for item in items]
    all_results: Dict[str, dict] = {}

    for mc in model_configs:
        t0 = time.time()

        # Use per-model batch_size if set, otherwise fall back to global
        gen_batch = mc.batch_size if mc.batch_size != 8 else config.batch_size
        logger.info("Using generation batch_size=%d for %s", gen_batch, mc.name)

        model, tokenizer = load_model(mc, device)

        logger.info("Generating with %s...", mc.name)
        continuations = generate_continuations(
            model, tokenizer, prompts, gen_params,
            batch_size=gen_batch, device=device,
        )

        full_texts = [f"{p} {c}" for p, c in zip(prompts, continuations)]

        if device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("Computing perplexity for %s...", mc.name)
        ppl_batch = min(gen_batch, 32)
        perplexities = compute_perplexity(
            model, tokenizer, full_texts,
            batch_size=ppl_batch, device=device,
        )

        elapsed = time.time() - t0
        logger.info("  %s: generated %d sentences in %.1fs", mc.name, len(continuations), elapsed)
        logger.info("  Mean perplexity: %.2f", sum(perplexities) / len(perplexities))

        all_results[mc.name] = {
            "continuations": continuations,
            "full_texts": full_texts,
            "perplexities": perplexities,
            "prompt_boundaries": [len(p) for p in prompts],
        }

        del model, tokenizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return all_results


def _step_3_gec(config: PipelineConfig, all_results: Dict[str, dict]) -> Dict[str, dict]:
    """Run GEC on all generated text — loads GEC model once for all models."""
    logger.info("=" * 60)
    logger.info("STEP 3: Grammatical Error Correction")
    logger.info("=" * 60)

    gec_config = config.gec
    device = get_device(gec_config.device)
    corrector = load_gec_corrector(gec_config, device)

    for model_name, results in all_results.items():
        logger.info("Correcting %s outputs...", model_name)

        corrected_continuations: list[str] = []
        for i in range(0, len(results["continuations"]), gec_config.batch_size):
            batch = results["continuations"][i : i + gec_config.batch_size]
            corrected_continuations.extend(corrector.correct(batch))

        corrected_full_texts: list[str] = []
        for i in range(0, len(results["full_texts"]), gec_config.batch_size):
            batch = results["full_texts"][i : i + gec_config.batch_size]
            corrected_full_texts.extend(corrector.correct(batch))

        results["corrected_continuations"] = corrected_continuations
        results["corrected_full_texts"] = corrected_full_texts
        logger.info("Corrected %d sentences for %s", len(corrected_continuations), model_name)

    del corrector
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return all_results


def _add_learner_baseline(items: List[dict], all_results: Dict[str, dict]) -> None:
    """Add learner baseline as a pseudo-model entry for comparison."""
    logger.info("Adding learner baseline...")
    all_results["learner_baseline"] = {
        "full_texts": [item["full"] for item in items],
        "continuations": [item["reference"] for item in items],
        "perplexities": [0.0] * len(items),
        "prompt_boundaries": [len(item["prompt"]) for item in items],
    }


def _step_4_annotate(config: PipelineConfig, all_results: Dict[str, dict]) -> Dict[str, dict]:
    """Run ERRANT annotation."""
    logger.info("=" * 60)
    logger.info("STEP 4: ERRANT Error Annotation")
    logger.info("=" * 60)

    ann_config = config.annotation

    for model_name, results in all_results.items():
        logger.info("Annotating errors for %s...", model_name)
        run_annotation(ann_config, results)

        # Region classification on full-text annotations
        if "full_text_annotations" in results and "prompt_boundaries" in results:
            classify_errors_by_region(results["full_text_annotations"], results["prompt_boundaries"])
            results["region_error_summary"] = summarize_errors_by_region(results["full_text_annotations"])

    return all_results


def _step_5_analyze(
    config: PipelineConfig,
    all_results: Dict[str, dict],
    items: List[dict],
) -> Tuple[list, dict]:
    """Compute summaries, comparisons, plots, CSVs."""
    logger.info("=" * 60)
    logger.info("STEP 5: Analysis & Visualization")
    logger.info("=" * 60)

    analysis_config = config.analysis
    analysis_config.output_dir = config.output_dir
    analysis_config.skip_plots = config.skip_plots

    # Save raw data for reproducibility
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _serialize_annotations(annotations):
        return [
            {
                "original": a.original,
                "corrected": a.corrected,
                "num_errors": a.num_errors,
                "error_types": a.error_type_counts,
                "prompt_error_count": a.prompt_error_count,
                "generation_error_count": a.generation_error_count,
                "prompt_error_type_counts": a.prompt_error_type_counts,
                "generation_error_type_counts": a.generation_error_type_counts,
                "errors": [
                    {
                        "orig_tokens": e.original_tokens,
                        "corr_tokens": e.corrected_tokens,
                        "type": e.error_type,
                        "char_start": e.char_start,
                        "char_end": e.char_end,
                        "region": e.region,
                    }
                    for e in a.errors
                ],
            }
            for a in annotations
        ]

    raw_output = {}
    for model_name, results in all_results.items():
        raw_output[model_name] = {
            "continuations": results["continuations"],
            "full_texts": results["full_texts"],
            "corrected_continuations": results.get("corrected_continuations", []),
            "corrected_full_texts": results.get("corrected_full_texts", []),
            "perplexities": results["perplexities"],
            "prompt_boundaries": results.get("prompt_boundaries", []),
            "error_summary": results.get("error_summary", {}),
            "annotations": _serialize_annotations(results.get("annotations", [])),
            "full_text_annotations": _serialize_annotations(
                results.get("full_text_annotations", [])),
            "full_text_error_summary": results.get("full_text_error_summary", {}),
            "region_error_summary": results.get("region_error_summary", {}),
        }

    with open(output_dir / "raw_results.json", "w") as f:
        json.dump(raw_output, f, indent=2, default=str)
    logger.info("Saved raw results to %s", output_dir / "raw_results.json")

    with open(output_dir / "prompts.json", "w") as f:
        json.dump(items, f, indent=2)

    return run_analysis(analysis_config, all_results, items)


def _load_raw_results(output_dir: str, items: List[dict]) -> Dict[str, dict]:
    """Load raw_results.json and reconstruct missing keys."""
    raw_path = Path(output_dir) / "raw_results.json"
    logger.info("Loading existing results from %s", raw_path)
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

    return all_results


def run_pipeline(config: PipelineConfig) -> Tuple[list, dict]:
    """
    Run the full pipeline end-to-end.

    Returns:
        (summaries, comparison)
    """
    if config.skip_gec:
        config.skip_generation = True

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = get_device(config.device)
    logger.info("Using device: %s", device)

    # Step 1: Load data
    items = _step_1_load_data(config)

    if not config.skip_generation:
        # Step 2: Generate
        all_results = _step_2_generate(config, items, device)

        # Add learner baseline before GEC so it flows through GEC → ERRANT → Analysis
        if config.include_learner_baseline:
            _add_learner_baseline(items, all_results)

        # Free GPU before loading GEC model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            free_mb = torch.cuda.mem_get_info()[0] // (1024 * 1024)
            logger.info("GPU memory free before GEC: %d MiB", free_mb)

        # Step 3: GEC
        all_results = _step_3_gec(config, all_results)
    else:
        all_results = _load_raw_results(config.output_dir, items)

        # Add learner baseline
        _add_learner_baseline(items, all_results)

        if not config.skip_gec:
            all_results = _step_3_gec(config, all_results)

    # Step 4: Annotate
    all_results = _step_4_annotate(config, all_results)

    # Step 5: Analyze
    summaries, comparison = _step_5_analyze(config, all_results, items)

    logger.info("Pipeline complete!")
    return summaries, comparison
