#!/usr/bin/env python3
"""
Quick-start script for running the pipeline interactively or as a simple script.
Edit the CONFIG section below and run:  python quickstart.py
"""

import sys
sys.path.insert(0, ".")

import torch
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("quickstart")

# ============================================================================
# CONFIG — Edit these paths
# ============================================================================

DATA_PATH = "data/efcamdat_sentences.txt"       # Your EFCAMDAT sentences
LEARNER_CHECKPOINT = "path/to/your/learner.pt"   # Your .pt checkpoint
OUTPUT_DIR = "results/quickstart"
MAX_SENTENCES = 100                               # Start small for testing
DEVICE = "auto"                                   # "cuda", "cpu", or "auto"

# Models to compare
MODELS_TO_RUN = ["gpt2-base", "gpt2-learner"]
# Uncomment to add Pythia:
# MODELS_TO_RUN = ["gpt2-base", "gpt2-learner", "pythia-70m", "pythia-160m"]

# GEC settings
GEC_METHOD = "dedicated"  # "dedicated" (coedit) or "llm" (gemma)

# ============================================================================
# RUN
# ============================================================================

from config import ModelConfig, GenerationConfig, GECConfig
from data_loader import load_sentences, make_prompts
from generation import get_device, load_model, generate_continuations, compute_perplexity
from gec import load_gec_corrector
from error_annotation import ERRANTAnnotator, summarize_errors
from analysis import compute_model_summary, compare_models, save_results, generate_all_plots
from csv_export import export_csv, export_errors_long_format

# Model registry
REGISTRY = {
    "gpt2-base": ModelConfig(name="gpt2-base", hf_model_id="gpt2"),
    "gpt2-learner": ModelConfig(
        name="gpt2-learner", hf_model_id="gpt2",
        is_learner_tuned=True, checkpoint_path=LEARNER_CHECKPOINT,
    ),
    "pythia-70m": ModelConfig(name="pythia-70m", hf_model_id="EleutherAI/pythia-70m"),
    "pythia-160m": ModelConfig(name="pythia-160m", hf_model_id="EleutherAI/pythia-160m"),
    "pythia-410m": ModelConfig(name="pythia-410m", hf_model_id="EleutherAI/pythia-410m"),
}


def main():
    device = get_device(DEVICE)
    torch.manual_seed(42)
    logger.info(f"Device: {device}")

    # --- Step 1: Data ---
    logger.info("Loading data...")
    sentences = load_sentences(DATA_PATH, max_sentences=MAX_SENTENCES)
    items = make_prompts(sentences)
    prompts = [item["prompt"] for item in items]
    logger.info(f"Loaded {len(items)} sentences")

    gen_config = GenerationConfig(max_new_tokens=50, temperature=1.0)

    # --- Step 2: Generate ---
    all_results = {}
    for model_name in MODELS_TO_RUN:
        cfg = REGISTRY[model_name]
        logger.info(f"\n--- {model_name} ---")

        model, tokenizer = load_model(cfg, device)

        continuations = generate_continuations(
            model, tokenizer, prompts, gen_config, batch_size=8, device=device,
        )
        full_texts = [f"{p} {c}" for p, c in zip(prompts, continuations)]
        perplexities = compute_perplexity(
            model, tokenizer, full_texts, batch_size=8, device=device,
        )

        all_results[model_name] = {
            "continuations": continuations,
            "full_texts": full_texts,
            "perplexities": perplexities,
        }

        logger.info(f"  Mean PPL: {sum(perplexities)/len(perplexities):.2f}")

        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # --- Step 3: GEC ---
    logger.info("\nRunning GEC...")
    gec_config = GECConfig()
    corrector = load_gec_corrector(gec_config, device, method=GEC_METHOD)

    for name, res in all_results.items():
        res["corrected_continuations"] = corrector.correct(res["continuations"])
        res["corrected_full_texts"] = corrector.correct(res["full_texts"])

    del corrector
    torch.cuda.empty_cache() if device.type == "cuda" else None

    # --- Step 4: ERRANT ---
    logger.info("\nAnnotating errors with ERRANT...")
    annotator = ERRANTAnnotator()

    for name, res in all_results.items():
        annotations = annotator.annotate_batch(
            res["continuations"], res["corrected_continuations"],
        )
        res["annotations"] = annotations
        res["error_summary"] = summarize_errors(annotations)
        logger.info(f"  {name}: {res['error_summary']['total_errors']} errors, "
                     f"rate={res['error_summary']['error_rate']:.3f}")

    # --- Step 5: Analysis ---
    logger.info("\nAnalyzing results...")
    summaries = []
    for name, res in all_results.items():
        s = compute_model_summary(name, res["perplexities"], res["error_summary"], res["annotations"])
        summaries.append(s)

    comparison = compare_models(summaries)

    # Print table
    print("\n" + "=" * 72)
    print(f"{'Model':<20} {'PPL':>8} {'Err Rate':>10} {'Avg Err':>10} {'PPL×Err':>10}")
    print("-" * 72)
    for s in summaries:
        print(f"{s['model_name']:<20} {s['ppl_mean']:>8.1f} {s['error_rate']:>10.3f} "
              f"{s['avg_errors_per_sentence']:>10.3f} {s['ppl_x_errors']:>10.1f}")
    print("=" * 72)

    # Save
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    save_results(summaries, comparison, OUTPUT_DIR)

    try:
        generate_all_plots(summaries, f"{OUTPUT_DIR}/plots")
    except Exception as e:
        logger.warning(f"Plotting failed: {e}")

    # --- Step 6: CSV Export ---
    logger.info("\nExporting CSVs...")
    model_names = list(all_results.keys())

    export_csv(
        items=items,
        all_results=all_results,
        model_names=model_names,
        output_path=f"{OUTPUT_DIR}/full_results.csv",
    )

    export_errors_long_format(
        all_results=all_results,
        model_names=model_names,
        output_path=f"{OUTPUT_DIR}/errors_long_format.csv",
    )

    logger.info(f"\nDone! Results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
