"""
Step 5: Analysis and visualization.

Computes:
- Perplexity distributions per model
- Error counts and rates per model
- Error type breakdowns
- Combined perplexity + error count metrics
- Statistical significance tests
- Publication-quality plots
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports for optional deps
def _import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

def _import_scipy():
    from scipy import stats
    return stats


###############################################################################
# Core analysis
###############################################################################

def compute_model_summary(
    model_name: str,
    perplexities: List[float],
    error_summaries: Dict,
    annotations: List,
) -> Dict:
    """
    Compute comprehensive summary metrics for a single model.
    """
    ppl_array = np.array(perplexities)
    errors_per_sent = np.array(error_summaries["errors_per_sentence"])

    summary = {
        "model_name": model_name,
        # Perplexity
        "ppl_mean": float(np.mean(ppl_array)),
        "ppl_median": float(np.median(ppl_array)),
        "ppl_std": float(np.std(ppl_array)),
        "ppl_25th": float(np.percentile(ppl_array, 25)),
        "ppl_75th": float(np.percentile(ppl_array, 75)),
        # Errors
        "total_errors": error_summaries["total_errors"],
        "avg_errors_per_sentence": error_summaries["avg_errors_per_sentence"],
        "error_rate": error_summaries["error_rate"],
        "sentences_with_errors": error_summaries["sentences_with_errors"],
        "total_sentences": error_summaries["total_sentences"],
        # Error type distribution
        "top_10_error_types": error_summaries["top_10_error_types"],
        "error_type_counts": error_summaries["error_type_counts"],
        # Combined metric: avg_ppl * avg_errors (higher = worse)
        "ppl_x_errors": float(np.mean(ppl_array) * error_summaries["avg_errors_per_sentence"]),
        # Per-sentence combined
        "per_sentence_ppl_plus_errors": [
            {"ppl": float(p), "errors": int(e)}
            for p, e in zip(ppl_array, errors_per_sent)
        ],
    }

    return summary


def compare_models(summaries: List[Dict]) -> Dict:
    """
    Compare multiple model summaries with statistical tests.
    """
    stats_mod = _import_scipy()

    comparison = {
        "models": [s["model_name"] for s in summaries],
        "ppl_means": [s["ppl_mean"] for s in summaries],
        "error_rates": [s["error_rate"] for s in summaries],
        "avg_errors": [s["avg_errors_per_sentence"] for s in summaries],
        "ppl_x_errors": [s["ppl_x_errors"] for s in summaries],
    }

    # Pairwise comparisons between base and learner-tuned models
    pairwise = []
    for i in range(len(summaries)):
        for j in range(i + 1, len(summaries)):
            s1, s2 = summaries[i], summaries[j]

            ppls1 = [x["ppl"] for x in s1["per_sentence_ppl_plus_errors"]]
            ppls2 = [x["ppl"] for x in s2["per_sentence_ppl_plus_errors"]]
            errs1 = [x["errors"] for x in s1["per_sentence_ppl_plus_errors"]]
            errs2 = [x["errors"] for x in s2["per_sentence_ppl_plus_errors"]]

            # Mann-Whitney U test (non-parametric) for perplexity
            try:
                ppl_stat, ppl_pval = stats_mod.mannwhitneyu(ppls1, ppls2, alternative="two-sided")
            except Exception:
                ppl_stat, ppl_pval = None, None

            # Mann-Whitney U test for error counts
            try:
                err_stat, err_pval = stats_mod.mannwhitneyu(errs1, errs2, alternative="two-sided")
            except Exception:
                err_stat, err_pval = None, None

            pairwise.append({
                "model_a": s1["model_name"],
                "model_b": s2["model_name"],
                "ppl_test_stat": ppl_stat,
                "ppl_p_value": ppl_pval,
                "error_test_stat": err_stat,
                "error_p_value": err_pval,
            })

    comparison["pairwise_tests"] = pairwise
    return comparison


###############################################################################
# Visualization
###############################################################################

def plot_perplexity_comparison(summaries: List[Dict], output_path: str):
    """Bar chart of mean perplexity per model."""
    plt = _import_matplotlib()

    names = [s["model_name"] for s in summaries]
    means = [s["ppl_mean"] for s in summaries]
    stds = [s["ppl_std"] for s in summaries]

    colors = ["#2196F3" if "learner" not in n else "#F44336" for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax.set_ylabel("Mean Perplexity", fontsize=12)
    ax.set_title("Perplexity by Model", fontsize=14)
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved perplexity plot: {output_path}")


def plot_error_comparison(summaries: List[Dict], output_path: str):
    """Bar chart of average errors per sentence."""
    plt = _import_matplotlib()

    names = [s["model_name"] for s in summaries]
    avg_errors = [s["avg_errors_per_sentence"] for s in summaries]
    error_rates = [s["error_rate"] for s in summaries]

    colors = ["#2196F3" if "learner" not in n else "#F44336" for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.bar(names, avg_errors, color=colors, alpha=0.8)
    ax1.set_ylabel("Avg Errors per Sentence", fontsize=12)
    ax1.set_title("Error Count by Model", fontsize=14)
    ax1.tick_params(axis="x", rotation=30)

    ax2.bar(names, error_rates, color=colors, alpha=0.8)
    ax2.set_ylabel("Fraction of Sentences with Errors", fontsize=12)
    ax2.set_title("Error Rate by Model", fontsize=14)
    ax2.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved error comparison plot: {output_path}")


def plot_error_type_breakdown(summaries: List[Dict], output_path: str, top_n: int = 10):
    """Grouped bar chart of top N error types across models."""
    plt = _import_matplotlib()

    # Collect all error types across models, pick top N by total frequency
    all_types: Dict[str, int] = {}
    for s in summaries:
        for etype, count in s["error_type_counts"].items():
            all_types[etype] = all_types.get(etype, 0) + count

    top_types = sorted(all_types.items(), key=lambda x: -x[1])[:top_n]
    type_names = [t[0] for t in top_types]

    x = np.arange(len(type_names))
    width = 0.8 / len(summaries)

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, s in enumerate(summaries):
        counts = [s["error_type_counts"].get(t, 0) for t in type_names]
        offset = (i - len(summaries) / 2 + 0.5) * width
        bars = ax.bar(x + offset, counts, width, label=s["model_name"], alpha=0.8)

    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Top {top_n} Error Types by Model", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(type_names, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved error type breakdown plot: {output_path}")


def plot_ppl_vs_errors_scatter(summaries: List[Dict], output_path: str):
    """Scatter plot: perplexity vs error count per sentence, colored by model."""
    plt = _import_matplotlib()

    fig, ax = plt.subplots(figsize=(10, 8))

    for s in summaries:
        ppls = [x["ppl"] for x in s["per_sentence_ppl_plus_errors"]]
        errs = [x["errors"] for x in s["per_sentence_ppl_plus_errors"]]
        ax.scatter(ppls, errs, alpha=0.3, s=20, label=s["model_name"])

    ax.set_xlabel("Perplexity", fontsize=12)
    ax.set_ylabel("Error Count", fontsize=12)
    ax.set_title("Perplexity vs Grammatical Errors per Sentence", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved scatter plot: {output_path}")


def plot_combined_metric(summaries: List[Dict], output_path: str):
    """Bar chart of the combined PPL × Errors metric."""
    plt = _import_matplotlib()

    names = [s["model_name"] for s in summaries]
    combined = [s["ppl_x_errors"] for s in summaries]
    colors = ["#2196F3" if "learner" not in n else "#F44336" for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(names, combined, color=colors, alpha=0.8)
    ax.set_ylabel("Mean PPL × Avg Errors", fontsize=12)
    ax.set_title("Combined Metric: Perplexity × Error Rate", fontsize=14)
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved combined metric plot: {output_path}")


###############################################################################
# Export
###############################################################################

def save_results(
    summaries: List[Dict],
    comparison: Dict,
    output_dir: str,
):
    """Save all results to JSON files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-model summaries
    for s in summaries:
        path = output_dir / f"{s['model_name']}_summary.json"
        with open(path, "w") as f:
            json.dump(s, f, indent=2, default=str)
        logger.info(f"Saved: {path}")

    # Comparison
    path = output_dir / "model_comparison.json"
    with open(path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    logger.info(f"Saved: {path}")


def generate_all_plots(summaries: List[Dict], output_dir: str):
    """Generate all visualization plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_perplexity_comparison(summaries, str(output_dir / "perplexity_comparison.png"))
    plot_error_comparison(summaries, str(output_dir / "error_comparison.png"))
    plot_error_type_breakdown(summaries, str(output_dir / "error_type_breakdown.png"))
    plot_ppl_vs_errors_scatter(summaries, str(output_dir / "ppl_vs_errors_scatter.png"))
    plot_combined_metric(summaries, str(output_dir / "combined_metric.png"))
