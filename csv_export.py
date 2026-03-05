"""
Step 6: Comprehensive CSV export.

Produces a single CSV where each row is one input sentence, with columns for
every model's generation, correction, ERRANT errors, perplexity, etc.

This makes it easy to inspect individual sentences, filter/sort in Excel,
and do downstream analysis in R/pandas.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _attr(obj, key, default=None):
    """Access a field from either a dataclass object or a plain dict."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _clean_for_tsv(text: str) -> str:
    """Clean text for TSV output: remove newlines, tabs, normalize whitespace."""
    if not isinstance(text, str):
        return str(text)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    # Collapse multiple spaces
    text = " ".join(text.split())
    return text.strip()


def build_csv_rows(
    items: List[dict],
    all_results: Dict[str, dict],
    model_names: List[str],
) -> List[dict]:
    """
    Build flat rows for CSV export.

    Each row = one input sentence, with columns for every model's outputs.

    Args:
        items: List of {"prompt", "reference", "full"} from data_loader
        all_results: Dict[model_name -> {"continuations", "corrected_continuations",
                     "full_texts", "corrected_full_texts", "perplexities", "annotations", ...}]
        model_names: Ordered list of model names to include

    Returns:
        List of dicts (one per sentence), ready for csv.DictWriter
    """
    n_sentences = len(items)
    rows = []

    for i in range(n_sentences):
        row = {
            "sentence_id": i,
            "prompt": items[i]["prompt"],
            "reference_continuation": items[i]["reference"],
            "full_original": items[i]["full"],
        }

        for model_name in model_names:
            if model_name not in all_results:
                continue

            res = all_results[model_name]
            prefix = model_name.replace("-", "_")  # safe column name

            # --- Generation ---
            continuation = res["continuations"][i] if i < len(res["continuations"]) else ""
            full_text = res["full_texts"][i] if i < len(res["full_texts"]) else ""
            row[f"{prefix}__continuation"] = _clean_for_tsv(continuation)
            row[f"{prefix}__full_text"] = _clean_for_tsv(full_text)

            # --- GEC correction ---
            corr_cont = res.get("corrected_continuations", [])
            corr_full = res.get("corrected_full_texts", [])
            row[f"{prefix}__corrected_continuation"] = _clean_for_tsv(corr_cont[i]) if i < len(corr_cont) else ""
            row[f"{prefix}__corrected_full_text"] = _clean_for_tsv(corr_full[i]) if i < len(corr_full) else ""

            # --- Perplexity ---
            ppl = res["perplexities"][i] if i < len(res["perplexities"]) else ""
            row[f"{prefix}__perplexity"] = round(ppl, 4) if isinstance(ppl, float) else ppl

            # --- ERRANT annotations ---
            # Annotations may be SentenceAnnotation objects (live run) or
            # plain dicts (loaded from raw_results.json). Use _attr() for both.
            annotations = res.get("annotations", [])
            if i < len(annotations):
                ann = annotations[i]
                num_errors = _attr(ann, "num_errors", 0)
                errors = _attr(ann, "errors", [])
                type_counts = _attr(ann, "error_type_counts", {})
                row[f"{prefix}__num_errors"] = num_errors

                # Error types as semicolon-separated string
                # e.g. "M:DET;R:VERB:SVA;R:PREP"
                error_types = [_attr(e, "error_type", _attr(e, "type", "")) for e in errors]
                row[f"{prefix}__error_types"] = ";".join(error_types) if error_types else ""

                # Error type counts as key=count pairs
                # e.g. "M:DET=1;R:VERB:SVA=2"
                type_counts_str = ";".join(
                    f"{k}={v}" for k, v in sorted(type_counts.items())
                )
                row[f"{prefix}__error_type_counts"] = type_counts_str

                # Detailed error list: orig->corr(type) separated by |
                # e.g. "->the(M:DET)|go->goes(R:VERB:SVA)"
                error_details = []
                for e in errors:
                    orig = _attr(e, "original_tokens", _attr(e, "orig_tokens", "")) or ""
                    corr = _attr(e, "corrected_tokens", _attr(e, "corr_tokens", "")) or ""
                    etype = _attr(e, "error_type", _attr(e, "type", ""))
                    error_details.append(f"{orig}->{corr}({etype})")
                row[f"{prefix}__error_details"] = "|".join(error_details) if error_details else ""

                # Binary flag: has errors?
                row[f"{prefix}__has_errors"] = 1 if num_errors > 0 else 0
            else:
                row[f"{prefix}__num_errors"] = ""
                row[f"{prefix}__error_types"] = ""
                row[f"{prefix}__error_type_counts"] = ""
                row[f"{prefix}__error_details"] = ""
                row[f"{prefix}__has_errors"] = ""

        rows.append(row)

    return rows


def export_csv(
    items: List[dict],
    all_results: Dict[str, dict],
    model_names: List[str],
    output_path: str,
) -> str:
    """
    Export the full pipeline results to a single CSV file.

    Columns per model (using gpt2_base as example prefix):
        gpt2_base__continuation             - Raw generated text
        gpt2_base__full_text                - Prompt + continuation
        gpt2_base__corrected_continuation   - GEC-corrected continuation
        gpt2_base__corrected_full_text      - GEC-corrected full text
        gpt2_base__perplexity               - Model perplexity on full text
        gpt2_base__num_errors               - Total ERRANT error count
        gpt2_base__error_types              - Semicolon-separated error type list
        gpt2_base__error_type_counts        - Error types with counts
        gpt2_base__error_details            - Full error details (orig->corr(type))
        gpt2_base__has_errors               - Binary flag (0/1)

    Returns:
        Path to the saved file.
    """
    output_path = Path(output_path)
    # Use TSV by default — handles quotes/commas in generated text much better
    output_path = output_path.with_suffix(".tsv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = build_csv_rows(items, all_results, model_names)

    if not rows:
        logger.warning("No rows to export")
        return str(output_path)

    fieldnames = list(rows[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t",
                                quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Exported {len(rows)} rows × {len(fieldnames)} columns to {output_path}")
    return str(output_path)


def export_errors_long_format(
    all_results: Dict[str, dict],
    model_names: List[str],
    output_path: str,
) -> str:
    """
    Export errors in long format (one row per error) for easier
    filtering and analysis in R/pandas/Excel pivot tables.

    Columns:
        sentence_id, model, original_tokens, corrected_tokens,
        error_type, error_category, start_offset, end_offset
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name in model_names:
        if model_name not in all_results:
            continue
        annotations = all_results[model_name].get("annotations", [])
        for sent_id, ann in enumerate(annotations):
            errors = _attr(ann, "errors", [])
            for err in errors:
                # Parse error category (first part before colon)
                # e.g. "R:VERB:SVA" -> category="R", subcategory="VERB:SVA"
                etype = _attr(err, "error_type", _attr(err, "type", ""))
                parts = etype.split(":", 1)
                category = parts[0] if parts else ""
                subcategory = parts[1] if len(parts) > 1 else ""

                rows.append({
                    "sentence_id": sent_id,
                    "model": model_name,
                    "original_text": _clean_for_tsv(_attr(ann, "original", "")),
                    "corrected_text": _clean_for_tsv(_attr(ann, "corrected", "")),
                    "error_original_tokens": _attr(err, "original_tokens", _attr(err, "orig_tokens", "")),
                    "error_corrected_tokens": _attr(err, "corrected_tokens", _attr(err, "corr_tokens", "")),
                    "error_type": etype,
                    "error_operation": category,    # M=Missing, R=Replace, U=Unnecessary
                    "error_subcategory": subcategory,  # DET, VERB:SVA, PREP, etc.
                    "start_offset": _attr(err, "start_offset", 0),
                    "end_offset": _attr(err, "end_offset", 0),
                })

    if not rows:
        logger.warning("No errors to export in long format")
        fieldnames = [
            "sentence_id", "model", "original_text", "corrected_text",
            "error_original_tokens", "error_corrected_tokens", "error_type",
            "error_operation", "error_subcategory", "start_offset", "end_offset",
        ]
    else:
        fieldnames = list(rows[0].keys())

    output_path = Path(output_path).with_suffix(".tsv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t",
                                quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Exported {len(rows)} error rows (long format) to {output_path}")
    return str(output_path)
