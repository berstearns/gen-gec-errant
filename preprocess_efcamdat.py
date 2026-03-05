#!/usr/bin/env python3
"""
Preprocess the EFCAMDAT CSV into sentence-level data for the pipeline.

The raw EFCAMDAT CSV has:
- Full essays (not sentences) in the text columns
- Multiline text fields
- Rich metadata (CEFR level, L1, topic, etc.)

This script:
1. Reads the CSV and auto-detects the text columns
2. Splits essays into individual sentences (using spaCy)
3. Outputs a clean CSV with one sentence per row + metadata

Usage:
    python preprocess_efcamdat.py \
        --input data/cleaned_efcamdat.csv \
        --output data/efcamdat_sentences.csv \
        --min_words 5 \
        --max_words 60
"""

import argparse
import csv
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("preprocess")


def detect_columns(header: list) -> dict:
    """
    Auto-detect which columns contain what, based on header names.
    Returns a mapping of role -> column index.
    """
    header_lower = [h.strip().lower() for h in header]

    mapping = {}

    # Try to find text columns by common patterns
    text_candidates = [
        "text", "original_text", "writing", "essay", "learner_text",
        "corrected", "cleaned_text", "corrected_text",
    ]

    for role, candidates in [
        ("original_text", ["text", "original_text", "writing", "essay", "learner_text"]),
        ("corrected_text", ["corrected", "cleaned_text", "corrected_text"]),
        ("cefr_level", ["cefr", "cefr_level", "level", "proficiency"]),
        ("l1", ["l1", "l1_language", "language", "native_language", "mother_tongue"]),
        ("topic", ["topic", "task", "prompt", "writing_topic"]),
    ]:
        for cand in candidates:
            for i, h in enumerate(header_lower):
                if cand == h or cand in h:
                    if role not in mapping:
                        mapping[role] = i
                    break

    return mapping


def detect_columns_by_position(header: list, sample_rows: list) -> dict:
    """
    Fallback: detect text columns by content heuristics (longest string fields).
    """
    if not sample_rows:
        return {}

    # Find the two longest text columns (likely original + corrected text)
    avg_lengths = []
    for col_idx in range(len(header)):
        lengths = []
        for row in sample_rows:
            if col_idx < len(row):
                lengths.append(len(str(row[col_idx])))
        avg_lengths.append(sum(lengths) / max(len(lengths), 1))

    # Sort by avg length descending
    sorted_cols = sorted(enumerate(avg_lengths), key=lambda x: -x[1])

    mapping = {}
    text_cols_found = 0
    for col_idx, avg_len in sorted_cols:
        if avg_len > 50 and text_cols_found < 2:  # Likely a text column
            if text_cols_found == 0:
                mapping["original_text"] = col_idx
            else:
                mapping["corrected_text"] = col_idx
            text_cols_found += 1

    # Try to find CEFR level (short column with A1/A2/B1/B2/C1/C2 values)
    cefr_pattern = re.compile(r'^[ABC][12]$')
    for col_idx in range(len(header)):
        vals = set()
        for row in sample_rows:
            if col_idx < len(row):
                vals.add(str(row[col_idx]).strip())
        if any(cefr_pattern.match(v) for v in vals):
            mapping["cefr_level"] = col_idx
            break

    # Try to find L1 language (column with language names)
    lang_names = {"arabic", "mandarin", "french", "portuguese", "italian", "spanish",
                  "german", "japanese", "korean", "russian", "turkish", "thai", "hindi"}
    for col_idx in range(len(header)):
        vals = set()
        for row in sample_rows:
            if col_idx < len(row):
                vals.add(str(row[col_idx]).strip().lower())
        if vals & lang_names:
            mapping["l1"] = col_idx
            break

    return mapping


def split_into_sentences(text: str) -> list:
    """
    Split text into sentences. Uses spaCy if available, otherwise regex fallback.
    """
    # Clean the text first
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace

    try:
        import spacy
        if not hasattr(split_into_sentences, '_nlp'):
            split_into_sentences._nlp = spacy.load("en_core_web_sm")
        doc = split_into_sentences._nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    except (ImportError, OSError):
        # Regex fallback
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


def clean_text(text: str) -> str:
    """Basic text cleaning."""
    if not text:
        return ""
    # Remove excessive whitespace / indentation
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove common artifacts
    text = text.replace('\n', ' ').replace('\r', ' ')
    return text


def main():
    parser = argparse.ArgumentParser(description="Preprocess EFCAMDAT CSV")
    parser.add_argument("--input", "-i", required=True, help="Path to cleaned_efcamdat.csv")
    parser.add_argument("--output", "-o", required=True, help="Output path for sentence-level CSV")
    parser.add_argument("--min_words", type=int, default=5, help="Min words per sentence")
    parser.add_argument("--max_words", type=int, default=60, help="Max words per sentence")
    parser.add_argument("--max_essays", type=int, default=None, help="Max essays to process")
    parser.add_argument("--cefr_filter", type=str, default=None,
                        help="Filter by CEFR level, e.g. 'A1' or 'A1,A2,B1'")
    parser.add_argument("--l1_filter", type=str, default=None,
                        help="Filter by L1 language, e.g. 'Arabic' or 'Arabic,Mandarin'")

    # Column override (if auto-detect fails)
    parser.add_argument("--text_col", type=str, default=None,
                        help="Column name or index for the original text")
    parser.add_argument("--corrected_col", type=str, default=None,
                        help="Column name or index for the corrected text")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Parse CEFR filter
    cefr_filter = None
    if args.cefr_filter:
        cefr_filter = set(args.cefr_filter.upper().split(","))

    l1_filter = None
    if args.l1_filter:
        l1_filter = set(x.strip().lower() for x in args.l1_filter.split(","))

    # --- Read CSV ---
    logger.info(f"Reading {input_path}...")

    rows_raw = []
    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            rows_raw.append(row)

    logger.info(f"Read {len(rows_raw)} essays")
    logger.info(f"Header ({len(header)} columns): {header[:10]}...")

    # --- Detect columns ---
    col_map = detect_columns(header)
    if not col_map.get("original_text"):
        logger.info("Header-based detection incomplete, trying positional detection...")
        fallback = detect_columns_by_position(header, rows_raw[:100])
        for k, v in fallback.items():
            if k not in col_map:
                col_map[k] = v

    # Manual overrides
    if args.text_col:
        col_map["original_text"] = (
            int(args.text_col) if args.text_col.isdigit()
            else header.index(args.text_col)
        )
    if args.corrected_col:
        col_map["corrected_text"] = (
            int(args.corrected_col) if args.corrected_col.isdigit()
            else header.index(args.corrected_col)
        )

    logger.info(f"Column mapping: { {k: (header[v] if v < len(header) else v) for k, v in col_map.items()} }")

    if "original_text" not in col_map:
        logger.error(
            "Could not detect the text column! Please specify --text_col.\n"
            f"Available columns: {list(enumerate(header))}"
        )
        sys.exit(1)

    text_idx = col_map["original_text"]
    corr_idx = col_map.get("corrected_text")
    cefr_idx = col_map.get("cefr_level")
    l1_idx = col_map.get("l1")
    topic_idx = col_map.get("topic")

    # --- Process essays into sentences ---
    output_rows = []
    essay_count = 0
    skipped_filter = 0

    for row in rows_raw:
        if args.max_essays and essay_count >= args.max_essays:
            break

        if text_idx >= len(row):
            continue

        # Extract metadata
        cefr = row[cefr_idx].strip() if (cefr_idx is not None and cefr_idx < len(row)) else ""
        l1 = row[l1_idx].strip() if (l1_idx is not None and l1_idx < len(row)) else ""
        topic = row[topic_idx].strip() if (topic_idx is not None and topic_idx < len(row)) else ""

        # Apply filters
        if cefr_filter and cefr.upper() not in cefr_filter:
            skipped_filter += 1
            continue
        if l1_filter and l1.lower() not in l1_filter:
            skipped_filter += 1
            continue

        original_text = clean_text(row[text_idx])
        corrected_text = clean_text(row[corr_idx]) if (corr_idx is not None and corr_idx < len(row)) else ""

        if not original_text:
            continue

        # Split into sentences
        orig_sentences = split_into_sentences(original_text)

        for sent_idx, sentence in enumerate(orig_sentences):
            words = sentence.split()
            if len(words) < args.min_words or len(words) > args.max_words:
                continue

            output_rows.append({
                "essay_id": essay_count,
                "sentence_idx": sent_idx,
                "sentence": sentence,
                "cefr_level": cefr,
                "l1_language": l1,
                "topic": topic,
                "word_count": len(words),
            })

        essay_count += 1

    logger.info(f"Processed {essay_count} essays -> {len(output_rows)} sentences")
    if skipped_filter:
        logger.info(f"Skipped {skipped_filter} essays due to filters")

    # --- Write output ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["essay_id", "sentence_idx", "sentence", "cefr_level",
                  "l1_language", "topic", "word_count"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    logger.info(f"Saved to {output_path}")

    # --- Print summary ---
    if output_rows:
        cefr_dist = {}
        l1_dist = {}
        for r in output_rows:
            cefr_dist[r["cefr_level"]] = cefr_dist.get(r["cefr_level"], 0) + 1
            l1_dist[r["l1_language"]] = l1_dist.get(r["l1_language"], 0) + 1

        logger.info(f"\nCEFR distribution:")
        for k in sorted(cefr_dist.keys()):
            logger.info(f"  {k}: {cefr_dist[k]}")

        logger.info(f"\nTop 10 L1 languages:")
        for k, v in sorted(l1_dist.items(), key=lambda x: -x[1])[:10]:
            logger.info(f"  {k}: {v}")

        logger.info(f"\nSample sentences:")
        for r in output_rows[:5]:
            logger.info(f"  [{r['cefr_level']}] {r['sentence'][:80]}...")


if __name__ == "__main__":
    main()
