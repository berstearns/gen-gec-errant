"""
Step 1: Data loading and preprocessing.

Supports:
- Plain text files (one sentence per line)
- CSV/TSV with a 'text' or 'sentence' column
- HuggingFace datasets (efcamdat if available)

Each sentence is used as a *prompt* for generation. We take the first N tokens
as the prompt prefix and ask the model to continue.
"""

import csv
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def load_sentences(
    path: str,
    max_sentences: Optional[int] = None,
    min_words: int = 10,
    max_words: int = 10000,
) -> List[str]:
    """
    Load and filter sentences from a file.

    Args:
        path: Path to data file (.txt, .csv, .tsv)
        max_sentences: Cap on number of sentences (None = all)
        min_words: Minimum word count to keep a sentence
        max_words: Maximum word count to keep a sentence

    Returns:
        List of cleaned sentences
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    raw_sentences = []

    if path.suffix == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            raw_sentences = [line.strip() for line in f if line.strip()]

    elif path.suffix in (".csv", ".tsv"):
        delimiter = "\t" if path.suffix == ".tsv" else ","
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            # Try common column names (in priority order)
            text_col = None
            for row in reader:
                if text_col is None:
                    # Auto-detect text column on first row
                    for candidate in ["sentence", "text", "original_text",
                                      "writing", "learner_text", "corrected"]:
                        if candidate in row and row[candidate].strip():
                            text_col = candidate
                            logger.info(f"Auto-detected text column: '{text_col}'")
                            break
                    if text_col is None:
                        # Fallback: pick the longest string field
                        text_col = max(row.keys(), key=lambda k: len(str(row.get(k, ""))))
                        logger.info(f"Fallback text column: '{text_col}'")

                text = row.get(text_col, "").strip()
                if text:
                    raw_sentences.append(text)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    # Filter by length
    sentences = [
        s for s in raw_sentences
        if min_words <= len(s.split()) <= max_words
    ]

    logger.info(f"Loaded {len(raw_sentences)} raw sentences, {len(sentences)} after filtering")

    if max_sentences is not None:
        sentences = sentences[:max_sentences]
        logger.info(f"Capped to {max_sentences} sentences")

    return sentences


def make_prompts(
    sentences: List[str],
    prompt_ratio: float = 0.5,
    min_prompt_words: int = 3,
) -> List[dict]:
    """
    Split each sentence into a prompt prefix and a reference continuation.

    Args:
        sentences: Full sentences
        prompt_ratio: Fraction of words to use as prompt (0.5 = first half)
        min_prompt_words: Minimum words in the prompt

    Returns:
        List of dicts: {"prompt": str, "reference": str, "full": str}
    """
    items = []
    for sent in sentences:
        words = sent.split()
        split_idx = max(min_prompt_words, int(len(words) * prompt_ratio))
        if split_idx >= len(words):
            split_idx = len(words) - 1  # Leave at least 1 word as reference

        prompt = " ".join(words[:split_idx])
        reference = " ".join(words[split_idx:])

        items.append({
            "prompt": prompt,
            "reference": reference,
            "full": sent,
        })

    return items
