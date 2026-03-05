"""
Step 4: Error annotation with ERRANT.

ERRANT (ERRor ANnotation Toolkit) aligns original and corrected sentences
and classifies each edit by error type (e.g., M:DET = Missing Determiner,
R:VERB:SVA = Replacement Verb Subject-Verb Agreement, etc.).

This module wraps ERRANT to produce structured error annotations.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import errant
    ERRANT_AVAILABLE = True
except ImportError:
    ERRANT_AVAILABLE = False
    logger.warning("ERRANT not installed. Install with: pip install errant")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


@dataclass
class ErrorAnnotation:
    """A single error found by ERRANT."""
    original_tokens: str        # The erroneous span
    corrected_tokens: str       # The corrected span
    error_type: str             # ERRANT error type code, e.g. "M:DET", "R:VERB:SVA"
    start_offset: int           # Token start index in original
    end_offset: int             # Token end index in original


@dataclass
class SentenceAnnotation:
    """All errors found in one sentence."""
    original: str
    corrected: str
    errors: List[ErrorAnnotation] = field(default_factory=list)
    num_errors: int = 0

    # Breakdown by error category
    error_type_counts: Dict[str, int] = field(default_factory=dict)


class ERRANTAnnotator:
    """Wrapper around ERRANT for batch error annotation."""

    def __init__(self, lang: str = "en"):
        if not ERRANT_AVAILABLE:
            raise ImportError(
                "ERRANT is required. Install with:\n"
                "  pip install errant\n"
                "  python -m spacy download en_core_web_sm"
            )
        self.annotator = errant.load(lang)
        logger.info("ERRANT annotator loaded")

    def annotate_pair(self, original: str, corrected: str) -> SentenceAnnotation:
        """
        Annotate errors between an original and corrected sentence.

        Returns:
            SentenceAnnotation with all errors found.
        """
        orig_parsed = self.annotator.parse(original)
        corr_parsed = self.annotator.parse(corrected)

        edits = self.annotator.annotate(orig_parsed, corr_parsed)

        errors = []
        type_counts: Dict[str, int] = {}

        for edit in edits:
            # Skip "noop" edits (no change)
            if edit.type == "noop":
                continue

            err = ErrorAnnotation(
                original_tokens=edit.o_str,
                corrected_tokens=edit.c_str,
                error_type=edit.type,
                start_offset=edit.o_start,
                end_offset=edit.o_end,
            )
            errors.append(err)

            # Count by type
            type_counts[edit.type] = type_counts.get(edit.type, 0) + 1

        return SentenceAnnotation(
            original=original,
            corrected=corrected,
            errors=errors,
            num_errors=len(errors),
            error_type_counts=type_counts,
        )

    def annotate_batch(
        self,
        originals: List[str],
        correcteds: List[str],
    ) -> List[SentenceAnnotation]:
        """Annotate a batch of sentence pairs."""
        assert len(originals) == len(correcteds), "Mismatched lengths"

        annotations = []
        for orig, corr in zip(originals, correcteds):
            try:
                ann = self.annotate_pair(orig, corr)
            except Exception as e:
                logger.warning(f"ERRANT annotation failed for: '{orig[:50]}...' -> {e}")
                ann = SentenceAnnotation(original=orig, corrected=corr)
            annotations.append(ann)

        return annotations


def summarize_errors(annotations: List[SentenceAnnotation]) -> Dict:
    """
    Aggregate error statistics across all annotated sentences.

    Returns:
        Dict with summary statistics.
    """
    total_errors = 0
    total_sentences = len(annotations)
    sentences_with_errors = 0
    global_type_counts: Dict[str, int] = {}
    errors_per_sentence = []

    for ann in annotations:
        total_errors += ann.num_errors
        errors_per_sentence.append(ann.num_errors)
        if ann.num_errors > 0:
            sentences_with_errors += 1
        for etype, count in ann.error_type_counts.items():
            global_type_counts[etype] = global_type_counts.get(etype, 0) + count

    avg_errors = total_errors / max(total_sentences, 1)

    # Sort error types by frequency
    sorted_types = sorted(global_type_counts.items(), key=lambda x: -x[1])

    return {
        "total_sentences": total_sentences,
        "total_errors": total_errors,
        "sentences_with_errors": sentences_with_errors,
        "error_rate": sentences_with_errors / max(total_sentences, 1),
        "avg_errors_per_sentence": avg_errors,
        "error_type_counts": dict(sorted_types),
        "errors_per_sentence": errors_per_sentence,
        "top_10_error_types": sorted_types[:10],
    }
