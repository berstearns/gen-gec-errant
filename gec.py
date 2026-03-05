"""
Step 3: Grammatical Error Correction.

Two GEC approaches:
1. Instruction-tuned LLM (e.g. Gemma-2-2b-it) — flexible, good for diverse errors
2. Dedicated GEC model (e.g. grammarly/coedit-large) — purpose-built, more reliable

Both produce corrected sentences that are then compared with originals via ERRANT.
"""

import logging
import re
from typing import List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from config import GECConfig

logger = logging.getLogger(__name__)


class LLMCorrector:
    """GEC using an instruction-tuned LLM (e.g., Gemma)."""

    def __init__(self, config: GECConfig, device: torch.device):
        self.config = config
        self.device = device

        logger.info(f"Loading LLM corrector: {config.llm_model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.llm_model_id,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        ).to(device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def correct(self, sentences: List[str]) -> List[str]:
        """Correct a list of sentences using the LLM."""
        corrected = []
        for sent in sentences:
            prompt = self.config.gec_prompt_template.format(sentence=sent)

            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=len(sent.split()) + 20,  # slight headroom
                temperature=0.1,       # low temp for faithful correction
                do_sample=False,       # greedy for consistency
                pad_token_id=self.tokenizer.pad_token_id,
            )

            # Decode only the new tokens
            prompt_len = inputs["input_ids"].shape[1]
            result = self.tokenizer.decode(
                outputs[0][prompt_len:], skip_special_tokens=True
            ).strip()

            # Clean up: take only the first sentence/line
            result = result.split("\n")[0].strip()
            # Remove any trailing explanation the model might add
            result = re.split(r"(?:Explanation|Note|Reason):", result)[0].strip()

            # Fallback: if the model returns empty or garbage, keep original
            if not result or len(result) < 3:
                result = sent

            corrected.append(result)

        return corrected


class DedicatedGECCorrector:
    """GEC using a purpose-built seq2seq model (e.g., coedit-large)."""

    def __init__(self, config: GECConfig, device: torch.device):
        self.config = config
        self.device = device

        model_id = config.dedicated_gec_model_id
        logger.info(f"Loading dedicated GEC model: {model_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
        self.model.eval()

    @torch.no_grad()
    def correct(self, sentences: List[str]) -> List[str]:
        """Correct a list of sentences (batched)."""
        if not sentences:
            return []

        input_texts = [f"Fix grammatical errors in this sentence: {s}" for s in sentences]

        inputs = self.tokenizer(
            input_texts, return_tensors="pt", truncation=True,
            max_length=512, padding=True,
        ).to(self.device)

        max_tok = max(len(s.split()) for s in sentences) + 20

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tok,
            num_beams=4,
        )

        corrected = []
        for i, sent in enumerate(sentences):
            result = self.tokenizer.decode(outputs[i], skip_special_tokens=True).strip()
            if not result or len(result) < 3:
                result = sent
            corrected.append(result)

        return corrected


def load_gec_corrector(
    config: GECConfig,
    device: torch.device,
    method: str = "dedicated",
) -> object:
    """
    Factory to load the appropriate GEC corrector.

    Args:
        method: "llm" for instruction-tuned LLM, "dedicated" for coedit-style model
    """
    if method == "llm":
        return LLMCorrector(config, device)
    elif method == "dedicated":
        return DedicatedGECCorrector(config, device)
    else:
        raise ValueError(f"Unknown GEC method: {method}")
