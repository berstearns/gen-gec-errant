"""GEC runner: correct generated text using LLM or dedicated GEC model."""

import logging
import re
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from gen_gec_errant.gec.config import GECConfig

logger = logging.getLogger(__name__)


def _get_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


class LLMCorrector:
    """GEC using an instruction-tuned LLM (e.g., Gemma)."""

    def __init__(self, config: GECConfig, device: torch.device):
        self.config = config
        self.device = device

        logger.info("Loading LLM corrector: %s", config.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        ).to(device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def correct(self, sentences: List[str]) -> List[str]:
        corrected = []
        for sent in sentences:
            prompt = self.config.prompt_template.format(sentence=sent)

            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=len(sent.split()) + 20,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            prompt_len = inputs["input_ids"].shape[1]
            result = self.tokenizer.decode(
                outputs[0][prompt_len:], skip_special_tokens=True
            ).strip()

            result = result.split("\n")[0].strip()
            result = re.split(r"(?:Explanation|Note|Reason):", result)[0].strip()

            if not result or len(result) < 3:
                result = sent

            corrected.append(result)
        return corrected


class DedicatedGECCorrector:
    """GEC using a purpose-built seq2seq model (e.g., coedit-large)."""

    def __init__(self, config: GECConfig, device: torch.device):
        self.config = config
        self.device = device

        logger.info("Loading dedicated GEC model: %s", config.model_id)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            config.model_id, torch_dtype=dtype,
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def correct(self, sentences: List[str]) -> List[str]:
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


def load_gec_corrector(config: GECConfig, device: torch.device):
    """Factory to load the appropriate GEC corrector."""
    if config.method == "llm":
        return LLMCorrector(config, device)
    elif config.method == "dedicated":
        return DedicatedGECCorrector(config, device)
    else:
        raise ValueError(f"Unknown GEC method: {config.method}")


def run_gec(config: GECConfig, generation_results: dict) -> dict:
    """
    Run GEC on generation results.

    Args:
        config: GECConfig
        generation_results: Dict with keys continuations, full_texts, etc.

    Returns:
        Updated dict with corrected_continuations and corrected_full_texts added.
    """
    device = _get_device(config.device)
    corrector = load_gec_corrector(config, device)

    logger.info("Correcting outputs (method=%s)...", config.method)

    # Correct continuations in batches
    corrected_continuations: list[str] = []
    for i in range(0, len(generation_results["continuations"]), config.batch_size):
        batch = generation_results["continuations"][i : i + config.batch_size]
        corrected_continuations.extend(corrector.correct(batch))

    # Correct full texts in batches
    corrected_full_texts: list[str] = []
    for i in range(0, len(generation_results["full_texts"]), config.batch_size):
        batch = generation_results["full_texts"][i : i + config.batch_size]
        corrected_full_texts.extend(corrector.correct(batch))

    generation_results["corrected_continuations"] = corrected_continuations
    generation_results["corrected_full_texts"] = corrected_full_texts

    logger.info("Corrected %d sentences", len(corrected_continuations))

    del corrector
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return generation_results
