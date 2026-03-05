"""
Step 2: Text generation with base and learner-tuned models.

Handles:
- Loading HuggingFace models (GPT-2, Pythia)
- Loading .pt checkpoints for learner-tuned variants
- Batched generation with configurable parameters
- Perplexity computation
"""

import logging
import math
from typing import Dict, List, Optional

import torch
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

from config import GenerationConfig, ModelConfig

logger = logging.getLogger(__name__)


def get_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def load_model(
    model_config: ModelConfig,
    device: torch.device,
) -> tuple:
    """
    Load a model and tokenizer.

    For learner-tuned models, loads the base HF model first, then applies
    the .pt checkpoint weights.

    Returns:
        (model, tokenizer)
    """
    logger.info(f"Loading model: {model_config.name} ({model_config.hf_model_id})")

    tokenizer = AutoTokenizer.from_pretrained(model_config.hf_model_id)

    # GPT-2 / Pythia don't have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_config.hf_model_id)

    # Load learner-tuned checkpoint if provided
    if model_config.is_learner_tuned and model_config.checkpoint_path:
        logger.info(f"Loading learner checkpoint from: {model_config.checkpoint_path}")
        checkpoint = torch.load(model_config.checkpoint_path, map_location="cpu")

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            # Common formats: {"model_state_dict": ...}, {"state_dict": ...}, or raw state dict
            state_dict = (
                checkpoint.get("model_state_dict")
                or checkpoint.get("state_dict")
                or checkpoint.get("model")
                or checkpoint  # Assume it's already a state dict
            )
        else:
            state_dict = checkpoint

        # Strip "_orig_mod." prefix from keys (added by torch.compile)
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("_orig_mod.", "")
            cleaned_state_dict[new_key] = v

        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            logger.info("Stripped '_orig_mod.' prefix from checkpoint keys (torch.compile artifact)")

        state_dict = cleaned_state_dict

        # Transpose weights that were saved from nn.Linear but need to load
        # into HuggingFace Conv1D (nanoGPT vs HF convention).
        # Conv1D stores [in_features, out_features], Linear stores [out_features, in_features]
        model_state = model.state_dict()
        transposed_keys = []
        for k, v in state_dict.items():
            if k in model_state and v.shape != model_state[k].shape:
                if v.dim() == 2 and v.shape == model_state[k].shape[::-1]:
                    state_dict[k] = v.t()
                    transposed_keys.append(k)

        if transposed_keys:
            logger.info(f"Transposed {len(transposed_keys)} weight tensors (nanoGPT -> HF Conv1D)")

        # Try to load; use strict=False to handle minor mismatches
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys in checkpoint: {missing[:5]}...")
        if unexpected:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected[:5]}...")
        logger.info("Learner checkpoint loaded successfully")

    model = model.to(device)
    model.eval()

    return model, tokenizer


@torch.no_grad()
def generate_continuations(
    model,
    tokenizer,
    prompts: List[str],
    gen_config: GenerationConfig,
    batch_size: int = 8,
    device: torch.device = torch.device("cpu"),
) -> List[str]:
    """
    Generate text continuations for a list of prompts.

    Returns:
        List of generated continuations (prompt text stripped).
    """
    continuations = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        prompt_lengths = inputs["attention_mask"].sum(dim=1)

        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_config.max_new_tokens,
            min_new_tokens=gen_config.min_new_tokens,
            temperature=gen_config.temperature,
            top_k=gen_config.top_k,
            top_p=gen_config.top_p,
            do_sample=gen_config.do_sample,
            num_return_sequences=gen_config.num_return_sequences,
            repetition_penalty=gen_config.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
        )

        for j, output_ids in enumerate(outputs):
            # Strip the prompt tokens to get only the continuation
            prompt_len = prompt_lengths[j].item()
            continuation_ids = output_ids[prompt_len:]
            continuation = tokenizer.decode(continuation_ids, skip_special_tokens=True).strip()
            continuations.append(continuation)

        if (i // batch_size) % 10 == 0:
            logger.info(f"  Generated {min(i + batch_size, len(prompts))}/{len(prompts)}")

    return continuations


@torch.no_grad()
def compute_perplexity(
    model,
    tokenizer,
    texts: List[str],
    batch_size: int = 8,
    device: torch.device = torch.device("cpu"),
) -> List[float]:
    """
    Compute per-sentence perplexity for a list of texts.

    Returns:
        List of perplexity values (one per text).
    """
    perplexities = []
    loss_fn = CrossEntropyLoss(reduction="none")

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        outputs = model(**inputs, labels=inputs["input_ids"])

        # Compute per-token loss manually for per-sentence perplexity
        logits = outputs.logits[:, :-1, :].contiguous()
        labels = inputs["input_ids"][:, 1:].contiguous()
        attention_mask = inputs["attention_mask"][:, 1:].contiguous()

        # Flatten for loss computation
        batch_sz, seq_len, vocab_size = logits.shape
        flat_logits = logits.view(-1, vocab_size)
        flat_labels = labels.view(-1)

        token_losses = loss_fn(flat_logits, flat_labels).view(batch_sz, seq_len)

        # Mask out padding tokens and compute mean loss per sentence
        token_losses = token_losses * attention_mask.float()
        seq_lengths = attention_mask.sum(dim=1).float()
        mean_losses = token_losses.sum(dim=1) / seq_lengths.clamp(min=1)

        # Perplexity = exp(mean loss)
        for loss_val in mean_losses:
            ppl = math.exp(loss_val.item())
            perplexities.append(ppl)

    return perplexities
