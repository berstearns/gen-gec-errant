"""
Configuration for the L2 Error Analysis Pipeline.

Pipeline overview:
1. Load EFCAMDAT sentences
2. Generate continuations with base GPT-2 and learner-tuned GPT-2 (+ Pythia variants)
3. Run GEC (Gemma instruction-tuned / coedit) on generated text
4. Annotate errors with ERRANT
5. Compute perplexity for each generation
6. Aggregate and analyze results
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for a single model in the pipeline."""
    name: str                          # Human-readable label, e.g. "gpt2-base"
    hf_model_id: str                   # HuggingFace model ID, e.g. "gpt2"
    checkpoint_path: Optional[str] = None  # Path to .pt checkpoint (for learner-tuned)
    model_family: str = "gpt2"         # "gpt2" or "pythia"
    is_learner_tuned: bool = False     # Whether this is the learner-tuned variant


@dataclass
class GenerationConfig:
    """Parameters for text generation."""
    max_new_tokens: int = 50
    min_new_tokens: int = 10
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    do_sample: bool = True
    num_return_sequences: int = 1
    repetition_penalty: float = 1.2


@dataclass
class GECConfig:
    """Configuration for grammatical error correction."""
    # Primary GEC: instruction-tuned LLM
    llm_model_id: str = "google/gemma-2-2b-it"
    # Secondary GEC (optional): dedicated GEC model
    dedicated_gec_model_id: str = "grammarly/coedit-large"
    use_dedicated_gec: bool = True     # Use dedicated GEC alongside LLM
    gec_prompt_template: str = (
        "Correct any grammatical errors in the following sentence. "
        "Only fix grammar — do not change meaning, vocabulary, or style. "
        "If the sentence is already correct, return it unchanged.\n\n"
        "Sentence: {sentence}\n\n"
        "Corrected sentence:"
    )


@dataclass
class PipelineConfig:
    """Master configuration for the full pipeline."""
    # --- Paths ---
    data_path: str = "data/efcamdat_sentences.txt"       # One sentence per line
    output_dir: str = "results"
    learner_checkpoint_path: str = ""                      # SET THIS to your .pt file

    # --- Models to evaluate ---
    models: list = field(default_factory=lambda: [
        # GPT-2 family
        ModelConfig(name="gpt2-base",       hf_model_id="gpt2",        model_family="gpt2"),
        ModelConfig(name="gpt2-medium",     hf_model_id="gpt2-medium", model_family="gpt2"),
        # Learner-tuned (checkpoint loaded on top of gpt2)
        ModelConfig(
            name="gpt2-learner",
            hf_model_id="gpt2",
            checkpoint_path="SET_YOUR_PATH",   # <-- overridden at runtime
            model_family="gpt2",
            is_learner_tuned=True,
        ),
        # Pythia family
        ModelConfig(name="pythia-70m",  hf_model_id="EleutherAI/pythia-70m",  model_family="pythia"),
        ModelConfig(name="pythia-160m", hf_model_id="EleutherAI/pythia-160m", model_family="pythia"),
        ModelConfig(name="pythia-410m", hf_model_id="EleutherAI/pythia-410m", model_family="pythia"),
    ])

    generation: GenerationConfig = field(default_factory=GenerationConfig)
    gec: GECConfig = field(default_factory=GECConfig)

    # --- Analysis ---
    max_sentences: Optional[int] = None   # Limit for debugging; None = all
    batch_size: int = 8
    device: str = "auto"                  # "auto", "cuda", "cpu"
    seed: int = 42
