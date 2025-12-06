"""
SFT Trainer - Supervised Fine-Tuning with PEFT Support

A reusable package for supervised fine-tuning of language models with support
for various Parameter-Efficient Fine-Tuning (PEFT) methods.

Supported PEFT Methods:
    - LoRA (Low-Rank Adaptation)
    - DoRA (Weight-Decomposed Low-Rank Adaptation)
    - VeRA (Vector-based Random Matrix Adaptation)
    - QLoRA (Quantized LoRA)
    - AdaLoRA (Adaptive LoRA)
    - IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)
    - Prompt Tuning
    - Prefix Tuning

Example - Full Fine-Tuning:
    >>> from sft_trainer import SFTTrainerWrapper
    >>> 
    >>> trainer = SFTTrainerWrapper(
    ...     model_name="HuggingFaceTB/SmolLM2-135M",
    ...     dataset_name="banghua/DL-SFT-Dataset",
    ... )
    >>> trainer.train()

Example - LoRA Fine-Tuning:
    >>> from sft_trainer import SFTTrainerWrapper
    >>> from sft_trainer.peft import PEFTConfig
    >>> 
    >>> peft_config = PEFTConfig.from_preset("lora_default")
    >>> trainer = SFTTrainerWrapper(
    ...     model_name="HuggingFaceTB/SmolLM2-135M",
    ...     dataset_name="banghua/DL-SFT-Dataset",
    ...     peft_config=peft_config,
    ... )
    >>> trainer.train()
"""

__version__ = "0.1.0"
__author__ = "pleiadian53"

# Core imports
from sft_trainer.core.model_loader import ModelLoader, load_model_and_tokenizer
from sft_trainer.core.trainer import SFTTrainerWrapper, TrainingConfig
from sft_trainer.core.inference import generate_response, test_model, batch_generate

# PEFT imports
from sft_trainer.peft.config import PEFTConfig, PEFTMethod

# Utils imports
from sft_trainer.utils.dataset import load_sft_dataset, display_dataset
from sft_trainer.utils.display import print_section, print_training_summary

__all__ = [
    # Core
    "ModelLoader",
    "load_model_and_tokenizer",
    "SFTTrainerWrapper",
    "TrainingConfig",
    "generate_response",
    "test_model",
    "batch_generate",
    # PEFT
    "PEFTConfig",
    "PEFTMethod",
    # Utils
    "load_sft_dataset",
    "display_dataset",
    "print_section",
    "print_training_summary",
]
