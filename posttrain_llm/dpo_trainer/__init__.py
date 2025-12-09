"""
DPO Trainer - Direct Preference Optimization Training Package

A reusable package for training language models using Direct Preference Optimization (DPO).
DPO is a method for aligning language models with human preferences without explicit reward modeling.

Key Features:
    - Identity shift training (change model's self-identification)
    - Preference-based fine-tuning
    - Support for custom preference datasets
    - Integration with HuggingFace Transformers and TRL

Example - Basic DPO Training:
    >>> from dpo_trainer import DPOTrainerWrapper
    >>> 
    >>> trainer = DPOTrainerWrapper(
    ...     model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
    ...     dataset_name="banghua/DL-DPO-Dataset",
    ... )
    >>> trainer.train()

Example - Identity Shift Training:
    >>> from dpo_trainer import DPOTrainerWrapper, build_identity_shift_dataset
    >>> 
    >>> # Build preference dataset for identity shift
    >>> dataset = build_identity_shift_dataset(
    ...     model, tokenizer,
    ...     original_name="Qwen",
    ...     new_name="Deep Qwen",
    ...     raw_dataset=raw_ds,
    ... )
    >>> 
    >>> trainer = DPOTrainerWrapper(
    ...     model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
    ...     train_dataset=dataset,
    ... )
    >>> trainer.train()

References:
    - Rafailov et al. (2023). "Direct Preference Optimization: Your Language Model 
      is Secretly a Reward Model"
    - TRL Library: https://github.com/huggingface/trl
"""

__version__ = "0.1.0"
__author__ = "pleiadian53"

# Core imports
from dpo_trainer.core.model_loader import ModelLoader, load_model_and_tokenizer
from dpo_trainer.core.trainer import DPOTrainerWrapper, DPOTrainingConfig
from dpo_trainer.core.inference import generate_response, test_model, batch_generate

# Dataset utilities
from dpo_trainer.core.dataset import (
    build_dpo_dataset,
    build_identity_shift_dataset,
    load_dpo_dataset,
    DPODatasetBuilder,
)

# Utils imports
from dpo_trainer.utils.display import print_section, print_training_summary

__all__ = [
    # Core
    "ModelLoader",
    "load_model_and_tokenizer",
    "DPOTrainerWrapper",
    "DPOTrainingConfig",
    "generate_response",
    "test_model",
    "batch_generate",
    # Dataset
    "build_dpo_dataset",
    "build_identity_shift_dataset",
    "load_dpo_dataset",
    "DPODatasetBuilder",
    # Utils
    "print_section",
    "print_training_summary",
]
