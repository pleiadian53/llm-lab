"""Core modules for DPO training."""

from .model_loader import ModelLoader, load_model_and_tokenizer
from .trainer import DPOTrainerWrapper, DPOTrainingConfig
from .inference import generate_response, test_model, batch_generate
from .dataset import (
    build_dpo_dataset,
    build_identity_shift_dataset,
    load_dpo_dataset,
    DPODatasetBuilder,
)

__all__ = [
    "ModelLoader",
    "load_model_and_tokenizer",
    "DPOTrainerWrapper",
    "DPOTrainingConfig",
    "generate_response",
    "test_model",
    "batch_generate",
    "build_dpo_dataset",
    "build_identity_shift_dataset",
    "load_dpo_dataset",
    "DPODatasetBuilder",
]
