"""PEFT (Parameter-Efficient Fine-Tuning) module.

Provides configuration and utilities for various PEFT methods:
- LoRA (Low-Rank Adaptation)
- DoRA (Weight-Decomposed Low-Rank Adaptation)
- VeRA (Vector-based Random Matrix Adaptation)
- QLoRA (Quantized LoRA)
- AdaLoRA (Adaptive LoRA)
- IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)
- Prompt Tuning
- Prefix Tuning
"""

from .config import PEFTConfig, PEFTMethod
from .utils import get_target_modules, print_trainable_parameters

__all__ = [
    "PEFTConfig",
    "PEFTMethod",
    "get_target_modules",
    "print_trainable_parameters",
]
