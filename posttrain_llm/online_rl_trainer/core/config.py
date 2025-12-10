"""
Configuration module for Online RL trainers.

Supports GRPO, RLOO, and PPO configurations.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
from enum import Enum


class TrainerType(str, Enum):
    """Supported trainer types."""
    GRPO = "grpo"
    RLOO = "rloo"
    PPO = "ppo"


@dataclass
class TrainerConfig:
    """
    Unified configuration for Online RL trainers.
    
    Attributes:
        trainer_type: Type of trainer (grpo, rloo, ppo)
        model_name_or_path: Path to model or HuggingFace model name
        output_dir: Directory to save checkpoints
        
        # Training hyperparameters
        learning_rate: Learning rate for optimizer
        per_device_train_batch_size: Batch size per device
        gradient_accumulation_steps: Number of gradient accumulation steps
        num_train_epochs: Number of training epochs
        max_steps: Maximum training steps (-1 for unlimited)
        
        # RL-specific parameters
        num_generations: Number of generations per prompt (for GRPO/RLOO)
        kl_coef: KL penalty coefficient
        clip_range: PPO clip range (epsilon)
        
        # Hardware
        use_gpu: Whether to use GPU
        bf16: Use bfloat16 precision
        
        # Logging
        logging_steps: Log every N steps
        save_steps: Save checkpoint every N steps
        
        # System prompt for generation
        system_prompt: System prompt to use
    """
    # Trainer selection
    trainer_type: TrainerType = TrainerType.GRPO
    
    # Model
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    output_dir: str = "./outputs/online_rl"
    
    # Training hyperparameters
    learning_rate: float = 5e-6
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 1
    max_steps: int = -1
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # RL-specific parameters
    num_generations: int = 4  # G in GRPO paper, can be 64-128 for full training
    kl_coef: float = 0.1
    clip_range: float = 0.2  # epsilon for PPO clipping
    
    # Hardware
    use_gpu: bool = False
    bf16: bool = False
    
    # Logging and saving
    logging_steps: int = 2
    save_steps: int = 100
    eval_steps: int = 100
    
    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.7
    
    # System prompt
    system_prompt: str = (
        "You are a helpful assistant that solves problems step-by-step. "
        "Always include the final numeric answer inside \\boxed{}."
    )
    
    # Reference model (for KL penalty)
    ref_model_name_or_path: Optional[str] = None  # None means use initial model
    
    def __post_init__(self):
        """Validate configuration."""
        if isinstance(self.trainer_type, str):
            self.trainer_type = TrainerType(self.trainer_type.lower())


def get_trainer_config(
    trainer_type: str = "grpo",
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
    use_gpu: bool = False,
    num_generations: int = 4,
    learning_rate: float = 5e-6,
    num_train_epochs: int = 1,
    **kwargs
) -> TrainerConfig:
    """
    Factory function to create trainer configuration.
    
    Args:
        trainer_type: One of 'grpo', 'rloo', 'ppo'
        model_name_or_path: Model path or HuggingFace model name
        use_gpu: Whether to use GPU
        num_generations: Number of generations per prompt
        learning_rate: Learning rate
        num_train_epochs: Number of epochs
        **kwargs: Additional config parameters
        
    Returns:
        TrainerConfig instance
    """
    return TrainerConfig(
        trainer_type=TrainerType(trainer_type.lower()),
        model_name_or_path=model_name_or_path,
        use_gpu=use_gpu,
        num_generations=num_generations,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        **kwargs
    )


# Preset configurations for common use cases
PRESET_CONFIGS = {
    "grpo_math_small": TrainerConfig(
        trainer_type=TrainerType.GRPO,
        model_name_or_path="HuggingFaceTB/SmolLM2-135M-Instruct",
        num_generations=4,
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
    ),
    "grpo_math_medium": TrainerConfig(
        trainer_type=TrainerType.GRPO,
        model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
        num_generations=8,
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        use_gpu=True,
    ),
    "rloo_math": TrainerConfig(
        trainer_type=TrainerType.RLOO,
        model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
        num_generations=4,
        learning_rate=5e-6,
    ),
}


def get_preset_config(preset_name: str) -> TrainerConfig:
    """Get a preset configuration by name."""
    if preset_name not in PRESET_CONFIGS:
        available = ", ".join(PRESET_CONFIGS.keys())
        raise ValueError(f"Unknown preset: {preset_name}. Available: {available}")
    return PRESET_CONFIGS[preset_name]
