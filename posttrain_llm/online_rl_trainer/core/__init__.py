"""
Online RL Trainer - Core modules for reinforcement learning fine-tuning of LLMs.

Supports multiple training algorithms:
- GRPO (Group Relative Policy Optimization)
- RLOO (REINFORCE Leave-One-Out)
- PPO (Proximal Policy Optimization)
"""

from .config import TrainerConfig, get_trainer_config
from .rewards import (
    RewardFunction,
    MathRewardFunction,
    CodeRewardFunction,
    FormatRewardFunction,
    CompositeRewardFunction,
)
from .data import DatasetLoader, GSM8KLoader
from .trainer import OnlineRLTrainer
from .inference import ModelEvaluator

__all__ = [
    "TrainerConfig",
    "get_trainer_config",
    "RewardFunction",
    "MathRewardFunction", 
    "CodeRewardFunction",
    "FormatRewardFunction",
    "CompositeRewardFunction",
    "DatasetLoader",
    "GSM8KLoader",
    "OnlineRLTrainer",
    "ModelEvaluator",
]
