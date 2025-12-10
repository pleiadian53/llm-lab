"""
Online RL Trainer - A modular framework for reinforcement learning fine-tuning of LLMs.

This package provides a unified interface for training LLMs using various
Online RL algorithms including GRPO, RLOO, and PPO.

Example usage:
    from posttrain_llm.online_rl_trainer import (
        TrainerConfig,
        OnlineRLTrainer,
        MathRewardFunction,
        GSM8KLoader,
    )
    
    # Load dataset
    loader = GSM8KLoader()
    train_data = loader.load("train", num_samples=100)
    
    # Create reward function
    reward_fn = MathRewardFunction()
    
    # Configure and train
    config = TrainerConfig(
        trainer_type="grpo",
        model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
        num_generations=4,
    )
    
    trainer = OnlineRLTrainer(config, reward_fn, train_data)
    trainer.train()
"""

from .core import (
    # Config
    TrainerConfig,
    get_trainer_config,
    
    # Rewards
    RewardFunction,
    MathRewardFunction,
    CodeRewardFunction,
    FormatRewardFunction,
    CompositeRewardFunction,
    
    # Data
    DatasetLoader,
    GSM8KLoader,
    
    # Trainer
    OnlineRLTrainer,
    
    # Inference
    ModelEvaluator,
)

__version__ = "0.1.0"
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
