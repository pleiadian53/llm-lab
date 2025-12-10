"""
Unified trainer interface for Online RL methods.

Supports GRPO, RLOO, and PPO trainers from TRL.
"""

import warnings
from typing import Optional, Union, Callable, List
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset

from .config import TrainerConfig, TrainerType
from .rewards import RewardFunction


class OnlineRLTrainer:
    """
    Unified interface for Online RL training.
    
    Supports multiple training algorithms:
    - GRPO (Group Relative Policy Optimization)
    - RLOO (REINFORCE Leave-One-Out)
    - PPO (Proximal Policy Optimization)
    """
    
    def __init__(
        self,
        config: TrainerConfig,
        reward_fn: Union[RewardFunction, Callable],
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration
            reward_fn: Reward function (RewardFunction instance or callable)
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
        """
        self.config = config
        self.reward_fn = reward_fn
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model()
        
        # Create the appropriate trainer
        self.trainer = self._create_trainer()
    
    def _load_model(self):
        """Load model and tokenizer."""
        print(f"Loading model: {self.config.model_name_or_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=True
        )
        
        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Determine device and dtype
        if self.config.use_gpu and torch.cuda.is_available():
            device_map = "auto"
            torch_dtype = torch.bfloat16 if self.config.bf16 else torch.float16
        else:
            device_map = "cpu"
            torch_dtype = torch.float32
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True
        )
        
        return model, tokenizer
    
    def _get_trainer_config(self):
        """Get the appropriate trainer config based on trainer type."""
        if self.config.trainer_type == TrainerType.GRPO:
            from trl import GRPOConfig
            return GRPOConfig(
                output_dir=self.config.output_dir,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                num_generations=self.config.num_generations,
                num_train_epochs=self.config.num_train_epochs,
                max_steps=self.config.max_steps if self.config.max_steps > 0 else -1,
                learning_rate=self.config.learning_rate,
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps,
                warmup_ratio=self.config.warmup_ratio,
                weight_decay=self.config.weight_decay,
                no_cuda=not self.config.use_gpu,
                bf16=self.config.bf16 and self.config.use_gpu,
            )
        
        elif self.config.trainer_type == TrainerType.RLOO:
            from trl import RLOOConfig
            return RLOOConfig(
                output_dir=self.config.output_dir,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                num_generations=self.config.num_generations,
                num_train_epochs=self.config.num_train_epochs,
                learning_rate=self.config.learning_rate,
                logging_steps=self.config.logging_steps,
                kl_coef=self.config.kl_coef,
            )
        
        elif self.config.trainer_type == TrainerType.PPO:
            from trl import PPOConfig
            return PPOConfig(
                output_dir=self.config.output_dir,
                batch_size=self.config.per_device_train_batch_size,
                learning_rate=self.config.learning_rate,
                kl_penalty="kl",
                init_kl_coef=self.config.kl_coef,
                cliprange=self.config.clip_range,
            )
        
        else:
            raise ValueError(f"Unknown trainer type: {self.config.trainer_type}")
    
    def _create_trainer(self):
        """Create the appropriate trainer based on config."""
        trainer_config = self._get_trainer_config()
        
        # Convert reward function to callable if needed
        if isinstance(self.reward_fn, RewardFunction):
            reward_callable = self.reward_fn
        else:
            reward_callable = self.reward_fn
        
        if self.config.trainer_type == TrainerType.GRPO:
            from trl import GRPOTrainer
            return GRPOTrainer(
                model=self.model,
                args=trainer_config,
                reward_funcs=reward_callable,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
            )
        
        elif self.config.trainer_type == TrainerType.RLOO:
            from trl import RLOOTrainer
            return RLOOTrainer(
                model=self.model,
                args=trainer_config,
                reward_funcs=reward_callable,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
            )
        
        elif self.config.trainer_type == TrainerType.PPO:
            from trl import PPOTrainer
            # PPO requires a reference model
            ref_model = AutoModelForCausalLM.from_pretrained(
                self.config.ref_model_name_or_path or self.config.model_name_or_path,
                trust_remote_code=True
            )
            return PPOTrainer(
                model=self.model,
                ref_model=ref_model,
                tokenizer=self.tokenizer,
                config=trainer_config,
                dataset=self.train_dataset,
            )
        
        else:
            raise ValueError(f"Unknown trainer type: {self.config.trainer_type}")
    
    def train(self):
        """Run training."""
        print(f"Starting {self.config.trainer_type.value.upper()} training...")
        print(f"  Model: {self.config.model_name_or_path}")
        print(f"  Epochs: {self.config.num_train_epochs}")
        print(f"  Generations per prompt: {self.config.num_generations}")
        print(f"  Learning rate: {self.config.learning_rate}")
        
        self.trainer.train()
        
        print("Training complete!")
        return self.trainer
    
    def save_model(self, output_path: Optional[str] = None):
        """Save the trained model."""
        output_path = output_path or self.config.output_dir
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print(f"Model saved to: {output_path}")
    
    def get_model(self):
        """Get the trained model."""
        return self.model
    
    def get_tokenizer(self):
        """Get the tokenizer."""
        return self.tokenizer


def create_trainer(
    trainer_type: str = "grpo",
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
    reward_fn: Union[RewardFunction, Callable] = None,
    train_dataset: Dataset = None,
    use_gpu: bool = False,
    **kwargs
) -> OnlineRLTrainer:
    """
    Factory function to create an Online RL trainer.
    
    Args:
        trainer_type: One of 'grpo', 'rloo', 'ppo'
        model_name_or_path: Model path or HuggingFace model name
        reward_fn: Reward function
        train_dataset: Training dataset
        use_gpu: Whether to use GPU
        **kwargs: Additional config parameters
        
    Returns:
        OnlineRLTrainer instance
    """
    from .config import get_trainer_config
    
    config = get_trainer_config(
        trainer_type=trainer_type,
        model_name_or_path=model_name_or_path,
        use_gpu=use_gpu,
        **kwargs
    )
    
    return OnlineRLTrainer(
        config=config,
        reward_fn=reward_fn,
        train_dataset=train_dataset,
    )
