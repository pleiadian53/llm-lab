#!/usr/bin/env python3
"""
Training script for Online RL fine-tuning on math datasets.

This script demonstrates how to use the Online RL Trainer to fine-tune
an LLM on math problems using GRPO, RLOO, or PPO.

Usage:
    # Basic GRPO training on GSM8K
    python train_math.py --model Qwen/Qwen2.5-0.5B-Instruct --trainer grpo
    
    # RLOO training with more generations
    python train_math.py --trainer rloo --num-generations 8 --use-gpu
    
    # Small model for testing
    python train_math.py --model HuggingFaceTB/SmolLM2-135M-Instruct --num-samples 10
"""

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from posttrain_llm.online_rl_trainer import (
    TrainerConfig,
    OnlineRLTrainer,
    MathRewardFunction,
    GSM8KLoader,
)
from posttrain_llm.online_rl_trainer.core.config import TrainerType


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an LLM on math problems using Online RL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="HuggingFaceTB/SmolLM2-135M-Instruct",
        help="Model name or path"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./outputs/online_rl_math",
        help="Output directory for checkpoints"
    )
    
    # Trainer arguments
    parser.add_argument(
        "--trainer", "-t",
        type=str,
        choices=["grpo", "rloo", "ppo"],
        default="grpo",
        help="Trainer type"
    )
    parser.add_argument(
        "--num-generations", "-g",
        type=int,
        default=4,
        help="Number of generations per prompt (G in GRPO)"
    )
    
    # Training arguments
    parser.add_argument(
        "--learning-rate", "-lr",
        type=float,
        default=5e-6,
        help="Learning rate"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=1,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1,
        help="Per-device batch size"
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="Gradient accumulation steps"
    )
    
    # Data arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        help="Dataset name (gsm8k, math, or custom path)"
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=None,
        help="Number of training samples (None for all)"
    )
    
    # Hardware arguments
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for training"
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 precision"
    )
    
    # Logging
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=2,
        help="Log every N steps"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Online RL Training for Math")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Trainer: {args.trainer.upper()}")
    print(f"Dataset: {args.dataset}")
    print(f"GPU: {args.use_gpu}")
    print("=" * 60)
    
    # Load dataset
    print("\n📚 Loading dataset...")
    loader = GSM8KLoader()
    train_dataset = loader.load("train", num_samples=args.num_samples)
    print(f"Loaded {len(train_dataset)} training examples")
    
    # Create reward function
    print("\n🎯 Creating reward function...")
    reward_fn = MathRewardFunction(normalize_numbers=True)
    print(f"Using reward function: {reward_fn.name}")
    
    # Create config
    print("\n⚙️ Creating trainer configuration...")
    config = TrainerConfig(
        trainer_type=TrainerType(args.trainer),
        model_name_or_path=args.model,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        num_train_epochs=args.epochs,
        use_gpu=args.use_gpu,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
    )
    
    # Create trainer
    print("\n🚀 Initializing trainer...")
    trainer = OnlineRLTrainer(
        config=config,
        reward_fn=reward_fn,
        train_dataset=train_dataset,
    )
    
    # Train
    print("\n🏋️ Starting training...")
    trainer.train()
    
    # Save model
    print("\n💾 Saving model...")
    trainer.save_model()
    
    print("\n✅ Training complete!")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
