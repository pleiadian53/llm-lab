# Online RL Trainer

A modular framework for reinforcement learning fine-tuning of Large Language Models.

## Overview

This package provides a unified interface for training LLMs using various Online RL algorithms:

| Trainer | Description | Best For |
|---------|-------------|----------|
| **GRPO** | Group Relative Policy Optimization | Memory-efficient, reasoning tasks |
| **RLOO** | REINFORCE Leave-One-Out | Variance reduction, stable training |
| **PPO** | Proximal Policy Optimization | Fine-grained control, complex rewards |

## Installation

The package requires TRL (Transformer Reinforcement Learning):

```bash
pip install trl transformers datasets torch
```

## Quick Start

### Training

```python
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

# Configure trainer
config = TrainerConfig(
    trainer_type="grpo",
    model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
    num_generations=4,
    learning_rate=5e-6,
)

# Train
trainer = OnlineRLTrainer(config, reward_fn, train_data)
trainer.train()
trainer.save_model("./outputs/my_model")
```

### Evaluation

```python
from posttrain_llm.online_rl_trainer import (
    ModelEvaluator,
    MathRewardFunction,
    GSM8KLoader,
)

# Load test data
loader = GSM8KLoader()
test_data = loader.load("test", num_samples=50)

# Evaluate
evaluator = ModelEvaluator("./outputs/my_model")
results = evaluator.evaluate(test_data, MathRewardFunction())
print(f"Accuracy: {results['accuracy']:.2%}")
```

## Command Line Usage

### Training Script

```bash
# Basic GRPO training
python -m posttrain_llm.online_rl_trainer.examples.train_math \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --trainer grpo \
    --num-generations 4

# RLOO with GPU
python -m posttrain_llm.online_rl_trainer.examples.train_math \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --trainer rloo \
    --use-gpu \
    --num-samples 1000

# Small model for testing
python -m posttrain_llm.online_rl_trainer.examples.train_math \
    --model HuggingFaceTB/SmolLM2-135M-Instruct \
    --num-samples 10
```

### Evaluation Script

```bash
# Evaluate a model
python -m posttrain_llm.online_rl_trainer.examples.evaluate_model \
    --model ./outputs/online_rl_math \
    --num-samples 20

# Compare two models
python -m posttrain_llm.online_rl_trainer.examples.evaluate_model \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --compare ./outputs/trained_model \
    --num-samples 50
```

## Module Structure

```text
online_rl_trainer/
├── __init__.py           # Package exports
├── README.md             # This file
├── core/
│   ├── __init__.py       # Core module exports
│   ├── config.py         # TrainerConfig, presets
│   ├── rewards.py        # Reward functions
│   ├── data.py           # Dataset loaders
│   ├── trainer.py        # OnlineRLTrainer
│   └── inference.py      # ModelEvaluator
└── examples/
    ├── train_math.py     # Training driver script
    └── evaluate_model.py # Evaluation driver script
```

## Reward Functions

### Built-in Rewards

| Reward | Description | Use Case |
|--------|-------------|----------|
| `MathRewardFunction` | Extracts `\boxed{}` answer, compares to ground truth | Math problems |
| `CodeRewardFunction` | Runs code against test cases | Coding problems |
| `FormatRewardFunction` | Checks format compliance | Any structured output |
| `CompositeRewardFunction` | Weighted combination of rewards | Multi-objective |

### Custom Rewards

```python
from posttrain_llm.online_rl_trainer import RewardFunction

class MyReward(RewardFunction):
    @property
    def name(self):
        return "my_reward"
    
    def __call__(self, completions, **kwargs):
        rewards = []
        for completion in completions:
            content = completion[0]["content"]
            # Your reward logic here
            reward = 1.0 if "good" in content else 0.0
            rewards.append(reward)
        return rewards
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trainer_type` | `"grpo"` | Trainer algorithm (grpo, rloo, ppo) |
| `model_name_or_path` | - | Model to fine-tune |
| `num_generations` | `4` | Samples per prompt (G in GRPO) |
| `learning_rate` | `5e-6` | Learning rate |
| `num_train_epochs` | `1` | Training epochs |
| `kl_coef` | `0.1` | KL penalty coefficient |
| `clip_range` | `0.2` | PPO clip range (ε) |
| `use_gpu` | `False` | Use GPU acceleration |
| `bf16` | `False` | Use bfloat16 precision |

## Supported Datasets

- **GSM8K**: Grade school math (8K problems)
- **MATH**: Competition mathematics
- **Custom**: Any dataset with prompt/answer columns

## References

- [TRL Documentation](https://huggingface.co/docs/trl)
- [GRPO Paper (DeepSeekMath)](https://arxiv.org/abs/2402.03300)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [DeepSeek-R1 Paper](https://arxiv.org/abs/2501.12948)
