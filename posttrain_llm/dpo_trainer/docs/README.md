# DPO Trainer Documentation

This directory contains documentation for the DPO (Direct Preference Optimization) Trainer package.

## Contents

- **DPO_explainer.md** - Technical explanation of the DPO algorithm
- **DPO_for_computational_biology.md** - DPO use cases in computational biology
- **DPO_for_splicing_prediction.md** - DPO for adaptive splice site prediction

## Quick Start

### Basic DPO Training

```python
from dpo_trainer import DPOTrainerWrapper

trainer = DPOTrainerWrapper(
    model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
    dataset_name="banghua/DL-DPO-Dataset",
)
trainer.train()
```

### Identity Shift Training

```python
from dpo_trainer import DPOTrainerWrapper, build_identity_shift_dataset, load_model_and_tokenizer

# Load model
model, tokenizer = load_model_and_tokenizer("HuggingFaceTB/SmolLM2-135M-Instruct")

# Build preference dataset
dataset = build_identity_shift_dataset(
    model, tokenizer,
    original_name="Qwen",
    new_name="Deep Qwen",
)

# Train
trainer = DPOTrainerWrapper(
    model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
    train_dataset=dataset,
)
trainer.train()
```

### CLI Usage

```bash
# Basic DPO training
python -m dpo_trainer train --model HuggingFaceTB/SmolLM2-135M-Instruct --dataset banghua/DL-DPO-Dataset

# Identity shift training
python -m dpo_trainer identity-shift --model HuggingFaceTB/SmolLM2-135M-Instruct --original-name Qwen --new-name "Deep Qwen"

# Test a trained model
python -m dpo_trainer test --model ./dpo_output --question "What is your name?"

# Compare before/after
python -m dpo_trainer compare --before HuggingFaceTB/SmolLM2-135M-Instruct --after ./dpo_output
```

## Key Concepts

### What is DPO?

Direct Preference Optimization (DPO) is a method for aligning language models with human preferences without needing an explicit reward model. Instead of:

1. Training a reward model on preference data
2. Using RL (PPO) to optimize the policy against the reward model

DPO directly optimizes the policy using a simple binary cross-entropy loss on preference pairs.

### Preference Pairs

DPO requires datasets with "chosen" and "rejected" responses:

```python
{
    "chosen": [
        {"role": "system", "content": "You're a helpful assistant."},
        {"role": "user", "content": "What is your name?"},
        {"role": "assistant", "content": "I'm Deep Qwen, a helpful AI assistant."},
    ],
    "rejected": [
        {"role": "system", "content": "You're a helpful assistant."},
        {"role": "user", "content": "What is your name?"},
        {"role": "assistant", "content": "I'm Qwen, a helpful AI assistant."},
    ]
}
```

### Beta Parameter

The `beta` parameter controls how much the model can deviate from the reference model:
- **Lower beta (0.1)**: More conservative, stays closer to reference
- **Higher beta (0.5)**: More aggressive, allows larger deviations

## API Reference

### DPOTrainerWrapper

Main class for DPO training.

```python
DPOTrainerWrapper(
    model_name: str,                    # HuggingFace model ID or path
    dataset_name: Optional[str],        # HuggingFace dataset ID
    train_dataset: Optional[Dataset],   # Pre-loaded dataset
    training_config: Optional[DPOTrainingConfig],
    ref_model: Optional[PreTrainedModel],  # Reference model (None = implicit)
    use_gpu: bool = False,
    max_samples: Optional[int] = None,
)
```

### DPOTrainingConfig

Configuration dataclass for training.

```python
DPOTrainingConfig(
    beta: float = 0.2,
    learning_rate: float = 5e-5,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    logging_steps: int = 2,
    output_dir: str = "./dpo_output",
)
```

### Dataset Builders

```python
# Identity shift dataset
build_identity_shift_dataset(
    model, tokenizer,
    original_name: str,
    new_name: str,
    raw_dataset: Optional[Dataset] = None,
    dataset_name: str = "mrfakename/identity",
    system_prompt: str = "You're a helpful assistant.",
    max_samples: Optional[int] = None,
)

# Generic DPO dataset
build_dpo_dataset(
    model, tokenizer,
    raw_dataset: Dataset,
    prompt_column: str = "prompt",
    chosen_transform: Optional[Callable] = None,
)

# Load pre-built dataset
load_dpo_dataset(
    dataset_name: str,
    split: str = "train",
    max_samples: Optional[int] = None,
)
```

## References

1. Rafailov et al. (2023). "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
2. TRL Library: https://github.com/huggingface/trl
