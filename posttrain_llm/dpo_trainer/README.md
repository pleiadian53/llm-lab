# DPO Trainer

A reusable package for Direct Preference Optimization (DPO) training of language models.

## Overview

DPO Trainer provides a simplified interface for training language models using Direct Preference Optimization. DPO is a method for aligning language models with human preferences without needing an explicit reward model.

## Features

- **DPOTrainerWrapper**: High-level wrapper for DPO training
- **Identity Shift Training**: Change a model's self-identification
- **Custom Dataset Builders**: Create preference datasets programmatically
- **CLI Interface**: Command-line tools for training and testing
- **GPU Support**: CUDA and Apple Silicon (MPS) acceleration

## Installation

The package is part of the `llm-lab` repository. Ensure you have the required dependencies:

```bash
pip install transformers trl datasets torch typer rich
```

## Quick Start

### Python API

```python
from dpo_trainer import DPOTrainerWrapper, DPOTrainingConfig

# Basic DPO training
trainer = DPOTrainerWrapper(
    model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
    dataset_name="banghua/DL-DPO-Dataset",
)
trainer.train()
trainer.save_model("./my_dpo_model")
```

### CLI

```bash
# Train with DPO
python -m dpo_trainer train \
    --model HuggingFaceTB/SmolLM2-135M-Instruct \
    --dataset banghua/DL-DPO-Dataset \
    --output ./dpo_output

# Identity shift training
python -m dpo_trainer identity-shift \
    --model HuggingFaceTB/SmolLM2-135M-Instruct \
    --original-name Qwen \
    --new-name "Deep Qwen"

# Test a model
python -m dpo_trainer test \
    --model ./dpo_output \
    --question "What is your name?"
```

## Package Structure

```
dpo_trainer/
├── __init__.py          # Package exports
├── __main__.py          # Module entry point
├── cli.py               # Command-line interface
├── core/
│   ├── __init__.py
│   ├── model_loader.py  # Model loading utilities
│   ├── trainer.py       # DPOTrainerWrapper
│   ├── inference.py     # Generation utilities
│   └── dataset.py       # Dataset builders
├── utils/
│   ├── __init__.py
│   └── display.py       # Display utilities
└── docs/
    └── README.md        # Documentation
```

## Key Components

### DPOTrainerWrapper

The main class for DPO training:

```python
from dpo_trainer import DPOTrainerWrapper, DPOTrainingConfig

config = DPOTrainingConfig(
    beta=0.2,              # DPO beta parameter
    learning_rate=5e-5,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
)

trainer = DPOTrainerWrapper(
    model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
    dataset_name="banghua/DL-DPO-Dataset",
    training_config=config,
    use_gpu=True,
)

metrics = trainer.train()
trainer.save_model("./output")
```

### Identity Shift Dataset

Build a dataset for changing a model's identity:

```python
from dpo_trainer import build_identity_shift_dataset, load_model_and_tokenizer

model, tokenizer = load_model_and_tokenizer("HuggingFaceTB/SmolLM2-135M-Instruct")

dataset = build_identity_shift_dataset(
    model, tokenizer,
    original_name="Qwen",
    new_name="Deep Qwen",
    max_samples=100,
)
```

### Inference

Test trained models:

```python
from dpo_trainer import load_model_and_tokenizer, test_model

model, tokenizer = load_model_and_tokenizer("./dpo_output")

questions = [
    "What is your name?",
    "Who created you?",
]

test_model(model, tokenizer, questions, title="DPO Model Output")
```

## Relationship to Lesson 5

This package is a refactored version of the `Lesson_5.ipynb` notebook from the post-training LLM course. The notebook demonstrated:

1. Loading an instruct model
2. Building a DPO dataset for identity shift
3. Training with DPOTrainer
4. Testing the trained model

This package provides the same functionality in a reusable, modular format.

## Documentation

See the `docs/` directory for detailed documentation:

- `docs/README.md` - API reference and examples
- `docs/DPO_explainer.md` - Technical explanation of DPO
- `docs/DPO_for_computational_biology.md` - DPO in biology
- `docs/DPO_for_splicing_prediction.md` - DPO for splice site prediction

## References

1. Rafailov et al. (2023). "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
2. [TRL Library](https://github.com/huggingface/trl)
