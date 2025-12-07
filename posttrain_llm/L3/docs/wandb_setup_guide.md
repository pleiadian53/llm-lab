# Weights & Biases (WandB) Setup Guide

This guide walks you through setting up WandB for experiment tracking with `SFTTrainer` and other ML training workflows.

## What is WandB?

[Weights & Biases](https://wandb.ai) is an experiment tracking platform that helps you:

- **Track experiments**: Log hyperparameters, metrics, and artifacts
- **Visualize results**: Interactive dashboards with loss curves, comparisons
- **Reproduce runs**: Full configuration and code versioning
- **Collaborate**: Share results with team members

## Quick Setup

### Step 1: Create a WandB Account

1. Go to [wandb.ai/site](https://wandb.ai/site)
2. Click **"Sign Up"** (free tier available)
3. You can sign up with GitHub, Google, or email

### Step 2: Get Your API Key

1. After logging in, go to [wandb.ai/authorize](https://wandb.ai/authorize)
2. Copy your API key (it looks like a long alphanumeric string)

> **Important**: Keep your API key secret! Never commit it to version control.

### Step 3: Install WandB

```bash
pip install wandb
```

Or if using conda/mamba:

```bash
mamba install -c conda-forge wandb
```

### Step 4: Authenticate

You have several options to authenticate:

#### Option A: Interactive Login (Recommended for local development)

```bash
wandb login
```

This will prompt you to paste your API key and save it to `~/.netrc`.

#### Option B: Environment Variable (Recommended for servers/CI)

```bash
export WANDB_API_KEY=your_api_key_here
```

Add this to your `~/.bashrc` or `~/.zshrc` for persistence:

```bash
echo 'export WANDB_API_KEY=your_api_key_here' >> ~/.bashrc
source ~/.bashrc
```

#### Option C: Python Login

```python
import wandb
wandb.login(key="your_api_key_here")
```

> **Note**: Avoid hardcoding API keys in scripts. Use environment variables instead.

## Using WandB with SFTTrainer

The `SFTTrainer` (and Hugging Face `Trainer`) automatically integrates with WandB when it's installed and configured.

### Basic Usage

```python
from trl import SFTTrainer, SFTConfig

sft_config = SFTConfig(
    output_dir="./output",
    num_train_epochs=1,
    logging_steps=10,
    report_to="wandb",  # Enable WandB logging
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()
```

### Setting Project and Run Names

```python
import os

# Set WandB project name
os.environ["WANDB_PROJECT"] = "my-sft-experiments"

# Optional: Set run name
os.environ["WANDB_RUN_NAME"] = "qwen3-sft-v1"

# Optional: Add tags
os.environ["WANDB_TAGS"] = "sft,qwen3,experiment"
```

Or configure in `SFTConfig`:

```python
sft_config = SFTConfig(
    output_dir="./output",
    run_name="qwen3-sft-v1",  # WandB run name
    report_to="wandb",
)
```

## Environment Variables Reference

| Variable | Description | Example |
|----------|-------------|---------|
| `WANDB_API_KEY` | Your API key | `abc123...` |
| `WANDB_PROJECT` | Project name | `my-sft-project` |
| `WANDB_RUN_NAME` | Run name | `experiment-v1` |
| `WANDB_ENTITY` | Team/username | `my-team` |
| `WANDB_TAGS` | Comma-separated tags | `sft,production` |
| `WANDB_NOTES` | Run description | `Testing new LR` |
| `WANDB_MODE` | `online`, `offline`, `disabled` | `online` |
| `WANDB_DIR` | Local log directory | `/tmp/wandb` |

## Disabling WandB

If you want to run without WandB logging:

```python
# Option 1: In config
sft_config = SFTConfig(
    report_to="none",  # Disable all reporting
)

# Option 2: Environment variable
os.environ["WANDB_MODE"] = "disabled"

# Option 3: Offline mode (logs locally, sync later)
os.environ["WANDB_MODE"] = "offline"
```

## Viewing Your Experiments

1. Go to [wandb.ai/home](https://wandb.ai/home)
2. Select your project
3. View runs, compare metrics, and analyze results

### Dashboard Features

- **Loss curves**: Visualize training/validation loss over time
- **System metrics**: GPU utilization, memory usage
- **Hyperparameter comparison**: Compare runs with different configs
- **Artifact tracking**: Model checkpoints, datasets

## Troubleshooting

### "wandb: ERROR api_key not configured"

```bash
# Solution: Set your API key
export WANDB_API_KEY=your_key_here
# Or run:
wandb login
```

### "wandb: Network error"

```bash
# Run in offline mode
export WANDB_MODE=offline

# Later, sync your runs
wandb sync ./wandb/offline-run-*
```

### "Permission denied" errors

```bash
# Set a writable directory
export WANDB_DIR=/tmp/wandb
```

### Disable WandB prompts in notebooks

```python
import os
os.environ["WANDB_SILENT"] = "true"
```

## Best Practices

1. **Use environment variables** for API keys, never hardcode them
2. **Name your runs** descriptively for easy identification
3. **Use tags** to categorize experiments (e.g., `baseline`, `ablation`, `production`)
4. **Log hyperparameters** to reproduce experiments
5. **Set up alerts** for long-running experiments

## Example: Complete SFT Training with WandB

```python
import os
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

# Configure WandB
os.environ["WANDB_PROJECT"] = "llm-sft-experiments"
os.environ["WANDB_RUN_NAME"] = "smollm-sft-v1"

# Load model and tokenizer
model_name = "HuggingFaceTB/SmolLM2-135M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load dataset
dataset = load_dataset("banghua/DL-SFT-Dataset", split="train")

# Configure training
sft_config = SFTConfig(
    output_dir="./sft_output",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    logging_steps=10,
    report_to="wandb",
)

# Train
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()

# Finish WandB run
wandb.finish()
```

## References

- [WandB Quickstart](https://docs.wandb.ai/quickstart/)
- [WandB + Hugging Face Integration](https://docs.wandb.ai/guides/integrations/huggingface)
- [Find Your API Key](https://wandb.ai/authorize)
- [WandB Environment Variables](https://docs.wandb.ai/guides/track/environment-variables)
