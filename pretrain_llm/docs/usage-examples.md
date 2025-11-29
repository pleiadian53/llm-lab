# Usage Examples

Quick examples for common tasks using the refactored modules.

## Example 1: Download and Load a Model

```python
from pathlib import Path
from llm_lab.pretrain_llm import load_model_with_fallback

# Download and cache model locally
model, tokenizer = load_model_with_fallback(
    model_name="upstage/TinySolar-248m-4k",
    local_dir=Path("./models/TinySolar-248m-4k"),
    device_map="cpu",
)

# Generate text
prompt = "I am an engineer. I love"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Example 2: Using ModelLoader Class

```python
import torch
from pathlib import Path
from llm_lab.pretrain_llm import ModelLoader

# Initialize loader
loader = ModelLoader(
    model_name_or_path="upstage/TinySolar-248m-4k",
    local_cache_dir=Path("./models/TinySolar-248m-4k"),
    device_map="cpu",
    torch_dtype=torch.bfloat16,
)

# Load model and tokenizer
model, tokenizer = loader.load_model_and_tokenizer()
```

## Example 3: Pre-download Models

```python
from pathlib import Path
from llm_lab.pretrain_llm import download_model_manually

# Download multiple models
models = [
    "upstage/TinySolar-248m-4k",
    "upstage/TinySolar-248m-4k-code-instruct",
    "upstage/TinySolar-248m-4k-py",
]

for model_name in models:
    output_dir = Path(f"./models/{model_name.split('/')[-1]}")
    download_model_manually(model_name, output_dir)
```

## Example 4: Notebook-Friendly Pattern

```python
import torch
from pathlib import Path
from llm_lab.pretrain_llm import load_model_with_fallback

def load_model_smart(model_name: str, local_dir: Path):
    """Load model with smart caching."""
    model, tokenizer = load_model_with_fallback(
        model_name=model_name,
        local_dir=local_dir if local_dir.exists() else None,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
    )
    
    source = "local cache" if local_dir.exists() else "HuggingFace Hub"
    print(f"✓ Model loaded from {source}")
    return model, tokenizer

# Use in notebook
model, tokenizer = load_model_smart(
    "upstage/TinySolar-248m-4k",
    Path("./models/TinySolar-248m-4k")
)
```

## Example 5: Compare Multiple Models

```python
from pathlib import Path
from llm_lab.pretrain_llm import load_model_with_fallback

models_to_compare = {
    "general": "upstage/TinySolar-248m-4k",
    "code": "upstage/TinySolar-248m-4k-code-instruct",
    "python": "upstage/TinySolar-248m-4k-py",
}

prompt = "def find_max(numbers):"

for name, model_id in models_to_compare.items():
    print(f"\n{'='*60}")
    print(f"Model: {name}")
    print(f"{'='*60}")
    
    model, tokenizer = load_model_with_fallback(
        model_id,
        local_dir=Path(f"./models/{model_id.split('/')[-1]}")
    )
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(generated)
```

## Example 6: Using with Training Pipeline

```python
from pathlib import Path
from llm_lab.pretrain_llm import (
    PretrainingConfig,
    LanguageModelTrainer,
    load_model_with_fallback,
)

# Load model
model, tokenizer = load_model_with_fallback(
    "gpt2",
    local_dir=Path("./models/gpt2"),
)

# Configure training
config = PretrainingConfig(
    seed=42,
    output_dir=Path("./artifacts/my_experiment"),
)

# Update config with loaded model
config.model.pretrained_model_name = "gpt2"

# Create trainer
trainer = LanguageModelTrainer.from_config(config)

# Run training
trainer.run()
```

## Example 7: Offline Mode

```python
import os
from pathlib import Path
from llm_lab.pretrain_llm import load_model_with_fallback

# Enable offline mode
os.environ['HF_HUB_OFFLINE'] = '1'

# This will only work if model is already cached locally
model, tokenizer = load_model_with_fallback(
    "upstage/TinySolar-248m-4k",
    local_dir=Path("./models/TinySolar-248m-4k"),
)
```

## Example 8: Custom Device Mapping

```python
import torch
from pathlib import Path
from llm_lab.pretrain_llm import ModelLoader

# For GPU with specific device
loader = ModelLoader(
    model_name_or_path="upstage/TinySolar-248m-4k",
    local_cache_dir=Path("./models/TinySolar-248m-4k"),
    device_map="cuda:0",  # Specific GPU
    torch_dtype=torch.float16,
)

model, tokenizer = loader.load_model_and_tokenizer()
```

## Example 9: Batch Processing

```python
from pathlib import Path
from llm_lab.pretrain_llm import load_model_with_fallback

model, tokenizer = load_model_with_fallback(
    "upstage/TinySolar-248m-4k",
    local_dir=Path("./models/TinySolar-248m-4k"),
)

prompts = [
    "I am an engineer. I love",
    "The future of AI is",
    "Python is a programming language that",
]

# Batch tokenization
inputs = tokenizer(prompts, return_tensors="pt", padding=True)

# Generate
outputs = model.generate(**inputs, max_new_tokens=30)

# Decode
for i, output in enumerate(outputs):
    print(f"\nPrompt {i+1}: {prompts[i]}")
    print(f"Generated: {tokenizer.decode(output, skip_special_tokens=True)}")
```

## Example 10: Error Handling

```python
from pathlib import Path
from llm_lab.pretrain_llm import load_model_with_fallback
import logging

logging.basicConfig(level=logging.INFO)

def safe_load_model(model_name: str, local_dir: Path):
    """Load model with error handling."""
    try:
        model, tokenizer = load_model_with_fallback(
            model_name=model_name,
            local_dir=local_dir,
        )
        return model, tokenizer
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        # Fallback to smaller model
        logging.info("Falling back to gpt2...")
        return load_model_with_fallback("gpt2")

# Use it
model, tokenizer = safe_load_model(
    "upstage/TinySolar-248m-4k",
    Path("./models/TinySolar-248m-4k")
)
```
