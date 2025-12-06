# SFT Trainer

A reusable package for **Supervised Fine-Tuning (SFT)** of language models with comprehensive support for **Parameter-Efficient Fine-Tuning (PEFT)** methods.

## Features

- **Easy-to-use API** for fine-tuning language models
- **Multiple PEFT methods** supported out of the box:
  - LoRA (Low-Rank Adaptation)
  - DoRA (Weight-Decomposed Low-Rank Adaptation)
  - VeRA (Vector-based Random Matrix Adaptation)
  - QLoRA (Quantized LoRA with 4-bit/8-bit)
  - AdaLoRA (Adaptive LoRA)
  - IA³ (Infused Adapter by Inhibiting and Amplifying)
  - Prompt Tuning
  - Prefix Tuning
- **Preset configurations** for quick experimentation
- **Multi-device support**: CUDA, Apple Silicon (MPS), CPU
- **HuggingFace Hub integration** for model sharing

## Installation

```bash
# From the llm-lab repository
cd posttrain_llm/sft_trainer
pip install -e .

# Or install dependencies directly
pip install torch transformers datasets peft trl accelerate bitsandbytes
```

## Quick Start

### Full Fine-Tuning

```python
from sft_trainer import SFTTrainerWrapper

trainer = SFTTrainerWrapper(
    model_name="HuggingFaceTB/SmolLM2-135M",
    dataset_name="banghua/DL-SFT-Dataset",
    use_gpu=True,
)
trainer.train()
trainer.save_model("./my_finetuned_model")
```

### LoRA Fine-Tuning

```python
from sft_trainer import SFTTrainerWrapper
from sft_trainer.peft import PEFTConfig

# Use a preset
peft_config = PEFTConfig.from_preset("lora_default")

trainer = SFTTrainerWrapper(
    model_name="HuggingFaceTB/SmolLM2-135M",
    dataset_name="banghua/DL-SFT-Dataset",
    peft_config=peft_config,
    use_gpu=True,
)
trainer.train()
```

### QLoRA (4-bit Quantized LoRA)

```python
from sft_trainer import SFTTrainerWrapper
from sft_trainer.peft import PEFTConfig

peft_config = PEFTConfig.from_preset("qlora_4bit")

trainer = SFTTrainerWrapper(
    model_name="meta-llama/Llama-2-7b-hf",
    dataset_name="your-dataset",
    peft_config=peft_config,
    use_gpu=True,
)
trainer.train()
```

### Custom PEFT Configuration

```python
from sft_trainer.peft import PEFTConfig, PEFTMethod

peft_config = PEFTConfig(
    method=PEFTMethod.LORA,
    r=32,
    lora_alpha=64,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
```

## Available PEFT Presets

| Preset | Description |
|--------|-------------|
| `lora_default` | Standard LoRA (r=16, alpha=32) |
| `lora_high_rank` | High-rank LoRA (r=64, alpha=128) |
| `dora` | DoRA configuration |
| `olora` | OLoRA with orthogonal initialization |
| `qlora_4bit` | QLoRA with 4-bit NF4 quantization |
| `qlora_8bit` | QLoRA with 8-bit quantization |
| `vera` | VeRA configuration |
| `adalora` | AdaLoRA with adaptive rank |
| `ia3` | IA³ configuration |
| `prompt_tuning` | Prompt tuning |
| `prefix_tuning` | Prefix tuning |

## Training Configuration

```python
from sft_trainer import TrainingConfig

config = TrainingConfig(
    learning_rate=8e-5,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    logging_steps=10,
    save_steps=500,
    output_dir="./output",
)

trainer = SFTTrainerWrapper(
    model_name="...",
    dataset_name="...",
    training_config=config,
)
```

## Inference

```python
from sft_trainer import load_model_and_tokenizer, generate_response, test_model

# Load model
model, tokenizer = load_model_and_tokenizer("./my_finetuned_model", use_gpu=True)

# Single response
response = generate_response(
    model, tokenizer,
    user_message="What is machine learning?",
    max_new_tokens=200,
)
print(response)

# Test with multiple questions
questions = [
    "Explain neural networks.",
    "What is gradient descent?",
]
test_model(model, tokenizer, questions, title="Model Test")
```

## Package Structure

```text
sft_trainer/
├── __init__.py          # Main package exports
├── setup.py             # Package installation
├── README.md            # This file
├── core/
│   ├── __init__.py
│   ├── model_loader.py  # Model/tokenizer loading
│   ├── trainer.py       # SFTTrainerWrapper
│   └── inference.py     # Generation utilities
├── peft/
│   ├── __init__.py
│   ├── config.py        # PEFTConfig, PEFTMethod
│   └── utils.py         # PEFT utilities
└── utils/
    ├── __init__.py
    ├── dataset.py       # Dataset utilities
    └── display.py       # Display utilities
```

## Comparison: Full Fine-Tuning vs PEFT

| Aspect | Full Fine-Tuning | LoRA | QLoRA |
|--------|-----------------|------|-------|
| Memory | High | Low | Very Low |
| Speed | Slow | Fast | Fast |
| Parameters | All | ~0.1-1% | ~0.1-1% |
| Quality | Best | Near-best | Good |
| Use Case | Small models | Medium models | Large models |

## Requirements

- Python >= 3.9
- PyTorch >= 2.0.0
- Transformers >= 4.40.0
- PEFT >= 0.10.0
- TRL >= 0.8.0
- Accelerate >= 0.27.0
- bitsandbytes >= 0.42.0 (for QLoRA)

## License

MIT License
