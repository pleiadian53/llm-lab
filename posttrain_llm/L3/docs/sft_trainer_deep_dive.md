# SFTTrainer Deep Dive

This document explains how the `SFTTrainer` from TRL (Transformer Reinforcement Learning) works behind the scenes.

## Overview

The `SFTTrainer` is a high-level abstraction built on top of Hugging Face's `Trainer` class, specifically designed for **Supervised Fine-Tuning (SFT)** of language models. It simplifies the process of adapting a pre-trained model to follow instructions or generate specific outputs.

```python
sft_trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)
sft_trainer.train()
```

## What Happens Behind the Scenes

### 1. Model Initialization

When you pass a model (or model name string), `SFTTrainer`:

- Loads the model from Hugging Face Hub if a string is provided
- Configures the model for training (enables gradient computation)
- Optionally wraps the model with PEFT adapters (LoRA, QLoRA) if `peft_config` is provided
- Sets up gradient checkpointing if enabled in config

### 2. Dataset Preprocessing

The trainer automatically handles different dataset formats:

| Format | Example |
|--------|---------|
| **Standard text** | `{"text": "The sky is blue."}` |
| **Conversational** | `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}` |
| **Prompt-completion** | `{"prompt": "The sky is", "completion": " blue."}` |

For conversational data, the trainer:
1. Applies the tokenizer's **chat template** to format messages
2. Concatenates all turns into a single sequence
3. Creates attention masks and labels

### 3. Tokenization

The `processing_class` (tokenizer) handles:

- Converting text to token IDs
- Adding special tokens (BOS, EOS, pad tokens)
- Truncating sequences to `max_seq_length`
- Creating attention masks

### 4. Data Collation

The `DataCollatorForLanguageModeling` (used internally) batches examples:

```python
# Pseudocode of what happens
def collate(examples):
    input_ids = pad([ex["input_ids"] for ex in examples])
    attention_mask = pad([ex["attention_mask"] for ex in examples])
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100  # Ignore padding in loss
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
```

Key behaviors:
- **Padding**: Sequences are padded to the longest in the batch
- **Label masking**: Padding tokens get label `-100` (ignored by CrossEntropyLoss)
- **Completion-only loss**: Optionally masks prompt tokens so loss is only computed on completions

### 5. Training Loop

The core training loop (inherited from `Trainer`):

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        outputs = model(**batch)  # Returns loss, logits
        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Gradient accumulation
        if step % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Logging (to console, WandB, TensorBoard)
        if step % logging_steps == 0:
            log_metrics(loss, learning_rate, ...)
```

### 6. Loss Computation

SFT uses **token-level cross-entropy loss**:

$$\mathcal{L}_{\text{SFT}}(\theta) = - \sum_{t=1}^{T} \log p_\theta(y_t \mid y_{<t})$$

Where:
- $y_t$ is the target token at position $t$
- The model predicts the next token given all previous tokens
- Padding tokens (label = -100) are excluded from the loss

**Label shifting**: The model predicts token $t+1$ from position $t$. Labels are automatically shifted by the model's forward pass.

### 7. Logging and Callbacks

`SFTTrainer` integrates with:

- **Weights & Biases (WandB)**: Automatic experiment tracking
- **TensorBoard**: Loss curves and metrics
- **Console**: Progress bars and step-wise metrics

Logged metrics include:
- `train/loss`: Training loss
- `train/learning_rate`: Current learning rate
- `train/epoch`: Current epoch
- `train/global_step`: Total steps completed

## Key Configuration Options (SFTConfig)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `learning_rate` | Optimizer learning rate | `5e-5` |
| `num_train_epochs` | Number of training epochs | `3` |
| `per_device_train_batch_size` | Batch size per GPU | `8` |
| `gradient_accumulation_steps` | Steps before optimizer update | `1` |
| `gradient_checkpointing` | Trade compute for memory | `False` |
| `max_seq_length` | Maximum sequence length | `1024` |
| `packing` | Pack multiple samples per sequence | `False` |
| `logging_steps` | Log every N steps | `500` |

## Advanced Features

### Packing

When `packing=True`, multiple short examples are concatenated into a single sequence to maximize GPU utilization:

```
[Example1][Example2][Example3] -> Single sequence of max_seq_length
```

### PEFT Integration

For parameter-efficient fine-tuning:

```python
from peft import LoraConfig

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
)

trainer = SFTTrainer(
    model=model,
    peft_config=peft_config,  # Automatically wraps model
    ...
)
```

### Completion-Only Loss

To train only on assistant responses (not user prompts):

```python
sft_config = SFTConfig(
    dataset_text_field="text",
    # The trainer will mask prompt tokens in loss computation
)
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         SFTTrainer                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    │
│  │   Dataset   │───▶│  Tokenizer   │───▶│  DataCollator   │    │
│  │  (messages) │    │(chat template)│    │   (batching)    │    │
│  └─────────────┘    └──────────────┘    └────────┬────────┘    │
│                                                   │             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Training Loop                         │   │
│  │  ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌────────┐ │   │
│  │  │ Forward │──▶│   Loss   │──▶│ Backward │──▶│Optimize│ │   │
│  │  │  Pass   │   │(CrossEnt)│   │   Pass   │   │  Step  │ │   │
│  │  └─────────┘   └──────────┘   └──────────┘   └────────┘ │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌───────────────────────────▼───────────────────────────────┐ │
│  │                      Logging                               │ │
│  │   WandB  │  TensorBoard  │  Console  │  Checkpoints       │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## References

- [TRL SFT Trainer Documentation](https://huggingface.co/docs/trl/en/sft_trainer)
- [TRL GitHub - sft_trainer.py](https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py)
- [Hugging Face Trainer Documentation](https://huggingface.co/docs/transformers/main_classes/trainer)
