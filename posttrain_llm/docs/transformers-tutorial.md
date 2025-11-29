# HuggingFace Transformers Tutorial

A comprehensive guide to using the `transformers` library for loading and using pretrained language models, as implemented in `serve_llm.py`.

## Table of Contents

1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [Model Loading](#model-loading)
4. [Tokenization](#tokenization)
5. [Text Generation](#text-generation)
6. [Quantization](#quantization)
7. [Device Management](#device-management)
8. [Best Practices](#best-practices)
9. [Common Patterns](#common-patterns)
10. [Troubleshooting](#troubleshooting)

---

## Introduction

The HuggingFace `transformers` library provides a unified API for working with thousands of pretrained models. This tutorial focuses on the key components used in our `ServeLLM` class.

### Installation

```bash
pip install transformers
pip install accelerate  # For device_map="auto"
pip install bitsandbytes  # For quantization
```

### Key Classes

- **`AutoModelForCausalLM`**: Automatically loads the correct model architecture for causal language modeling
- **`AutoTokenizer`**: Automatically loads the correct tokenizer for a model
- **`BitsAndBytesConfig`**: Configuration for model quantization

---

## Core Concepts

### 1. Model Hub

HuggingFace hosts thousands of models on their Hub. Models are identified by:

```python
# Organization/model-name format
model_name = "deepseek-ai/deepseek-math-7b-base"
model_name = "meta-llama/Llama-Guard-3-8B"

# Or local path
model_name = "/path/to/local/model"
```

### 2. Model Architecture

The `Auto*` classes automatically detect and load the correct architecture:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# These will load the correct architecture automatically
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

### 3. Trust Remote Code

Some models require custom code. Use `trust_remote_code=True`:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True  # Required for some models
)
```

---

## Model Loading

### Basic Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Load model
model = AutoModelForCausalLM.from_pretrained("gpt2")
```

### Advanced Loading Options

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,      # Allow custom model code
    low_cpu_mem_usage=True,      # Reduce CPU memory during loading
    torch_dtype=torch.float16,   # Use half precision
    device_map="auto",           # Automatically distribute across devices
)
```

### Loading from Local Path

```python
# If you've downloaded a model locally
model = AutoModelForCausalLM.from_pretrained(
    "/path/to/local/model",
    local_files_only=True  # Don't try to download
)
```

### Caching

Models are cached by default in `~/.cache/huggingface/hub/`. Control this with:

```python
import os

# Set custom cache directory
os.environ['HF_HOME'] = '/path/to/cache'
os.environ['TRANSFORMERS_CACHE'] = '/path/to/cache'

# Or specify in from_pretrained
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir="/path/to/cache"
)
```

---

## Tokenization

### Basic Tokenization

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Tokenize text
text = "Hello, how are you?"
tokens = tokenizer(text)

print(tokens)
# {'input_ids': [15496, 11, 703, 389, 345, 30], 'attention_mask': [1, 1, 1, 1, 1, 1]}
```

### Return Tensors

```python
# Return PyTorch tensors
tokens = tokenizer(text, return_tensors="pt")

# Return NumPy arrays
tokens = tokenizer(text, return_tensors="np")

# Return TensorFlow tensors
tokens = tokenizer(text, return_tensors="tf")
```

### Padding and Truncation

```python
# Pad to max length in batch
tokens = tokenizer(
    ["Short text", "This is a much longer text"],
    padding=True,           # Pad to longest in batch
    truncation=True,        # Truncate to max_length
    max_length=512,         # Maximum sequence length
    return_tensors="pt"
)
```

### Padding Token

Some models don't have a padding token. Add one:

```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    # Or use a different token
    # tokenizer.pad_token = tokenizer.unk_token
```

### Decoding

```python
# Encode text
input_ids = tokenizer.encode("Hello world", return_tensors="pt")

# Decode back to text
text = tokenizer.decode(input_ids[0])
print(text)  # "Hello world"

# Skip special tokens (like <s>, </s>, <pad>)
text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
```

### Batch Encoding/Decoding

```python
# Encode multiple texts
texts = ["First text", "Second text", "Third text"]
tokens = tokenizer(texts, padding=True, return_tensors="pt")

# Decode multiple sequences
decoded = tokenizer.batch_decode(tokens['input_ids'], skip_special_tokens=True)
```

---

## Text Generation

### Basic Generation

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Prepare input
prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)

# Decode
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

### Generation Parameters

```python
outputs = model.generate(
    **inputs,
    
    # Length control
    max_new_tokens=100,        # Maximum tokens to generate
    min_new_tokens=10,         # Minimum tokens to generate
    
    # Sampling parameters
    do_sample=True,            # Use sampling (vs greedy)
    temperature=0.7,           # Sampling temperature (0.0 = greedy)
    top_k=50,                  # Top-k sampling
    top_p=0.95,                # Nucleus sampling (top-p)
    
    # Special tokens
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    
    # Other
    num_return_sequences=1,    # Number of sequences to generate
    repetition_penalty=1.0,    # Penalize repetition
)
```

### Temperature

Controls randomness in generation:

```python
# Greedy decoding (deterministic)
outputs = model.generate(**inputs, temperature=0.0, do_sample=False)

# Low temperature (more focused, less random)
outputs = model.generate(**inputs, temperature=0.3, do_sample=True)

# Medium temperature (balanced)
outputs = model.generate(**inputs, temperature=0.7, do_sample=True)

# High temperature (more random, creative)
outputs = model.generate(**inputs, temperature=1.5, do_sample=True)
```

### Top-k and Top-p Sampling

```python
# Top-k: Sample from top k most likely tokens
outputs = model.generate(
    **inputs,
    do_sample=True,
    top_k=50,           # Consider only top 50 tokens
    temperature=0.7
)

# Top-p (nucleus): Sample from smallest set with cumulative prob >= p
outputs = model.generate(
    **inputs,
    do_sample=True,
    top_p=0.95,         # Consider tokens until 95% probability mass
    temperature=0.7
)

# Combine both
outputs = model.generate(
    **inputs,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.7
)
```

### Extracting Only Generated Text

```python
# Get input length
input_length = inputs['input_ids'].shape[1]

# Generate
outputs = model.generate(**inputs, max_new_tokens=50)

# Extract only the generated part (exclude prompt)
generated_ids = outputs[0][input_length:]
generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
```

---

## Quantization

Quantization reduces model size and memory usage by using lower precision.

### 4-bit Quantization (NF4)

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # Enable 4-bit loading
    bnb_4bit_compute_dtype=torch.float16,   # Compute dtype
    bnb_4bit_use_double_quant=True,         # Nested quantization
    bnb_4bit_quant_type="nf4"               # NormalFloat4 quantization
)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
```

### 8-bit Quantization

```python
# Configure 8-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
```

### Memory Savings

| Precision | Memory (7B model) | Speed | Quality |
|-----------|-------------------|-------|---------|
| FP32      | ~28 GB           | 1.0x  | 100%    |
| FP16      | ~14 GB           | 2.0x  | 99.9%   |
| 8-bit     | ~7 GB            | 1.5x  | 99%     |
| 4-bit     | ~3.5 GB          | 1.3x  | 95-98%  |

### When to Use Quantization

- **4-bit**: When memory is very limited (e.g., Colab free tier, consumer GPUs)
- **8-bit**: Good balance of memory and quality
- **FP16**: When you have enough memory and want best quality
- **FP32**: Only for training or when precision is critical

---

## Device Management

### Manual Device Placement

```python
import torch

# Check available devices
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")

# Load model and move to device
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move to CUDA
model = model.to("cuda")

# Move to MPS (Apple Silicon)
model = model.to("mps")

# Move to CPU
model = model.to("cpu")
```

### Automatic Device Mapping

```python
# Let transformers handle device placement
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"  # Automatically distribute across available devices
)

# This works with:
# - Single GPU
# - Multiple GPUs
# - CPU offloading
# - Disk offloading (for very large models)
```

### Moving Inputs to Device

```python
# Tokenize
inputs = tokenizer(text, return_tensors="pt")

# Move to same device as model
inputs = inputs.to(model.device)

# Or specify device explicitly
inputs = inputs.to("cuda")
inputs = {k: v.to("cuda") for k, v in inputs.items()}
```

### Memory Management

```python
import gc
import torch

# Delete model
del model
del tokenizer

# Collect garbage
gc.collect()

# Clear CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Clear MPS cache (Apple Silicon)
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    torch.mps.empty_cache()
```

---

## Best Practices

### 1. Use Context Managers

```python
class ServeLLM:
    def __enter__(self):
        self._initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

# Usage
with ServeLLM(model_name) as llm:
    response = llm.generate_response(prompt)
# Automatic cleanup happens here
```

### 2. Set Evaluation Mode

```python
# Always set model to eval mode for inference
model.eval()

# Use torch.no_grad() to save memory
with torch.no_grad():
    outputs = model.generate(**inputs)
```

### 3. Batch Processing

```python
# Process multiple prompts at once (more efficient)
prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
inputs = tokenizer(prompts, padding=True, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)

# Decode all at once
responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
```

### 4. Handle Long Sequences

```python
# Truncate long inputs
inputs = tokenizer(
    text,
    max_length=2048,
    truncation=True,
    return_tensors="pt"
)

# Or use sliding window for very long texts
def process_long_text(text, window_size=2048, stride=1024):
    tokens = tokenizer.encode(text)
    chunks = []
    
    for i in range(0, len(tokens), stride):
        chunk = tokens[i:i + window_size]
        chunks.append(chunk)
    
    return chunks
```

### 5. Error Handling

```python
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True
    )
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback or retry logic
```

---

## Common Patterns

### Pattern 1: Simple Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def simple_generate(model_name, prompt, max_tokens=100):
    """Simple text generation."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Pattern 2: Batch Processing

```python
def batch_generate(model_name, prompts, max_tokens=100):
    """Generate responses for multiple prompts."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    
    # Ensure padding token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize all prompts
    inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode all
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)
```

### Pattern 3: Streaming Generation

```python
from transformers import TextIteratorStreamer
from threading import Thread

def stream_generate(model_name, prompt):
    """Stream generated tokens as they're produced."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Create streamer
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    
    # Generate in separate thread
    generation_kwargs = dict(inputs, max_new_tokens=100, streamer=streamer)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Stream tokens
    for text in streamer:
        print(text, end="", flush=True)
    
    thread.join()
```

### Pattern 4: Multi-Device Setup

```python
def load_with_best_device(model_name):
    """Load model on best available device."""
    import torch
    
    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
        device_map = "auto"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        device_map = None
    else:
        device = "cpu"
        device_map = None
    
    print(f"Using device: {device}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32
    )
    
    # Move to device if needed
    if device_map is None:
        model = model.to(device)
    
    return model, device
```

### Pattern 5: Memory-Efficient Loading

```python
def load_memory_efficient(model_name, quantize="8bit"):
    """Load model with quantization for memory efficiency."""
    from transformers import BitsAndBytesConfig
    
    if quantize == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif quantize == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        bnb_config = None
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    return model
```

---

## Troubleshooting

### Issue 1: Out of Memory (OOM)

**Symptoms**: `RuntimeError: CUDA out of memory` or similar

**Solutions**:

```python
# 1. Use quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=BitsAndBytesConfig(load_in_8bit=True)
)

# 2. Reduce batch size
inputs = tokenizer(prompts[:1], return_tensors="pt")  # Process one at a time

# 3. Reduce max_length
inputs = tokenizer(text, max_length=512, truncation=True, return_tensors="pt")

# 4. Use gradient checkpointing (for training)
model.gradient_checkpointing_enable()

# 5. Clear cache between runs
torch.cuda.empty_cache()
```

### Issue 2: Slow Generation

**Symptoms**: Generation takes too long

**Solutions**:

```python
# 1. Use smaller max_new_tokens
outputs = model.generate(**inputs, max_new_tokens=50)  # Instead of 500

# 2. Use greedy decoding (faster than sampling)
outputs = model.generate(**inputs, do_sample=False)

# 3. Use GPU instead of CPU
model = model.to("cuda")

# 4. Use batch processing
# Process multiple prompts at once instead of one by one

# 5. Use quantization (slightly faster)
```

### Issue 3: Model Not Found

**Symptoms**: `OSError: model_name not found`

**Solutions**:

```python
# 1. Check model name spelling
model_name = "gpt2"  # Correct
# model_name = "gpt-2"  # Wrong

# 2. Check internet connection (for downloading)

# 3. Use local_files_only if model is cached
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    local_files_only=True
)

# 4. Specify cache directory
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir="/path/to/cache"
)
```

### Issue 4: Padding Token Missing

**Symptoms**: `ValueError: Padding token not found`

**Solutions**:

```python
# Add padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Or use a different token
if tokenizer.pad_token is None:
    tokenizer.pad_token = "[PAD]"
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
```

### Issue 5: Device Mismatch

**Symptoms**: `RuntimeError: Expected all tensors to be on the same device`

**Solutions**:

```python
# Ensure inputs are on same device as model
inputs = inputs.to(model.device)

# Or move both to same device
device = "cuda"
model = model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}
```

---

## Implementation in ServeLLM

Our `ServeLLM` class implements these best practices:

```python
class ServeLLM:
    def __init__(self, model_name, seed=42, device="auto", dtype=torch.float16, quantize=None):
        # Auto-detect best device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
    
    def load_model(self):
        # Load with appropriate configuration
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        
        # Handle padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Configure quantization if requested
        if self.quantize:
            bnb_config = BitsAndBytesConfig(...)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype
            )
        
        return model, tokenizer
    
    def generate_response(self, prompts, temperature=0.0, top_p=1.0, max_tokens=100):
        # Tokenize
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        
        # Generate with torch.no_grad()
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        
        # Decode
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def cleanup(self):
        # Proper cleanup
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()
```

---

## Additional Resources

### Official Documentation

- [HuggingFace Transformers Docs](https://huggingface.co/docs/transformers)
- [Model Hub](https://huggingface.co/models)
- [Generation Guide](https://huggingface.co/docs/transformers/main_classes/text_generation)

### Related Files

- `serve_llm.py` - Implementation of ServeLLM class
- `DEVICE_USAGE_GUIDE.md` - Device-specific usage guide
- `COLAB_SETUP.md` - Colab environment setup

### Key Takeaways

1. ✅ Use `Auto*` classes for automatic model/tokenizer loading
2. ✅ Always set `model.eval()` for inference
3. ✅ Use `torch.no_grad()` to save memory
4. ✅ Handle padding tokens properly
5. ✅ Use quantization for memory-constrained environments
6. ✅ Batch process when possible for efficiency
7. ✅ Clean up resources properly
8. ✅ Use context managers for automatic cleanup

---

**Last Updated**: November 2025  
**Maintained by**: llm-lab project
