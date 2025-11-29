# Changelog - M1 Module

## Overview

The M1 module provides tools for inspecting and evaluating fine-tuned language models. This notebook demonstrates how to compare base models, supervised fine-tuned (SFT) models, and reinforcement learning (RL) models.

## Features

### Multi-Device Support

The `ServeLLM` utility now works on multiple hardware platforms:

- **NVIDIA GPUs** (CUDA) - Best performance
- **Apple Silicon** (M1/M2/M3 with MPS) - Good performance on Mac
- **CPU** - Fallback for any system

**Auto-detection**: The system automatically selects the best available device.

### Memory Optimization

Quantization support for running large models on consumer hardware:

- **8-bit quantization** - Reduces memory usage by ~50%
- **4-bit quantization** - Reduces memory usage by ~75%

This enables running 7B parameter models on systems with 16GB RAM.

## Quick Start

### Basic Usage

```python
from utils.utils import ServeLLM

# Automatic device selection and model loading
with ServeLLM("deepseek-ai/deepseek-math-7b-base") as llm:
    response = llm.generate_response("What is 2+2?")
    print(response)
```

### With Quantization (Recommended for Limited Memory)

```python
# Use 8-bit quantization to reduce memory usage
with ServeLLM(model_name, quantize="8bit") as llm:
    response = llm.generate_response(prompt)
```

### Specify Device Explicitly

```python
# Force specific device
with ServeLLM(model_name, device="mps") as llm:  # For Apple Silicon
    response = llm.generate_response(prompt)
```

## System Requirements

### Minimum Requirements

- Python 3.8+
- 16GB RAM (with quantization)
- PyTorch 1.13+
- Transformers 4.30+

### Recommended for Best Performance

- NVIDIA GPU with 16GB+ VRAM
- 32GB+ RAM
- PyTorch 2.0+

### For Apple Silicon (M1/M2/M3)

- macOS 12.3+
- PyTorch 2.0+ (for MPS support)
- 16GB+ unified memory

## Documentation

- **[README_DEVICE_SUPPORT.md](README_DEVICE_SUPPORT.md)** - Quick start guide
- **[DEVICE_USAGE_GUIDE.md](DEVICE_USAGE_GUIDE.md)** - Detailed usage examples
- **[Transformers Tutorial](../docs/transformers-tutorial.md)** - Complete guide to the Transformers library

## Notebook Contents

The `M1_G1_Inspecting_Finetuned_vs_Base_Model.ipynb` notebook includes:

1. **Model Comparison** - Compare base, SFT, and RL models
2. **Prompt Processing** - Batch process prompts with different models
3. **Evaluation** - Evaluate model correctness on GSM8K dataset
4. **Safety Testing** - Test model safety with Llama Guard

## Performance

Approximate inference times for 7B models (single prompt):

| Hardware | Configuration | Time per Prompt |
|----------|--------------|-----------------|
| A100 GPU | FP16 | ~1s |
| T4 GPU (Colab) | FP16 | ~3s |
| M1 Pro | MPS + 4-bit | ~5s |
| M1 Pro | CPU + 4-bit | ~30s |

## Troubleshooting

### Out of Memory Errors

**Solution**: Use quantization
```python
with ServeLLM(model_name, quantize="8bit") as llm:
    ...
```

### Slow Performance

**Solutions**:
- Ensure you're using GPU/MPS (check with `device="auto"`)
- Reduce `max_tokens` in generation
- Use smaller batch sizes

### Model Not Found

**Solution**: Models download automatically from HuggingFace. Ensure you have:
- Internet connection
- Sufficient disk space (~14GB per 7B model)

## Support

For issues or questions:
- Check the documentation files listed above
- Review the notebook examples
- Consult the [Transformers Tutorial](../docs/transformers-tutorial.md)

---

**Last Updated**: November 2025  
**Module**: Post-Training LLM (M1)
