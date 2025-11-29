# Device & Quantization Usage Guide

## Overview

The updated `ServeLLM` class now supports multiple backends and quantization options while maintaining full backward compatibility.

## Supported Devices

| Device | When to Use | Performance |
|--------|-------------|-------------|
| **CUDA** | NVIDIA GPUs | Fastest ⚡⚡⚡ |
| **MPS** | Apple Silicon (M1/M2/M3) | Fast ⚡⚡ |
| **CPU** | No GPU available | Slow ⚡ |

## Usage Examples

### 1. Auto-Detection (Recommended)

```python
# Automatically selects best available: CUDA > MPS > CPU
with ServeLLM(model_name) as llm:
    response = llm.generate_response("What is 2+2?")
```

### 2. Explicit Device Selection

```python
# Force specific device
with ServeLLM(model_name, device="mps") as llm:  # For M1 Mac
    response = llm.generate_response("What is 2+2?")

with ServeLLM(model_name, device="cuda") as llm:  # For NVIDIA GPU
    response = llm.generate_response("What is 2+2?")

with ServeLLM(model_name, device="cpu") as llm:  # CPU only
    response = llm.generate_response("What is 2+2?")
```

### 3. Quantization (Memory Saving)

```python
# 4-bit quantization - Best for 16GB RAM
with ServeLLM(model_name, quantize="4bit") as llm:
    response = llm.generate_response("What is 2+2?")

# 8-bit quantization - Balance between speed and memory
with ServeLLM(model_name, quantize="8bit") as llm:
    response = llm.generate_response("What is 2+2?")
```

### 4. Combined Options

```python
# MPS + 4-bit quantization for M1 Mac with limited RAM
with ServeLLM(model_name, device="mps", quantize="4bit") as llm:
    response = llm.generate_response("What is 2+2?")
```

## Memory Requirements

### 7B Model (DeepSeek Math)

| Configuration | Memory Usage | M1 16GB? |
|---------------|--------------|----------|
| FP16 (default) | ~14GB | ⚠️ Tight |
| 8-bit | ~7GB | ✅ Good |
| 4-bit | ~4GB | ✅ Excellent |

### Recommendations by Device

#### M1 MacBook Pro 16GB
```python
# Recommended: Use 4-bit quantization
with ServeLLM(model_name, device="mps", quantize="4bit") as llm:
    response = llm.generate_response(prompt, max_tokens=512)
```

#### Google Colab (Free T4 GPU)
```python
# Use default settings - auto-detects CUDA
with ServeLLM(model_name) as llm:
    response = llm.generate_response(prompt, max_tokens=512)
```

#### Colab with Limited RAM
```python
# Use 8-bit quantization
with ServeLLM(model_name, quantize="8bit") as llm:
    response = llm.generate_response(prompt, max_tokens=512)
```

## Installation Requirements

### For Quantization (Optional)
```bash
# Install bitsandbytes for quantization support
pip install bitsandbytes

# For M1 Mac, you might need:
pip install bitsandbytes-darwin
```

### For MPS (Apple Silicon)
```bash
# Ensure you have PyTorch with MPS support
pip install torch>=2.0.0
```

## Backward Compatibility

All existing code continues to work:

```python
# Old code (still works!)
with ServeLLM(model_name, device="cuda") as llm:
    response = llm.generate_response(prompt)

# New default behavior (auto-detection)
with ServeLLM(model_name) as llm:  # device="auto" is default
    response = llm.generate_response(prompt)
```

## Troubleshooting

### MPS Not Available
```
Warning: MPS requested but not available. Falling back to CPU.
```
**Solution**: Update PyTorch to version 2.0+ with MPS support

### Quantization Import Error
```
Warning: bitsandbytes not installed. Install with: pip install bitsandbytes
Falling back to non-quantized loading
```
**Solution**: Install bitsandbytes: `pip install bitsandbytes`

### Out of Memory on M1
**Solution**: Use 4-bit quantization:
```python
with ServeLLM(model_name, quantize="4bit") as llm:
    response = llm.generate_response(prompt)
```

## Performance Benchmarks (Approximate)

### 7B Model Inference Time (per prompt)

| Device | Config | Time |
|--------|--------|------|
| CUDA (A100) | FP16 | ~1s |
| CUDA (T4) | FP16 | ~3s |
| MPS (M1) | FP16 | ~8s |
| MPS (M1) | 4-bit | ~5s |
| CPU (M1) | FP16 | ~45s |
| CPU (M1) | 4-bit | ~30s |

*Times are approximate and vary by model and prompt length*
