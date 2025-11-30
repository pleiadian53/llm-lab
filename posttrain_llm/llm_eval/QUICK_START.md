# LLM Eval - Quick Start Guide

## 🚀 Installation

```bash
cd posttrain_llm/llm_eval
pip install -e .
```

## 📖 Basic Usage

### 1. Simple Model Inference

```python
from llm_eval import ServeLLM

with ServeLLM("deepseek-ai/deepseek-math-7b-base") as llm:
    response = llm.generate_response("What is 2+2?")
    print(response)
```

### 2. Math Reasoning Evaluation

```python
from llm_eval import evaluate_math_reasoning
from datasets import load_dataset

# Load GSM8K dataset
dataset = load_dataset("gsm8k", "main", split="test")

# Evaluate model
accuracy, results = evaluate_math_reasoning(
    "deepseek-ai/deepseek-math-7b-instruct",
    dataset,
    num_samples=50
)

print(f"Accuracy: {accuracy:.2%}")
```

### 3. Safety Classification

```python
from llm_eval import evaluate_safety
from datasets import load_dataset

# Load datasets
harmful = load_dataset("jailbreakbench_harmful")
benign = [
    "What's the weather like today?",
    "How do I cook pasta?",
    "Explain quantum physics"
]

# Evaluate safety model
results = evaluate_safety(
    "meta-llama/Llama-Guard-3-8B",
    harmful,
    benign,
    num_harmful=20,
    num_benign=10
)

# Print metrics
metrics = results['metrics']
print(f"Harmful Detection Rate: {metrics['harmful_detection_rate']:.2%}")
print(f"False Positive Rate: {metrics['false_positive_rate']:.2%}")
```

### 4. Batch Processing

```python
from llm_eval import process_prompts

prompts = [
    "What is 2+2?",
    "What is 3+3?",
    "What is 4+4?"
]

results = process_prompts("model-name", prompts)
for prompt, response in zip(prompts, results):
    print(f"Q: {prompt}")
    print(f"A: {response}\n")
```

### 5. Extract Numbers from Responses

```python
from llm_eval import extract_number

# GSM8K format
answer1 = extract_number("The answer is #### 42")
print(answer1)  # 42.0

# Natural text
answer2 = extract_number("We calculate 2 + 2 = 4")
print(answer2)  # 4.0
```

### 6. Parse Safety Classifications

```python
from llm_eval import parse_llama_guard_response

# Unsafe content
result1 = parse_llama_guard_response("unsafe\nS1\nS5")
print(result1)
# {'classification': 'unsafe', 'categories': ['S1', 'S5']}

# Safe content
result2 = parse_llama_guard_response("safe")
print(result2)
# {'classification': 'safe', 'categories': []}
```

## 🎨 Display Utilities

```python
from llm_eval import (
    display_section_header,
    display_success,
    display_warning,
    display_info
)

display_section_header("Main Section", level=1)
display_success("Model loaded successfully")
display_warning("Low memory available")
display_info("Processing 100 samples...")
```

## 🔐 HuggingFace Authentication

```python
from llm_eval import validate_token

if validate_token():
    print("Ready to download models!")
else:
    print("Please set up your HuggingFace token")
```

## 📊 Complete Example

```python
from llm_eval import (
    ServeLLM,
    evaluate_math_reasoning,
    display_section_header,
    display_success
)
from datasets import load_dataset

# Display header
display_section_header("Model Evaluation Pipeline", level=1)

# Load dataset
dataset = load_dataset("gsm8k", "main", split="test").shuffle(seed=42)

# Evaluate multiple models
models = [
    "deepseek-ai/deepseek-math-7b-base",
    "deepseek-ai/deepseek-math-7b-instruct"
]

for model_name in models:
    print(f"\nEvaluating {model_name}...")
    
    accuracy, results = evaluate_math_reasoning(
        model_name,
        dataset,
        num_samples=30,
        verbose=True
    )
    
    display_success(f"{model_name}: {accuracy:.2%} accuracy")
```

## 🔧 Advanced Usage

### Custom Generation Parameters

```python
with ServeLLM("model-name") as llm:
    response = llm.generate_response(
        prompt="Your prompt",
        max_tokens=1024,
        temperature=0.8,
        top_p=0.95,
        do_sample=True
    )
```

### Device Selection

```python
# Automatic (default)
llm = ServeLLM("model-name", device="auto")

# Force CUDA
llm = ServeLLM("model-name", device="cuda")

# Force CPU
llm = ServeLLM("model-name", device="cpu")

# Apple Silicon MPS
llm = ServeLLM("model-name", device="mps")
```

### Memory Management

```python
from llm_eval import ServeLLM

# Automatic cleanup with context manager (recommended)
with ServeLLM("model-name") as llm:
    response = llm.generate_response("prompt")
# Model automatically cleaned up here

# Manual cleanup
llm = ServeLLM("model-name")
response = llm.generate_response("prompt")
llm.cleanup()  # Manually free memory

# Clean all GPU memory
ServeLLM.cleanup_all()
```

## 📚 More Information

- **Full Documentation**: See `README.md`
- **Package Summary**: See `PACKAGE_SUMMARY.md`
- **API Reference**: See docstrings in source code

## 🆘 Troubleshooting

### Import Error

```python
# Make sure you're in the right directory
import sys
sys.path.insert(0, '/path/to/posttrain_llm')

from llm_eval import ServeLLM
```

### Model Not Found

```python
# Validate your HuggingFace token first
from llm_eval import validate_token
validate_token()
```

### Out of Memory

```python
# Use smaller batch sizes or CPU
with ServeLLM("model-name", device="cpu") as llm:
    # Process in smaller batches
    pass
```

## 🎯 Next Steps

1. Install the package: `pip install -e .`
2. Try the basic examples above
3. Read the full `README.md`
4. Explore the source code
5. Build your own evaluation pipelines!

---

**Version**: 0.1.0  
**Last Updated**: 2025-11-30
