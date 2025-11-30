# LLM Eval - Reusable LLM Evaluation Toolkit

A modular, reusable Python package for evaluating Large Language Models on various tasks including mathematical reasoning and safety classification.

## 🎯 Purpose

This package extracts and generalizes the evaluation code from the DeepLearning.AI post-training course modules (M1, M2, ...) into a standalone, reusable toolkit. As you progress through the course, new evaluation capabilities will be added to this package.

## 📦 Installation

### From Source (Development)

```bash
cd posttrain_llm/llm_eval
pip install -e .
```

### With Development Dependencies

```bash
pip install -e ".[dev]"
```

## 🚀 Quick Start

### Basic Usage

```python
from llm_eval import ServeLLM, evaluate_math_reasoning
from datasets import load_dataset

# Load a model
with ServeLLM("deepseek-ai/deepseek-math-7b-base") as llm:
    response = llm.generate_response("What is 2+2?")
    print(response)

# Evaluate on GSM8K
dataset = load_dataset("gsm8k", "main", split="test")
accuracy, results = evaluate_math_reasoning("model-name", dataset, num_samples=50)
print(f"Accuracy: {accuracy:.2%}")
```

### Safety Evaluation

```python
from llm_eval import evaluate_safety
from datasets import load_dataset

harmful_dataset = load_dataset("jailbreakbench_harmful")
benign_prompts = ["What's the weather?", "How do I cook pasta?"]

results = evaluate_safety(
    "meta-llama/Llama-Guard-3-8B",
    harmful_dataset,
    benign_prompts,
    num_harmful=20,
    num_benign=10
)

print(f"Harmful Detection Rate: {results['metrics']['harmful_detection_rate']:.2%}")
print(f"False Positive Rate: {results['metrics']['false_positive_rate']:.2%}")
```

## 📚 Package Structure

```
llm_eval/
├── __init__.py                 # Main package exports
├── core/                       # Core model serving
│   ├── model_service.py        # ServeLLM class
│   └── inference.py            # Batch inference utilities
├── metrics/                    # Evaluation metrics
│   ├── math_reasoning.py       # GSM8K evaluation
│   └── safety.py               # Safety classification
├── utils/                      # Helper utilities
│   ├── display.py              # Formatted output
│   └── huggingface.py          # HF authentication
├── setup.py                    # Package installation
└── README.md                   # This file
```

## 🔧 Core Components

### ServeLLM - Model Service Class

Handles model loading, inference, and memory management:

```python
from llm_eval import ServeLLM

# Automatic device selection (CUDA/MPS/CPU)
with ServeLLM("model-name") as llm:
    response = llm.generate_response(
        "Your prompt here",
        max_tokens=512,
        temperature=0.7
    )
```

**Features:**
- Automatic device selection (CUDA, MPS, CPU)
- Retry logic for network issues
- Context manager for automatic cleanup
- Memory-efficient inference

### Mathematical Reasoning

Evaluate models on GSM8K-style math problems:

```python
from llm_eval.metrics import extract_number, evaluate_math_reasoning

# Extract numerical answers
answer = extract_number("The answer is #### 42")  # Returns 42.0

# Full evaluation
accuracy, results = evaluate_math_reasoning(
    model_path="model-name",
    dataset=gsm8k_dataset,
    num_samples=30
)
```

### Safety Classification

Evaluate safety models (Llama Guard):

```python
from llm_eval.metrics import parse_llama_guard_response, evaluate_safety

# Parse Llama Guard output
result = parse_llama_guard_response("unsafe\nS1\nS5")
# Returns: {'classification': 'unsafe', 'categories': ['S1', 'S5']}

# Full safety evaluation
results = evaluate_safety(
    model_path="Llama-Guard-3-8B",
    harmful_prompts=harmful_dataset,
    benign_prompts=benign_list
)
```

## 📊 API Reference

### Core Module

#### `ServeLLM(model_name, device="auto")`

Model service class for loading and running LLMs.

**Parameters:**
- `model_name` (str): HuggingFace model ID or local path
- `device` (str): Device to use ('auto', 'cuda', 'mps', 'cpu')

**Methods:**
- `generate_response(prompt, max_tokens=512, temperature=0.7, ...)`: Generate text
- `cleanup()`: Free memory
- Context manager support (`with` statement)

### Metrics Module

#### `evaluate_math_reasoning(model_path, dataset, num_samples=30)`

Evaluate model on GSM8K-style problems.

**Returns:** `(accuracy: float, results: List[Dict])`

#### `evaluate_safety(model_path, harmful_prompts, benign_prompts, ...)`

Evaluate safety classification model.

**Returns:** `Dict` with harmful_results, benign_results, and metrics

#### `extract_number(text)`

Extract numerical answer from text.

**Returns:** `float` or `None`

#### `parse_llama_guard_response(output)`

Parse Llama Guard model output.

**Returns:** `Dict` with classification and categories

### Utils Module

#### `validate_token()`

Validate HuggingFace token.

**Returns:** `bool`

#### `display_section_header(title, level=1)`

Display formatted section header.

## 🎓 Course Integration

This package is designed to grow with you as you progress through the DeepLearning.AI post-training course:

### Module M1 (Complete) ✅
- Mathematical reasoning evaluation (GSM8K)
- Safety classification (Llama Guard)
- Model serving with retry logic

### Module M2 (Coming Soon) 🔜
- Additional metrics to be added
- New evaluation tasks

### Module M3+ (Future) 📅
- More evaluation capabilities as you progress

## 🧪 Testing

Run the test suite:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run with coverage
pytest --cov=llm_eval tests/
```

## 📝 Examples

See the `examples/` directory for complete examples:

- `basic_inference.py` - Simple model inference
- `math_evaluation.py` - GSM8K evaluation
- `safety_evaluation.py` - Safety classification
- `batch_processing.py` - Batch inference

## 🤝 Contributing

As you work through the course and solve more problems:

1. Add new evaluation functions to appropriate modules
2. Update `__init__.py` to export new functions
3. Add tests for new functionality
4. Update this README with examples

## 📄 License

MIT License - See LICENSE file for details

## 🔗 Related

- **M1 Module**: `posttrain_llm/M1/` - Educational materials and exercises
- **Course**: DeepLearning.AI Post-Training for LLMs
- **Repository**: https://github.com/pleiadian53/llm-lab

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Version**: 0.1.0  
**Status**: Active Development  
**Last Updated**: 2025-11-30
