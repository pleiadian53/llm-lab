# Model Training Pipeline Comparison - Evaluation Framework

This directory contains a modular evaluation framework for comparing different stages of LLM training (Base, Fine-Tuned, RL) on mathematical reasoning and safety classification tasks.

## 📁 Project Structure

```
posttrain_llm/M1/
├── lib/                                    # Reusable evaluation modules
│   ├── __init__.py                         # Package initialization
│   ├── model_evaluation.py                # Mathematical reasoning evaluation
│   └── safety_evaluation.py               # Safety classification evaluation
├── utils/                                  # Core utilities
│   ├── serve_llm.py                        # LLM serving wrapper
│   └── utils.py                            # Helper functions
├── run_evaluation.py                       # Main driver script ⭐
├── M1_G1_Inspecting_Finetuned_vs_Base_Model.py  # Original notebook (completed)
└── M1_G1_Inspecting_Finetuned_vs_Base_Model.ipynb  # Jupyter notebook
```

## 🚀 Quick Start

### Run Quick Evaluation (3 test prompts)

```bash
cd posttrain_llm/M1
python run_evaluation.py --mode quick
```

### Run Full Evaluation (30 GSM8K problems)

```bash
python run_evaluation.py --mode full --num-samples 30
```

### Run Safety Evaluation (Llama Guard)

```bash
python run_evaluation.py --mode safety --num-harmful 10 --num-benign 5
```

### Run All Evaluations

```bash
python run_evaluation.py --mode all
```

## 📚 Module Documentation

### `lib/model_evaluation.py`

Mathematical reasoning evaluation functions:

- **`process_prompts(model_name, prompts)`** - Generate responses for a list of prompts
- **`extract_number(text)`** - Extract numerical answers from model outputs
- **`evaluate_model_correctness(model_path, dataset, num_samples)`** - Evaluate model accuracy on GSM8K
- **`score_response(response, expected_keyword)`** - Score a single response
- **`score_all_responses(model_results, expected_keywords)`** - Score multiple responses

### `lib/safety_evaluation.py`

Safety classification evaluation functions:

- **`parse_llama_guard_response(output)`** - Parse Llama Guard output format
- **`calculate_safety_metrics(harmful_results, benign_results)`** - Calculate TPR, TNR, FPR, FNR
- **`analyze_safety_categories(results)`** - Analyze violation category frequencies
- **`evaluate_safety_model(model_path, harmful_prompts, benign_prompts)`** - Full safety evaluation

## 💻 Usage Examples

### Example 1: Evaluate a Single Model

```python
from lib.model_evaluation import process_prompts

prompts = ["What is 2+2?", "Solve: x + 5 = 10"]
results = process_prompts("/app/models/deepseek-math-7b-base", prompts)

for prompt, response in zip(prompts, results):
    print(f"Q: {prompt}")
    print(f"A: {response}\n")
```

### Example 2: Extract Numbers from Responses

```python
from lib.model_evaluation import extract_number

text1 = "The answer is #### 42"
text2 = "We calculate 2 + 2 = 4"

print(extract_number(text1))  # 42.0
print(extract_number(text2))  # 4.0
```

### Example 3: Safety Classification

```python
from lib.safety_evaluation import parse_llama_guard_response

response = "unsafe\nS1\nS10"
result = parse_llama_guard_response(response)

print(result)
# {'classification': 'unsafe', 'categories': ['S1', 'S10']}
```

### Example 4: Full GSM8K Evaluation

```python
from datasets import load_from_disk
from lib.model_evaluation import evaluate_model_correctness

# Load dataset
dataset = load_from_disk("/app/data/gsm8k", "main")['test']

# Evaluate model
accuracy, results = evaluate_model_correctness(
    "/app/models/deepseek-math-7b-instruct",
    dataset,
    num_samples=50
)

print(f"Accuracy: {accuracy:.2%}")
```

## 🎯 Evaluation Modes

### Quick Mode
- **Purpose**: Fast sanity check
- **Dataset**: 3 handcrafted math problems
- **Models**: Base, Fine-Tuned, RL
- **Time**: ~2-3 minutes
- **Output**: Response comparison + keyword scoring

### Full Mode
- **Purpose**: Comprehensive accuracy measurement
- **Dataset**: GSM8K (configurable sample size)
- **Models**: Base, Fine-Tuned, RL
- **Time**: ~10-30 minutes (depending on sample size)
- **Output**: Accuracy metrics + detailed results

### Safety Mode
- **Purpose**: Safety classification evaluation
- **Dataset**: JailbreakBench harmful prompts + benign prompts
- **Model**: Llama Guard
- **Time**: ~5-10 minutes
- **Output**: TPR, TNR, FPR, FNR + category analysis

## 📊 Output Examples

### Quick Evaluation Output

```
==================================================
PROCESSING BASE MODEL
==================================================

Prompt 1: What is the area of a rectangle with a length of 8 units and a width of 5 units?
Base Model Response: The area is 8 × 5 = 40 square units.

SCORING RESULTS:
============================================================
   Prompt Expected  Base Score  SFT Score  RL Score
 Prompt 1       40           1          1         1
 Prompt 2    x = 2           0          1         1
 Prompt 3   cos(x)           1          1         1

Average Scores:
        Base Model: 0.67
  Fine-Tuned Model: 1.00
          RL Model: 1.00
```

### Full Evaluation Output

```
====================BASE MODEL====================
Evaluating /app/models/deepseek-math-7b-base on 30 GSM8K problems...
Processing: 100%|████████████████| 30/30 [05:23<00:00,  0.09it/s]

Example 1:
Question: Janet's ducks lay 16 eggs per day...
Gold: 18.0, Model: 18.0, Correct: True

Base Model Accuracy: 0.567 (56.7%)

CORRECTNESS SUMMARY:
========================================
    Base Model: 0.567 (56.7%)
     SFT Model: 0.833 (83.3%)
      RL Model: 0.900 (90.0%)
```

## 🔧 Command-Line Options

```bash
python run_evaluation.py --help
```

Options:
- `--mode {quick|full|safety|all}` - Evaluation mode (default: quick)
- `--num-samples N` - Number of GSM8K samples (default: 30)
- `--num-harmful N` - Number of harmful prompts (default: 10)
- `--num-benign N` - Number of benign prompts (default: 5)

## 📝 Notes

- **Model Paths**: Update paths in `run_evaluation.py` if your models are in different locations
- **Dataset Paths**: Datasets are expected at `/app/data/gsm8k` and `/app/data/jailbreakbench_harmful`
- **Memory**: Full evaluation requires ~16GB RAM for 7B models
- **GPU**: Recommended for faster inference (falls back to CPU if unavailable)

## 🐛 Troubleshooting

### Import Errors
```bash
# Make sure you're in the M1 directory
cd posttrain_llm/M1

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Model Not Found
```python
# Check model paths in run_evaluation.py
BASE_MODEL = "/app/models/deepseek-math-7b-base"  # Update this
```

### Dataset Not Found
```python
# Update dataset paths
gsm8k_dataset = load_from_disk("/app/data/gsm8k", "main")  # Update this
```

## 📖 Related Documentation

- [Device Usage Guide](DEVICE_USAGE_GUIDE.md) - Multi-device support (CUDA, MPS, CPU)
- [Changelog](CHANGELOG.md) - Module features and updates
- [Transformers Tutorial](../docs/transformers-tutorial.md) - HuggingFace library guide

## 🤝 Contributing

When adding new evaluation functions:
1. Add to appropriate module (`model_evaluation.py` or `safety_evaluation.py`)
2. Update `lib/__init__.py` exports
3. Add usage example to this README
4. Update `run_evaluation.py` if adding new modes

---

**Last Updated**: 2025-11-29  
**Maintainer**: pleiadian53
