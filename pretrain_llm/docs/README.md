# Pre-training Documentation

Welcome to the pre-training documentation! This directory contains guides and examples for working with pretrained language models.

## Available Documentation

### 🔧 [Environment Integration Guide](environment-integration.md)

How pretrain_llm integrates with the main llm-lab environment:

- Streamlined installation (2 steps instead of 4+)
- Sympy compatibility resolution
- Dual-source dependency architecture
- Troubleshooting import issues
- Keeping dependencies in sync

**Start here if:** You're setting up the environment or having import issues.

### 📚 [Model Downloading Guide](model-downloading-guide.md)

Comprehensive guide covering all methods for downloading and caching pretrained models:

- Using the driver script
- Python API methods
- HuggingFace CLI
- Git clone approach
- Direct transformers download
- Caching behavior and offline usage
- Troubleshooting common issues

**Start here if:** You need to download models or understand caching behavior.

### 💡 [Usage Examples](usage-examples.md)

Practical code examples for common tasks:

- Download and load models
- Compare multiple models
- Batch processing
- Error handling
- Integration with training pipeline

**Start here if:** You want to see practical code examples and patterns.

### [Tutorials](tutorials/)

In-depth guides on specific topics:

- **[Text Streaming Guide](tutorials/text-streaming-guide.md)** - Understanding `TextStreamer` for real-time generation
  - What is streaming and why use it
  - Configuration options (`skip_prompt`, `skip_special_tokens`)
  - Use cases: chatbots, story generation, code generation
  - Advanced patterns: custom streamers, file streaming, callbacks
  - Performance considerations and best practices

- **[Generation Parameters Guide](tutorials/generation-parameters-guide.md)** - Understanding `model.generate()` parameters
  - `do_sample`: Deterministic vs stochastic generation
  - `use_cache`: Key-value caching for 3-5x speedup
  - `temperature`: Controlling randomness in sampling
  - `repetition_penalty`: Avoiding repetitive output
  - Complete parameter reference and common patterns

**Start here if:** You want to learn about specific features in depth.

## Quick Reference

### Download a Model

```bash
cd pretrain_llm
python download_models.py \
  --model upstage/TinySolar-248m-4k \
  --output ./models/TinySolar-248m-4k \
  --verify
```

### Load in Python

```python
from llm_lab.pretrain_llm import load_model_with_fallback
from pathlib import Path

model, tokenizer = load_model_with_fallback(
    "upstage/TinySolar-248m-4k",
    local_dir=Path("./models/TinySolar-248m-4k")
)
```

### Load in Notebook

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

from src.llm_lab.pretrain_llm.model_loader import load_model_with_fallback

model, tokenizer = load_model_with_fallback(
    "upstage/TinySolar-248m-4k",
    local_dir=Path("./models/TinySolar-248m-4k")
)
```

## Models Used in Lessons

| Model | Size | Description | Use Case |
|-------|------|-------------|----------|
| TinySolar-248m-4k | 248M | General pretrained | Lesson 1 baseline |
| TinySolar-248m-4k-code-instruct | 248M | Fine-tuned on code | Lesson 1 comparison |
| TinySolar-248m-4k-py | 248M | Continued pretrained on Python | Lesson 1 comparison |

All models from [Upstage](https://huggingface.co/upstage).

## File Structure

```
docs/
├── README.md                      # This file
├── model-downloading-guide.md     # Comprehensive download guide
└── usage-examples.md              # Code examples
```

## Getting Help

1. **Check the guides** - Most questions are answered in the documentation
2. **Review examples** - See usage-examples.md for common patterns
3. **Check troubleshooting** - See the troubleshooting section in model-downloading-guide.md

## Contributing

When adding new documentation:

1. Keep it practical and example-driven
2. Include code snippets that can be copy-pasted
3. Add troubleshooting tips for common issues
4. Update this README with links to new docs
