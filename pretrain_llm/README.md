# Pre-training LLM Experiments

This directory contains notebooks, scripts, and utilities for pre-training language models.

## Directory Structure

```
pretrain_llm/
├── Lesson_1.ipynb              # Introduction to pretraining & model loading
├── Lesson_2.ipynb              # Data preparation & cleaning
├── Lesson_3.ipynb              # Data packaging & tokenization
├── download_models.py          # Driver script for downloading models
├── data/                       # Data files (preprocessed datasets)
├── models/                     # Downloaded model files
├── docs/                       # Documentation
│   └── model-downloading-guide.md
└── requirements.txt            # Python dependencies
```

## Quick Start

### 1. Download a Model

```bash
python download_models.py \
  --model upstage/TinySolar-248m-4k \
  --output ./models/TinySolar-248m-4k \
  --verify
```

### 2. Use in Notebook

```python
from src.llm_lab.pretrain_llm.model_loader import load_model_with_fallback
from pathlib import Path

model, tokenizer = load_model_with_fallback(
    "upstage/TinySolar-248m-4k",
    local_dir=Path("./models/TinySolar-248m-4k")
)
```

### 3. Run Notebooks

Open any of the Lesson notebooks in Jupyter:

```bash
jupyter notebook Lesson_1.ipynb
```

## Documentation

- **[Model Downloading Guide](docs/model-downloading-guide.md)** - Comprehensive guide on downloading and caching models

## Lessons Overview

### Lesson 1: Why Pretraining?

- Load pretrained models
- Generate text samples
- Compare general vs. fine-tuned vs. continued pretrained models

### Lesson 2: Data Preparation

- Source datasets from HuggingFace Hub
- Scrape data from the web
- Clean data (length filters, deduplication, language detection)
- Save preprocessed datasets

### Lesson 3: Data Packaging

- Tokenize text data
- Pack sequences for efficient training
- Create training-ready datasets

## Reusable Components

The notebooks have been refactored into reusable modules in `src/llm_lab/pretrain_llm/`:

- **`model_loader.py`** - Smart model downloading and caching
- **`config.py`** - Configuration management
- **`data.py`** - Dataset loading and processing
- **`model.py`** - Model wrappers
- **`trainer.py`** - Training loop

## Models Used

- **TinySolar-248m-4k** - General pretrained model (248M params)
- **TinySolar-248m-4k-code-instruct** - Fine-tuned for code
- **TinySolar-248m-4k-py** - Continued pretrained on Python

All models are from [Upstage](https://huggingface.co/upstage).

## Requirements

### Quick Setup (Recommended)

If you haven't set up the environment yet, use the streamlined installation:

```bash
# From the project root
mamba env create -f environment.yml
mamba activate llm-lab
```

This single command installs all dependencies including:
- `transformers` - HuggingFace transformers
- `datasets` - HuggingFace datasets
- `torch` - PyTorch (with sympy=1.13.1 pinned for compatibility)
- `fasttext` - Language detection (Lesson 2)
- `jupyter` & `ipykernel` - For running notebooks
- All dev tools (pytest, black, ruff, etc.)

See [Installation Guide](../docs/installation.md) for detailed setup instructions.

### Jupyter Kernel Setup

After installing, register the kernel for Jupyter notebooks:

```bash
python -m ipykernel install --user --name llm-lab --display-name "Python (llm-lab)"
```

Then select "Python (llm-lab)" as the kernel when running notebooks in VS Code or Jupyter.

### Manual Installation

If you only need pretrain_llm dependencies:

```bash
pip install -r requirements.txt
```

## Tips

1. **Download models before running notebooks** to avoid waiting during execution
2. **Use local paths** in notebooks for faster loading
3. **Check the docs/** directory for detailed guides
4. **Run with CPU** if you don't have GPU access (set `device_map="cpu"`)
