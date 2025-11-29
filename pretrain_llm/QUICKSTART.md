# Quick Start Guide

Get up and running with model downloading and loading in 5 minutes.

## Prerequisites

Ensure you have the environment set up. If not, run from the project root:

```bash
mamba env create -f environment.yml
mamba activate llm-lab
```

See [Installation Guide](../docs/installation.md) for details. This installs all dependencies including PyTorch with the correct sympy version (1.13.1) for compatibility.

### Register Jupyter Kernel

To run notebooks, register the kernel:

```bash
python -m ipykernel install --user --name llm-lab --display-name "Python (llm-lab)"
```

Then select "Python (llm-lab)" as your kernel in VS Code or Jupyter.

## Step 1: Download a Model

```bash
cd pretrain_llm

python download_models.py \
  --model upstage/TinySolar-248m-4k \
  --output ./models/TinySolar-248m-4k \
  --verify
```

This will:
- Download the model from HuggingFace Hub
- Save it to `./models/TinySolar-248m-4k/`
- Verify the download by loading and testing it

## Step 2: Use in Your Code

### In a Notebook

```python
# Add at the top of your notebook
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

from src.llm_lab.pretrain_llm.model_loader import load_model_with_fallback
import torch

# Load model (uses local cache if available)
model, tokenizer = load_model_with_fallback(
    "upstage/TinySolar-248m-4k",
    local_dir=Path("./models/TinySolar-248m-4k"),
    device_map="cpu",
    torch_dtype=torch.bfloat16
)

# Generate text
prompt = "I am an engineer. I love"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### In a Python Script

```python
from pathlib import Path
from llm_lab.pretrain_llm import load_model_with_fallback
import torch

# Load model
model, tokenizer = load_model_with_fallback(
    "upstage/TinySolar-248m-4k",
    local_dir=Path("./models/TinySolar-248m-4k"),
    device_map="cpu",
    torch_dtype=torch.bfloat16
)

# Use model
prompt = "Hello, world!"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Step 3: Download More Models (Optional)

```bash
# Download all models used in the lessons
python download_models.py \
  --model upstage/TinySolar-248m-4k-code-instruct \
  --output ./models/TinySolar-248m-4k-code-instruct

python download_models.py \
  --model upstage/TinySolar-248m-4k-py \
  --output ./models/TinySolar-248m-4k-py
```

## Common Options

### Download with GPU Support

```bash
python download_models.py \
  --model upstage/TinySolar-248m-4k \
  --output ./models/TinySolar-248m-4k \
  --device cuda \
  --dtype float16
```

### Download Without Verification

```bash
python download_models.py \
  --model upstage/TinySolar-248m-4k \
  --output ./models/TinySolar-248m-4k
```

### Download Model Only (No Tokenizer)

```bash
python download_models.py \
  --model upstage/TinySolar-248m-4k \
  --output ./models/TinySolar-248m-4k \
  --no-tokenizer
```

## Troubleshooting

### "Command not found: python"

Try `python3` instead:

```bash
python3 download_models.py --model ... --output ...
```

### "Module not found: transformers"

Install dependencies:

```bash
pip install -r requirements.txt
```

### "accelerate required" error

If you see an error about `accelerate` being required:

```bash
pip install accelerate
```

### "Placeholder storage has not been allocated on MPS device" (Apple Silicon)

This is a known PyTorch MPS backend issue with certain models. **Use CPU instead:**

```bash
python download_models.py \
  --model upstage/TinySolar-248m-4k \
  --output ./models/TinySolar-248m-4k \
  --device cpu \
  --verify
```

Or download without verification using `hf download` (recommended):

```bash
hf download upstage/TinySolar-248m-4k \
  --local-dir ./models/TinySolar-248m-4k
```

**Note:** The older `huggingface-cli download` command is deprecated. Use `hf download` instead.

### "Out of memory"

Use CPU instead of GPU:

```bash
python download_models.py \
  --model upstage/TinySolar-248m-4k \
  --output ./models/TinySolar-248m-4k \
  --device cpu
```

### "Connection error"

Check your internet connection. If behind a proxy, set:

```bash
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
```

## Next Steps

- **Read the full guide:** [Model Downloading Guide](docs/model-downloading-guide.md)
- **See more examples:** [Usage Examples](docs/usage-examples.md)
- **Explore notebooks:** Open `Lesson_1.ipynb`, `Lesson_2.ipynb`, or `Lesson_3.ipynb`
- **Check the refactoring summary:** [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)

## Help

For detailed documentation on all download methods and advanced usage, see:

- [docs/model-downloading-guide.md](docs/model-downloading-guide.md) - Comprehensive guide
- [docs/usage-examples.md](docs/usage-examples.md) - Code examples
- [docs/README.md](docs/README.md) - Documentation index

## Tips

1. **Download models before running notebooks** - Saves time during experimentation
2. **Use local paths in notebooks** - Faster loading, works offline
3. **Verify downloads** - Use `--verify` flag to catch issues early
4. **Check disk space** - Models are ~500MB each
5. **Use CPU if no GPU** - Set `device_map="cpu"` in your code
