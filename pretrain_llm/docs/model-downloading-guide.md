# Model Downloading Guide

This guide covers all the different ways to download and load pretrained models from HuggingFace Hub for use in your pre-training experiments.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Download Methods Comparison](#download-methods-comparison)
3. [Method 1: Using the Driver Script](#method-1-using-the-driver-script)
4. [Method 2: Using Python API](#method-2-using-python-api)
5. [Method 3: Using HuggingFace CLI](#method-3-using-huggingface-cli)
6. [Method 4: Using Git Clone](#method-4-using-git-clone)
7. [Method 5: Direct transformers Download](#method-5-direct-transformers-download)
8. [Caching Behavior](#caching-behavior)
9. [Offline Usage](#offline-usage)
10. [Troubleshooting](#troubleshooting)

---

## Quick Start

**Recommended for most users:**

```bash
# From the pretrain_llm directory
python download_models.py \
  --model upstage/TinySolar-248m-4k \
  --output ./models/TinySolar-248m-4k \
  --verify
```

Then in your notebook or script:

```python
from src.llm_lab.pretrain_llm.model_loader import load_model_with_fallback
from pathlib import Path

model, tokenizer = load_model_with_fallback(
    "upstage/TinySolar-248m-4k",
    local_dir=Path("./models/TinySolar-248m-4k")
)
```

---

## Download Methods Comparison

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Driver Script** | Easy CLI, verification, logging | Requires setup | First-time downloads |
| **Python API** | Programmatic, flexible | More code | Integration into workflows |
| **HF CLI** | Official tool, robust | Extra dependency | Production environments |
| **Git Clone** | Full repo history | Requires git-lfs, slow | Development/inspection |
| **Direct transformers** | Simple, one-liner | Less control | Quick experiments |

---

## Method 1: Using the Driver Script

### Basic Usage

```bash
python download_models.py \
  --model upstage/TinySolar-248m-4k \
  --output ./models/TinySolar-248m-4k
```

### With Verification

```bash
python download_models.py \
  --model upstage/TinySolar-248m-4k \
  --output ./models/TinySolar-248m-4k \
  --verify \
  --device cpu \
  --dtype bfloat16
```

### Download Multiple Models

```bash
# Create a simple bash script
for model in "gpt2" "upstage/TinySolar-248m-4k" "upstage/TinySolar-248m-4k-code-instruct"; do
  python download_models.py --model "$model" --output "./models/$(basename $model)"
done
```

### Available Options

- `--model`: HuggingFace model identifier (required)
- `--output`: Local directory path (required)
- `--device`: Device for loading (`auto`, `cpu`, `cuda`, `mps`)
- `--dtype`: Data type (`float32`, `float16`, `bfloat16`)
- `--no-tokenizer`: Skip tokenizer download
- `--verify`: Load model after download to verify

---

## Method 2: Using Python API

### Simple Download

```python
from pathlib import Path
from src.llm_lab.pretrain_llm.model_loader import download_model_manually

download_model_manually(
    model_name="upstage/TinySolar-248m-4k",
    output_dir=Path("./models/TinySolar-248m-4k"),
    include_tokenizer=True
)
```

### Smart Loading with Caching

```python
from pathlib import Path
from src.llm_lab.pretrain_llm.model_loader import load_model_with_fallback

# First run: downloads and caches
# Subsequent runs: loads from cache
model, tokenizer = load_model_with_fallback(
    model_name="upstage/TinySolar-248m-4k",
    local_dir=Path("./models/TinySolar-248m-4k"),
    device_map="auto",
    torch_dtype=torch.bfloat16
)
```

### Using ModelLoader Class

```python
from pathlib import Path
import torch
from src.llm_lab.pretrain_llm.model_loader import ModelLoader

loader = ModelLoader(
    model_name_or_path="upstage/TinySolar-248m-4k",
    local_cache_dir=Path("./models/TinySolar-248m-4k"),
    device_map="cpu",
    torch_dtype=torch.bfloat16,
    use_fast_tokenizer=True
)

# Load separately
model = loader.load_model()
tokenizer = loader.load_tokenizer()

# Or load together
model, tokenizer = loader.load_model_and_tokenizer()
```

### Notebook-Friendly Pattern

```python
import os
import torch
from pathlib import Path
from src.llm_lab.pretrain_llm.model_loader import load_model_with_fallback

model_name = "upstage/TinySolar-248m-4k"
local_path = Path("./models/TinySolar-248m-4k")

# This pattern handles both first-time download and subsequent loads
model, tokenizer = load_model_with_fallback(
    model_name=model_name,
    local_dir=local_path if local_path.exists() else None,
    device_map="cpu",
    torch_dtype=torch.bfloat16
)

print(f"Model loaded from: {local_path if local_path.exists() else 'HuggingFace Hub'}")
```

---

## Method 3: Using HuggingFace CLI

### Installation

```bash
pip install huggingface_hub
```

### Download Model (Recommended: `hf download`)

The modern `hf download` command is the recommended approach:

```bash
hf download upstage/TinySolar-248m-4k \
  --local-dir ./models/TinySolar-248m-4k
```

**Note:** The `--local-dir` option automatically downloads to a local folder without symlinks.

### Legacy Command (Deprecated)

The older `huggingface-cli download` still works but is deprecated:

```bash
huggingface-cli download upstage/TinySolar-248m-4k \
  --local-dir ./models/TinySolar-248m-4k \
  --local-dir-use-symlinks False
```

**Warning:** You'll see a deprecation warning. Use `hf download` instead.

### Download Specific Files (with `hf download`)

Using the modern command with include/exclude patterns:

```bash
# Download only safetensors and json files
hf download upstage/TinySolar-248m-4k \
  --include "*.safetensors" "*.json" \
  --local-dir ./models/TinySolar-248m-4k
```

### Exclude Unnecessary Files

```bash
# Exclude specific file types
hf download upstage/TinySolar-248m-4k \
  --exclude "*.msgpack" "*.h5" "*.ot" \
  --local-dir ./models/TinySolar-248m-4k
```

### Additional `hf download` Options

**Quiet mode** (only show final path):
```bash
hf download upstage/TinySolar-248m-4k --local-dir ./models/TinySolar-248m-4k --quiet
```

**Dry-run** (check what would be downloaded):
```bash
hf download upstage/TinySolar-248m-4k --dry-run
```

**Download specific revision** (branch, tag, or commit):
```bash
hf download upstage/TinySolar-248m-4k --revision main --local-dir ./models/TinySolar-248m-4k
```

**Download single file**:
```bash
hf download upstage/TinySolar-248m-4k config.json
```

### Using Python API

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="upstage/TinySolar-248m-4k",
    local_dir="./models/TinySolar-248m-4k",
    local_dir_use_symlinks=False,
    ignore_patterns=["*.msgpack", "*.h5"]
)
```

---

## Method 4: Using Git Clone

### Prerequisites

```bash
# Install git-lfs
brew install git-lfs  # macOS
# or
sudo apt-get install git-lfs  # Ubuntu/Debian

# Initialize git-lfs
git lfs install
```

### Clone Repository

```bash
git clone https://huggingface.co/upstage/TinySolar-248m-4k ./models/TinySolar-248m-4k
```

### Shallow Clone (Faster)

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://huggingface.co/upstage/TinySolar-248m-4k ./models/TinySolar-248m-4k
cd ./models/TinySolar-248m-4k
git lfs pull
```

### Pros and Cons

**Pros:**
- Full git history
- Easy to inspect model card and files
- Can track changes

**Cons:**
- Requires git-lfs setup
- Slower than other methods
- Downloads .git directory (extra space)

---

## Method 5: Direct transformers Download

### Auto-Download on First Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Downloads to ~/.cache/huggingface/hub/ automatically
model = AutoModelForCausalLM.from_pretrained(
    "upstage/TinySolar-248m-4k",
    device_map="cpu",
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained("upstage/TinySolar-248m-4k")
```

### Save to Custom Location

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Download and save
model = AutoModelForCausalLM.from_pretrained("upstage/TinySolar-248m-4k")
tokenizer = AutoTokenizer.from_pretrained("upstage/TinySolar-248m-4k")

# Save locally
model.save_pretrained("./models/TinySolar-248m-4k")
tokenizer.save_pretrained("./models/TinySolar-248m-4k")
```

---

## Caching Behavior

### Default Cache Location

```bash
~/.cache/huggingface/hub/models--upstage--TinySolar-248m-4k/
```

### How Caching Works

1. **First call to `from_pretrained()`:**
   - Checks cache → Not found
   - Downloads from HuggingFace Hub
   - Saves to cache directory
   - Returns model

2. **Subsequent calls:**
   - Checks cache → Found!
   - Loads from cache (no download)
   - **Only checks for updates if `force_download=True`**

3. **With local path:**
   ```python
   model = AutoModelForCausalLM.from_pretrained("./models/TinySolar-248m-4k")
   ```
   - **Never downloads** (always loads from local)
   - Fails if directory doesn't exist

### Check Cache Size

```bash
du -sh ~/.cache/huggingface/
```

### Clear Cache

```bash
# Clear all cached models
rm -rf ~/.cache/huggingface/hub/

# Clear specific model
rm -rf ~/.cache/huggingface/hub/models--upstage--TinySolar-248m-4k/
```

### Custom Cache Location

```bash
# Set environment variable
export HF_HOME=/path/to/custom/cache

# Or in Python
import os
os.environ['HF_HOME'] = '/path/to/custom/cache'
```

---

## Offline Usage

### Prepare for Offline Use

1. **Download models while online:**
   ```bash
   python download_models.py \
     --model upstage/TinySolar-248m-4k \
     --output ./models/TinySolar-248m-4k \
     --verify
   ```

2. **Use local paths in code:**
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       "./models/TinySolar-248m-4k",
       local_files_only=True  # Prevents network access
   )
   ```

3. **Set offline mode globally:**
   ```bash
   export HF_HUB_OFFLINE=1
   ```

### Verify Offline Capability

```python
import os
os.environ['HF_HUB_OFFLINE'] = '1'

from transformers import AutoModelForCausalLM

# This will fail if model not cached locally
model = AutoModelForCausalLM.from_pretrained("./models/TinySolar-248m-4k")
```

---

## Troubleshooting

### Issue: "Connection Error"

**Solution:** Use local path or check network connection
```python
# Instead of HF identifier
model = AutoModelForCausalLM.from_pretrained("upstage/TinySolar-248m-4k")

# Use local path
model = AutoModelForCausalLM.from_pretrained("./models/TinySolar-248m-4k")
```

### Issue: "Out of Memory"

**Solution:** Use CPU or quantization
```python
# Force CPU
model = AutoModelForCausalLM.from_pretrained(
    "upstage/TinySolar-248m-4k",
    device_map="cpu"
)

# Or use 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    "upstage/TinySolar-248m-4k",
    load_in_8bit=True,
    device_map="auto"
)
```

### Issue: "Model files not found"

**Solution:** Verify directory structure
```bash
ls -la ./models/TinySolar-248m-4k/

# Should contain:
# - config.json
# - pytorch_model.bin or model.safetensors
# - tokenizer.json
# - tokenizer_config.json
```

### Issue: "Slow downloads"

**Solutions:**
1. Use HuggingFace CLI (faster than transformers)
2. Download during off-peak hours
3. Use mirror sites (if available)
4. Resume interrupted downloads:
   ```bash
   huggingface-cli download upstage/TinySolar-248m-4k \
     --resume-download \
     --local-dir ./models/TinySolar-248m-4k
   ```

### Issue: "Disk space"

**Check model size before downloading:**
```python
from huggingface_hub import model_info

info = model_info("upstage/TinySolar-248m-4k")
print(f"Model size: {info.safetensors.total / 1e9:.2f} GB")
```

**Clean up old models:**
```bash
# List cached models
ls -lh ~/.cache/huggingface/hub/

# Remove specific model
rm -rf ~/.cache/huggingface/hub/models--upstage--TinySolar-248m-4k/
```

---

## File Formats

### Model Files

- **`pytorch_model.bin`**: PyTorch pickle format (older, less safe)
- **`model.safetensors`**: SafeTensors format (preferred, faster, safer)
- **`config.json`**: Model architecture configuration

### Tokenizer Files

- **`tokenizer.json`**: Fast tokenizer data
- **`tokenizer_config.json`**: Tokenizer settings
- **`special_tokens_map.json`**: Special tokens (BOS, EOS, PAD, etc.)
- **`vocab.json`** / **`merges.txt`**: Vocabulary (for BPE tokenizers)

### Optional Files

- **`generation_config.json`**: Default generation parameters
- **`README.md`**: Model card with usage info
- **`.gitattributes`**: Git LFS tracking

---

## Best Practices

1. **Use local caching for repeated experiments**
   ```python
   load_model_with_fallback(model_name, local_dir=Path("./models/..."))
   ```

2. **Verify downloads**
   ```bash
   python download_models.py --model ... --output ... --verify
   ```

3. **Use safetensors format when available**
   - Faster loading
   - Better security
   - Smaller file size

4. **Set appropriate device_map**
   - `"auto"`: Automatic device selection
   - `"cpu"`: Force CPU (safe for large models)
   - `"cuda"`: Force GPU

5. **Clean up cache periodically**
   ```bash
   du -sh ~/.cache/huggingface/
   rm -rf ~/.cache/huggingface/hub/models--old-model-name/
   ```

6. **Use environment variables for configuration**
   ```bash
   export HF_HOME=/custom/cache
   export HF_HUB_OFFLINE=1  # For offline mode
   ```

---

## Additional Resources

- [HuggingFace Hub Documentation](https://huggingface.co/docs/hub/index)
- [Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [Model Hub](https://huggingface.co/models)
- [TinySolar Models](https://huggingface.co/upstage)
