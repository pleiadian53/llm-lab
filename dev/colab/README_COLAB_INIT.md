# Colab Initialization Scripts

## Overview

These scripts set up your Google Colab environment for hybrid cloud-local development with the `llm-lab` project.

## Files

### `colab_init.py`

Complete initialization script that:

1. ✅ Mounts Google Drive
2. ✅ Configures persistent HuggingFace cache
3. ✅ Navigates to project directory
4. ✅ Installs `llm_eval` package
5. ✅ Sets up SSH tunnel for remote access

### Usage

**Option 1: Copy entire file into Colab cell**

```python
# Just copy the entire contents of colab_init.py into a Colab code cell and run
```

**Option 2: Load from Drive (after first sync)**

```python
# In Colab
%load /content/drive/MyDrive/work_MBP_M1/llm-lab/dev/colab/colab_init.py
# Then run the cell
```

**Option 3: Run directly from Drive**

```python
# In Colab
%run /content/drive/MyDrive/work_MBP_M1/llm-lab/dev/colab/colab_init.py
```

## What It Does

### 1. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

Your files appear at: `/content/drive/MyDrive/`

### 2. Configure Persistent Cache

```python
CACHE_DIR = "/content/drive/MyDrive/llm_cache"
os.environ['HF_HOME'] = CACHE_DIR
os.environ['HF_HUB_CACHE'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
```

**Benefits:**
- Downloaded models persist between sessions
- No need to re-download DeepSeek, Llama Guard, etc.
- Saves time and bandwidth

### 3. Navigate to Project

```python
PROJECT_DIR = "/content/drive/MyDrive/work_MBP_M1/llm-lab"
os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)
```

**Benefits:**
- Working directory set to your project
- Python can import your modules
- Ready to run code

### 4. Install llm_eval Package

```python
!pip install -e posttrain_llm/llm_eval
```

**Benefits:**
- Your reusable evaluation toolkit is available
- Import with: `from llm_eval import ServeLLM`
- No need to copy code between notebooks

### 5. Setup SSH Tunnel

```python
from colab_ssh import launch_ssh_cloudflared
launch_ssh_cloudflared(password="your_password")
```

**Benefits:**
- Connect VSCode to Colab
- Use terminal in Colab
- Full IDE experience with GPU access

## Configuration

### Change SSH Password

Edit this line in `colab_init.py`:

```python
SSH_PASSWORD = "change_me_123"  # TODO: Change this!
```

**Security Note:** This password is only for the current Colab session and resets when the runtime restarts.

### Custom Paths

If you use different paths, edit these variables:

```python
CACHE_DIR = "/content/drive/MyDrive/llm_cache"
PROJECT_DIR = "/content/drive/MyDrive/work_MBP_M1/llm-lab"
```

## After Initialization

### Verify Setup

```python
# Check current directory
!pwd
# Should show: /content/drive/MyDrive/work_MBP_M1/llm-lab

# Check llm_eval installation
import llm_eval
print(llm_eval.__version__)
# Should show: 0.1.0

# Check cache location
import os
print(os.environ['HF_HOME'])
# Should show: /content/drive/MyDrive/llm_cache
```

### Start Working

```python
# Import your package
from llm_eval import ServeLLM, evaluate_math_reasoning
from datasets import load_dataset

# Load a model (will cache to Drive)
with ServeLLM("deepseek-ai/deepseek-math-7b-base") as llm:
    response = llm.generate_response("What is 2+2?")
    print(response)

# Run evaluation
dataset = load_dataset("gsm8k", "main", split="test")
accuracy, results = evaluate_math_reasoning(
    "deepseek-ai/deepseek-math-7b-instruct",
    dataset,
    num_samples=30
)
print(f"Accuracy: {accuracy:.2%}")
```

## Workflow

### First Time Setup

1. **On Local Mac:**
   ```bash
   # Sync project to Google Drive
   ./scripts/sync_to_colab.py
   ```

2. **In Colab:**
   ```python
   # Run initialization script
   %run /content/drive/MyDrive/work_MBP_M1/llm-lab/dev/colab/colab_init.py
   ```

3. **Copy SSH config to local `~/.ssh/config`**

4. **Connect VSCode to Colab via SSH**

### Subsequent Sessions

1. **In Colab:**
   ```python
   # Just run the init script again
   %run /content/drive/MyDrive/work_MBP_M1/llm-lab/dev/colab/colab_init.py
   ```

2. **Update SSH config** (hostname changes each session)

3. **Reconnect VSCode**

### After Making Local Changes

1. **On Local Mac:**
   ```bash
   # Sync changes to Drive
   ./scripts/sync_to_colab.py
   ```

2. **In Colab:**
   ```python
   # Reload modules if needed
   import importlib
   import llm_eval
   importlib.reload(llm_eval)
   ```

## Troubleshooting

### Package Not Found

```
ModuleNotFoundError: No module named 'llm_eval'
```

**Solution:**
```python
# Manually install
!pip install -e /content/drive/MyDrive/work_MBP_M1/llm-lab/posttrain_llm/llm_eval
```

### Drive Not Mounted

```
FileNotFoundError: [Errno 2] No such file or directory: '/content/drive'
```

**Solution:**
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Project Directory Not Found

```
WARNING: Project directory not found
```

**Solution:**
```bash
# On local Mac, sync the project first
./scripts/sync_to_colab.py
```

### SSH Connection Failed

**Check:**
1. SSH password is correct
2. Hostname in `~/.ssh/config` matches Colab output
3. `cloudflared` is installed locally: `brew install cloudflared`
4. Colab runtime is still running

### Models Re-downloading

```
Downloading model even though I already have it
```

**Solution:**
```python
# Verify cache is set
import os
print(os.environ.get('HF_HOME'))
# Should show: /content/drive/MyDrive/llm_cache

# If not set, run init script again
```

## Tips

### Save Notebook to Drive

Save your Colab notebooks to Drive, not to Colab's default location:

```
File > Save a copy in Drive
Location: /content/drive/MyDrive/work_MBP_M1/llm-lab/notebooks/
```

### GPU Runtime

Make sure you're using a GPU runtime:

```
Runtime > Change runtime type > Hardware accelerator > GPU (T4 or better)
```

### Check GPU

```python
!nvidia-smi
```

### Monitor Disk Usage

```python
!df -h /content/drive
```

### Clear Cache (if needed)

```bash
# Only if you need to free space
!rm -rf /content/drive/MyDrive/llm_cache/hub/*
```

## Environment Variables Set

After running `colab_init.py`, these are set:

```python
HF_HOME=/content/drive/MyDrive/llm_cache
HF_HUB_CACHE=/content/drive/MyDrive/llm_cache
TRANSFORMERS_CACHE=/content/drive/MyDrive/llm_cache
HF_DATASETS_CACHE=/content/drive/MyDrive/llm_cache/datasets
```

## Related Documentation

- **Hybrid Setup Guide**: `dev/colab/Hybrid-Colab-SSH-Development-Setup.md`
- **Sync Script**: `scripts/README_COLAB_SYNC.md`
- **llm_eval Package**: `posttrain_llm/llm_eval/README.md`

---

**Last Updated**: 2025-11-30  
**Version**: 1.0.0
