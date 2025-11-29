# Google Colab Setup Guide

## 🆕 NEW: VS Code Extension (Recommended!)

**You can now connect to Colab directly from VS Code!**

### Setup VS Code Extension

1. **Install the extension**:
   - Open VS Code
   - Search for "Colab" in Extensions
   - Install the official Google Colab extension
   - [Extension Link](https://marketplace.visualstudio.com/items?itemName=Google.colab)

2. **Connect to Colab runtime**:
   - Open your notebook in VS Code
   - Click "Select Kernel" → "Colab Kernels"
   - Choose runtime type (GPU/CPU)
   - Start coding with Colab's resources directly in VS Code!

**Benefits**:
- ✅ Use VS Code's familiar interface
- ✅ Access Colab's free GPU
- ✅ Better code completion and debugging
- ✅ Seamless integration with your local files

**Repository**: [github.com/googlecolab/colabtools](https://github.com/googlecolab/colabtools)

---

## Alternative: Traditional Colab Web Interface

If you prefer the traditional Colab web interface, follow these steps:

## Quick Start: Running the Notebook on Colab

### Step 1: Upload to Google Drive

1. **Compress your project folder**:
   ```bash
   cd /Users/pleiadian53/work/llm-lab
   zip -r llm-lab.zip posttrain_llm/M1/
   ```

2. **Upload to Google Drive**:
   - Upload `llm-lab.zip` to your Google Drive
   - Or use Google Drive desktop app to sync the folder

### Step 2: Create Colab Notebook

Create a new Colab notebook with this setup code:

```python
# ============================================
# COLAB SETUP CELL - Run this first!
# ============================================

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Navigate to your project
import os
os.chdir('/content/drive/MyDrive/llm-lab/posttrain_llm/M1')

# Verify GPU is available
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Install required packages (if needed)
!pip install -q transformers datasets accelerate bitsandbytes

print("✅ Setup complete!")
```

### Step 3: Run the Notebook

After setup, you can run the notebook cells as normal. The code will automatically detect and use the Colab GPU.

## Alternative: Clone from GitHub

If you push to GitHub first:

```python
# Clone your repository
!git clone https://github.com/pleiadian53/llm-lab.git
%cd llm-lab/posttrain_llm/M1

# Install dependencies
!pip install -q transformers datasets accelerate bitsandbytes

# Verify GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Colab-Specific Optimizations

### 1. Enable GPU Runtime

1. Click **Runtime** → **Change runtime type**
2. Select **T4 GPU** (free tier) or **A100** (Colab Pro)
3. Click **Save**

### 2. Memory Management for Free Tier

Colab free tier has limited RAM (~12GB). Use quantization:

```python
# In your notebook cells, modify model loading:
with ServeLLM(model_name, quantize="8bit") as llm:
    response = llm.generate_response(prompt)
```

### 3. Prevent Disconnection

Colab disconnects after ~90 minutes of inactivity. To prevent:

```javascript
// Run this in browser console (F12)
function KeepClicking(){
   console.log("Clicking");
   document.querySelector("colab-connect-button").click()
}
setInterval(KeepClicking, 60000)
```

## Model Storage Options

### Check Your Google Drive Quota First! 📊

Before deciding on caching strategy, check your available space:

```python
# ============================================
# CHECK GOOGLE DRIVE QUOTA
# ============================================
from google.colab import drive
drive.mount('/content/drive')

# Method 1: Quick check of current usage
print("📁 Current Drive Usage:")
!du -sh /content/drive/MyDrive/

# Method 2: Detailed quota information (recommended)
try:
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
    from google.colab import auth
    from oauth2client.client import GoogleCredentials
    
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive_api = GoogleDrive(gauth)
    
    about = drive_api.GetAbout()
    
    total_gb = int(about['quotaBytesTotal']) / (1024**3)
    used_gb = int(about['quotaBytesUsed']) / (1024**3)
    free_gb = total_gb - used_gb
    
    print(f"\n📊 Google Drive Storage Quota:")
    print(f"  Total:     {total_gb:.2f} GB")
    print(f"  Used:      {used_gb:.2f} GB")
    print(f"  Available: {free_gb:.2f} GB")
    print(f"  Usage:     {(used_gb/total_gb)*100:.1f}%")
    
    # Model storage requirements
    print(f"\n📦 Model Storage Requirements:")
    print(f"  DeepSeek Math 7B (each): ~14 GB")
    print(f"  Total for 3 models:      ~42 GB")
    print(f"  Llama Guard 8B:          ~16 GB")
    print(f"  Grand Total:             ~58 GB")
    
    # Recommendation
    print(f"\n💡 Recommendation:")
    if free_gb >= 60:
        print(f"  ✅ You have enough space to cache all models!")
    elif free_gb >= 20:
        print(f"  ⚠️  Cache only the most-used model (SFT: ~14GB)")
    else:
        print(f"  ❌ Not enough space - use Option 1 (re-download each session)")
    
except ImportError:
    print("\n⚠️  Installing PyDrive for detailed quota check...")
    !pip install -q PyDrive
    print("Please re-run this cell after installation")
```

**Google Drive Storage Tiers**:
- **Free (15 GB)**: ❌ Not enough for model caching
- **Google One 100GB ($1.99/mo)**: ✅ Can cache 1-2 models  
- **Google One 200GB ($2.99/mo)**: ✅ Can cache all models
- **Google Workspace (30GB+)**: ✅ Sufficient

---

### Option 1: Download Each Session (No Caching)

**Best for**: Free tier users or those with limited Drive space

Models download from HuggingFace each session:

```python
# Models auto-download from HuggingFace (uses Colab's local disk)
BASE_MODEL = "deepseek-ai/deepseek-math-7b-base"
SFT_MODEL = "deepseek-ai/deepseek-math-7b-instruct"
RL_MODEL = "deepseek-ai/deepseek-math-7b-rl"

# Use normally - downloads to Colab's temporary storage
with ServeLLM(SFT_MODEL) as llm:
    response = llm.generate_response(prompt)
```

**Pros**: 
- No Drive storage used
- Always get latest model versions
- Simple setup

**Cons**: 
- ~5-10 min download time per session
- Need internet connection

---

### Option 2: Smart Caching to Google Drive

**Best for**: Users with 50GB+ free Drive space who work across multiple sessions

#### Step 1: Set Up HuggingFace Cache

```python
# ============================================
# SMART CACHING SETUP
# ============================================
import os
from google.colab import drive

# Mount Drive
drive.mount('/content/drive')

# Set cache directory
cache_dir = "/content/drive/MyDrive/llm_models"
os.makedirs(cache_dir, exist_ok=True)

# Configure HuggingFace to use Drive cache
os.environ['HF_HOME'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_DATASETS_CACHE'] = cache_dir

print(f"✅ HuggingFace cache set to: {cache_dir}")
print(f"📦 Models will be saved here and reused in future sessions")
```

#### Step 2: Use Models (They Auto-Cache)

```python
# First time: Downloads to Drive (~14GB, takes 5-10 min)
# Subsequent times: Loads from Drive (much faster - ~1 min)

with ServeLLM(SFT_MODEL) as llm:
    response = llm.generate_response(prompt)

# The model is now cached in your Drive!
# Next session: Just set the environment variables again and it will load from cache
```

#### Step 3: Selective Caching (Recommended)

If you don't have space for all models, cache only the most-used one:

```python
# Cache only SFT model (most frequently used in notebook)
CACHE_MODELS = ["deepseek-ai/deepseek-math-7b-instruct"]

# For other models, use Colab's local storage
import tempfile

def get_model_with_smart_cache(model_name):
    """Use Drive cache for selected models, local cache for others."""
    if model_name in CACHE_MODELS:
        # Use Drive cache (persists across sessions)
        os.environ['HF_HOME'] = "/content/drive/MyDrive/llm_models"
        print(f"📦 Using Drive cache for: {model_name}")
    else:
        # Use local cache (faster, but cleared after session)
        os.environ['HF_HOME'] = "/tmp/hf_cache"
        print(f"⚡ Using local cache for: {model_name}")
    
    return model_name

# Usage
model_to_use = get_model_with_smart_cache(SFT_MODEL)
with ServeLLM(model_to_use) as llm:
    response = llm.generate_response(prompt)
```

**Pros**: 
- Fast loading in subsequent sessions (~1 min vs ~10 min)
- Work across multiple sessions efficiently
- Selective caching saves Drive space

**Cons**: 
- Uses Drive storage (~14GB per model)
- Initial download still takes time
- Need to manage Drive quota

## Complete Colab Notebook Template

Here's a complete starter notebook:

```python
# ============================================
# CELL 1: Setup
# ============================================
from google.colab import drive
drive.mount('/content/drive')

!pip install -q transformers datasets accelerate bitsandbytes tqdm pandas

import torch
print(f"✅ CUDA: {torch.cuda.is_available()}")
print(f"✅ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# ============================================
# CELL 2: Clone/Load Project
# ============================================
# Option A: From GitHub
!git clone https://github.com/pleiadian53/llm-lab.git
%cd llm-lab/posttrain_llm/M1

# Option B: From Drive
# %cd /content/drive/MyDrive/llm-lab/posttrain_llm/M1

# ============================================
# CELL 3: Import and Test
# ============================================
import sys
sys.path.append('..')
from utils.utils import ServeLLM

# Test with a small model first
test_model = "deepseek-ai/deepseek-math-7b-instruct"

with ServeLLM(test_model, quantize="8bit") as llm:
    response = llm.generate_response("What is 2+2?", max_tokens=50)
    print(f"Response: {response}")

print("✅ Everything working!")

# ============================================
# CELL 4+: Run notebook cells
# ============================================
# Now run the rest of your notebook...
```

## Resource Limits

### Colab Free Tier
- **GPU**: T4 (16GB VRAM)
- **RAM**: ~12GB
- **Disk**: ~100GB
- **Session**: 12 hours max
- **Idle timeout**: 90 minutes

### Colab Pro ($10/month)
- **GPU**: T4, P100, or V100
- **RAM**: ~25GB
- **Session**: 24 hours max
- **Idle timeout**: Extended

### Colab Pro+ ($50/month)
- **GPU**: A100 (40GB VRAM)
- **RAM**: ~50GB
- **Session**: Extended
- **Background execution**: Yes

## Recommended Settings for This Notebook

```python
# For Colab Free (T4 GPU, 12GB RAM)
with ServeLLM(model_name, quantize="8bit") as llm:
    response = llm.generate_response(prompt, max_tokens=512)

# For Colab Pro (V100 GPU, 25GB RAM)
with ServeLLM(model_name) as llm:  # No quantization needed
    response = llm.generate_response(prompt, max_tokens=512)

# For Colab Pro+ (A100 GPU, 50GB RAM)
with ServeLLM(model_name) as llm:  # Full speed!
    response = llm.generate_response(prompt, max_tokens=1024)
```

## Troubleshooting

### "CUDA out of memory"
**Solution**: Use quantization
```python
with ServeLLM(model_name, quantize="8bit") as llm:
    ...
```

### "Runtime disconnected"
**Solution**: 
1. Reduce `num_samples` in evaluation functions
2. Add periodic checkpoints
3. Use the browser console trick above

### "Models downloading slowly"
**Solution**: Use Colab Pro or cache models in Google Drive

## Caching Decision Flowchart

```
Check Drive Space
      ↓
Free Space >= 60GB?
  ├─ YES → Cache all models (Option 2)
  └─ NO
      ↓
Free Space >= 20GB?
  ├─ YES → Cache SFT model only (Option 2 - Selective)
  └─ NO → Re-download each session (Option 1)
```

## Complete Setup Example

Here's a complete setup cell combining quota check and smart caching:

```python
# ============================================
# COMPLETE COLAB SETUP WITH SMART CACHING
# ============================================

from google.colab import drive
import os

# 1. Mount Drive
drive.mount('/content/drive')

# 2. Check quota (optional but recommended)
!du -sh /content/drive/MyDrive/

# 3. Decide on caching strategy
USE_DRIVE_CACHE = True  # Set to False if you have limited space
CACHE_ONLY_SFT = True   # Set to False to cache all models

# 4. Configure cache
if USE_DRIVE_CACHE:
    cache_dir = "/content/drive/MyDrive/llm_models"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['HF_HOME'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    print(f"✅ Using Drive cache: {cache_dir}")
else:
    print(f"✅ Using local cache (cleared after session)")

# 5. Install packages
!pip install -q transformers datasets accelerate bitsandbytes

# 6. Verify GPU
import torch
print(f"\n🎮 GPU Status:")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

print(f"\n✅ Setup complete! Ready to run notebook.")
```

## VS Code Extension Workflow (Recommended)

If using the VS Code extension:

1. **Open your notebook in VS Code**
2. **Install Colab extension** from VS Code marketplace
3. **Select Colab kernel**: Click kernel selector → "Colab Kernels" → Choose GPU
4. **Run setup cell** (the complete setup example above)
5. **Code normally** - you're now using Colab's GPU from VS Code!

**Benefits over web interface**:
- Better IntelliSense and code completion
- Familiar VS Code shortcuts
- Integrated terminal
- Git integration
- Multiple files open simultaneously

## Next Steps

### For VS Code Users:
1. ✅ Install Colab extension in VS Code
2. ✅ Open notebook in VS Code
3. ✅ Connect to Colab runtime
4. ✅ Run setup cell with caching decision
5. ✅ Start experimenting!

### For Web Interface Users:
1. ✅ Enable GPU runtime
2. ✅ Run setup cell with quota check
3. ✅ Decide on caching strategy
4. ✅ Test with small example
5. ✅ Run full notebook

Happy experimenting! 🚀
