# Using VS Code with Colab Kernel

## Quick Start

The notebook has been updated to work seamlessly with both local and Colab kernels!

### 1. Install VS Code Colab Extension

```bash
# In VS Code:
# 1. Open Extensions (Cmd+Shift+X)
# 2. Search for "Colab"
# 3. Install "Colab" by Google
```

Or install directly: [VS Code Colab Extension](https://marketplace.visualstudio.com/items?itemName=Google.colab)

### 2. Open Notebook in VS Code

```bash
# Open the notebook
code /Users/pleiadian53/work/llm-lab/posttrain_llm/M1/M1_G1_Inspecting_Finetuned_vs_Base_Model.ipynb
```

### 3. Select Colab Kernel

1. Click on the **kernel selector** (top right)
2. Choose **"Colab Kernels"**
3. Select runtime type:
   - **T4 GPU** (free tier - recommended)
   - **A100 GPU** (Colab Pro)
   - **CPU** (not recommended for this notebook)

### 4. Upload Utils Files to Colab

The notebook needs the `utils` folder. You have two options:

#### Option A: Upload Files Directly (Quick)

```python
# Run this in a new cell at the top of the notebook
from google.colab import files
import os

# Create utils directory
os.makedirs('utils', exist_ok=True)

# Upload files
print("Please upload serve_llm.py and utils.py from posttrain_llm/M1/utils/")
uploaded = files.upload()

# Move to utils folder
for filename in uploaded.keys():
    !mv {filename} utils/
    
print("✅ Files uploaded successfully!")
```

#### Option B: Use Google Drive (Persistent)

```python
# Run this in a new cell at the top
from google.colab import drive
drive.mount('/content/drive')

# Copy utils from Drive
import shutil
source = '/content/drive/MyDrive/llm-lab/posttrain_llm/M1/utils'
dest = './utils'

if os.path.exists(source):
    shutil.copytree(source, dest, dirs_exist_ok=True)
    print("✅ Utils copied from Drive!")
else:
    print("❌ Utils not found in Drive. Please upload first.")
```

### 5. Run the Notebook

Now just run the cells! The notebook will:
- ✅ Auto-detect it's running in Colab
- ✅ Download models from HuggingFace automatically
- ✅ Load datasets from HuggingFace
- ✅ Use Colab's GPU

## What Changed in the Notebook?

### Cell 4 (Imports)
```python
# Now detects Colab environment automatically
try:
    import google.colab
    IN_COLAB = True
    # Handles path setup for Colab
except ImportError:
    IN_COLAB = False
    # Uses local paths
```

### Cell 6 (Model Paths)
```python
# Automatically uses HuggingFace model names in Colab
if IN_COLAB:
    BASE_MODEL = "deepseek-ai/deepseek-math-7b-base"
    # Models download automatically
else:
    BASE_MODEL = "/app/models/deepseek-math-7b-base"
    # Uses local paths
```

### Cells 24 & 36 (Datasets)
```python
# Automatically loads from HuggingFace in Colab
if IN_COLAB:
    gsm8k_dataset = load_dataset("gsm8k", "main", split="test")
else:
    gsm8k_dataset = load_from_disk("/app/data/gsm8k")
```

## Memory Management in Colab

### Free Tier (T4 GPU - 16GB VRAM)

Use 8-bit quantization to fit models:

```python
# Modify ServeLLM calls to use quantization
with ServeLLM(SFT_MODEL, quantize="8bit") as llm:
    response = llm.generate_response(prompt)
```

### Colab Pro (A100 GPU - 40GB VRAM)

No quantization needed - use default settings:

```python
# Full precision works fine
with ServeLLM(SFT_MODEL) as llm:
    response = llm.generate_response(prompt)
```

## Caching Models (Optional)

To avoid re-downloading models each session:

```python
# Add this before running the notebook
import os
from google.colab import drive

drive.mount('/content/drive')

# Set cache to Drive
cache_dir = "/content/drive/MyDrive/llm_models"
os.makedirs(cache_dir, exist_ok=True)
os.environ['HF_HOME'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir

print(f"✅ Models will cache to: {cache_dir}")
```

**Note**: Each 7B model is ~14GB. Make sure you have enough Drive space!

## Troubleshooting

### "ModuleNotFoundError: No module named 'utils'"

**Solution**: Upload utils files using Option A or B above

### "CUDA out of memory"

**Solution**: Use quantization
```python
with ServeLLM(model_name, quantize="8bit") as llm:
    ...
```

### "Runtime disconnected"

**Solution**: 
1. Reduce `num_samples` in evaluation functions
2. Save intermediate results
3. Use browser console keep-alive (see COLAB_SETUP.md)

### Models downloading slowly

**Solution**: 
1. Use Colab Pro for faster download speeds
2. Or cache models to Drive (see above)

## Performance Comparison

| Environment | Setup Time | Inference Speed | Cost |
|-------------|------------|-----------------|------|
| **M1 Mac (MPS + 4-bit)** | 0 min | ~5s/prompt | Free |
| **Colab Free (T4 + 8-bit)** | 5-10 min | ~3s/prompt | Free |
| **Colab Pro (A100)** | 5-10 min | ~1s/prompt | $10/mo |

## Best Practices

### 1. Start Small
```python
# Test with small samples first
evaluate_model_correctness(SFT_MODEL, num_samples=5)
```

### 2. Use Quantization on Free Tier
```python
# Always use 8-bit on T4 GPU
with ServeLLM(model, quantize="8bit") as llm:
    ...
```

### 3. Save Results Periodically
```python
# Save results to Drive
import pickle
with open('/content/drive/MyDrive/results.pkl', 'wb') as f:
    pickle.dump(results, f)
```

### 4. Monitor GPU Memory
```python
# Check GPU usage
!nvidia-smi
```

## Complete Workflow Example

```python
# 1. Setup (run once per session)
from google.colab import drive, files
import os

drive.mount('/content/drive')

# Upload utils files
os.makedirs('utils', exist_ok=True)
# ... upload serve_llm.py and utils.py ...

# 2. Configure caching (optional)
os.environ['HF_HOME'] = '/content/drive/MyDrive/llm_models'

# 3. Install packages
!pip install -q transformers datasets accelerate bitsandbytes

# 4. Run notebook cells
# The notebook will auto-detect Colab and work correctly!

# 5. Use quantization on free tier
with ServeLLM(SFT_MODEL, quantize="8bit") as llm:
    response = llm.generate_response(prompt)
```

## Advantages of VS Code + Colab

✅ **Familiar Interface**: Use VS Code shortcuts and features  
✅ **Better IntelliSense**: Code completion works better  
✅ **Multiple Files**: Edit utils files alongside notebook  
✅ **Git Integration**: Commit changes easily  
✅ **Free GPU**: Access Colab's T4 GPU for free  
✅ **No Browser Tab**: No need to keep browser open  

## Next Steps

1. ✅ Install VS Code Colab extension
2. ✅ Open notebook in VS Code
3. ✅ Connect to Colab kernel (T4 GPU)
4. ✅ Upload utils files
5. ✅ Run cells with 8-bit quantization
6. ✅ Enjoy coding!

For more details on Colab setup, see [COLAB_SETUP.md](COLAB_SETUP.md)
