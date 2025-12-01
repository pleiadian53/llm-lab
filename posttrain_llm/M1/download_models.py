#!/usr/bin/env python3
"""
Download models to a specific local directory for offline use.

This script downloads HuggingFace models to a custom directory structure
instead of relying on the default cache.
"""

import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

# Define where you want to save models
MODELS_DIR = "/content/drive/MyDrive/work_MBP_M1/llm-lab/models"

# Models to download
MODELS = {
    "deepseek-math-7b-base": "deepseek-ai/deepseek-math-7b-base",
    "deepseek-math-7b-instruct": "deepseek-ai/deepseek-math-7b-instruct",
    "deepseek-math-7b-rl": "deepseek-ai/deepseek-math-7b-rl",
    "Llama-Guard-3-8B": "meta-llama/Llama-Guard-3-8B",
}

def download_model(model_id: str, save_path: str):
    """
    Download a model and tokenizer to a specific directory.
    
    Args:
        model_id: HuggingFace model identifier (e.g., "deepseek-ai/deepseek-math-7b-base")
        save_path: Local directory path to save the model
    """
    print(f"\n{'='*70}")
    print(f"Downloading: {model_id}")
    print(f"Save to: {save_path}")
    print(f"{'='*70}")
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    try:
        # Download tokenizer
        print("📥 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.save_pretrained(save_path)
        print("✅ Tokenizer saved")
        
        # Download model
        print("📥 Downloading model (this may take several minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype="auto",
            low_cpu_mem_usage=True
        )
        model.save_pretrained(save_path)
        print("✅ Model saved")
        
        print(f"✅ Successfully downloaded {model_id}")
        
    except Exception as e:
        print(f"❌ Error downloading {model_id}: {e}")
        return False
    
    return True

def main():
    print("="*70)
    print("🚀 Model Download Script")
    print("="*70)
    print(f"\nModels will be saved to: {MODELS_DIR}")
    print(f"Total models to download: {len(MODELS)}")
    print("\n⚠️  WARNING: Each model is ~7-8GB. This will take time and space!")
    print("="*70)
    
    # Create base directory
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    success_count = 0
    
    for local_name, model_id in MODELS.items():
        save_path = os.path.join(MODELS_DIR, local_name)
        
        # Check if already exists
        if os.path.exists(save_path) and os.listdir(save_path):
            print(f"\n⏭️  Skipping {local_name} (already exists at {save_path})")
            success_count += 1
            continue
        
        # Download
        if download_model(model_id, save_path):
            success_count += 1
    
    print("\n" + "="*70)
    print("📊 DOWNLOAD SUMMARY")
    print("="*70)
    print(f"✅ Successfully downloaded: {success_count}/{len(MODELS)}")
    print(f"📁 Models location: {MODELS_DIR}")
    print("\n🎯 To use these models, update your notebook:")
    print(f'   BASE_MODEL = "{MODELS_DIR}/deepseek-math-7b-base"')
    print(f'   SFT_MODEL = "{MODELS_DIR}/deepseek-math-7b-instruct"')
    print(f'   RL_MODEL = "{MODELS_DIR}/deepseek-math-7b-rl"')
    print(f'   llama_guard = "{MODELS_DIR}/Llama-Guard-3-8B"')
    print("="*70)

if __name__ == "__main__":
    main()
