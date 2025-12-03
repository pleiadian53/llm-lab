#!/usr/bin/env python3
"""Download models to local directory for Lesson 3."""

from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def download_model(repo_id, local_path):
    """Download model from HuggingFace and save locally."""
    print(f"\nDownloading {repo_id}...")
    print(f"Saving to: {local_path}")
    
    os.makedirs(local_path, exist_ok=True)
    
    # Download tokenizer
    print("  Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    tokenizer.save_pretrained(local_path)
    
    # Download model
    print("  Downloading model...")
    model = AutoModelForCausalLM.from_pretrained(repo_id)
    model.save_pretrained(local_path)
    
    print(f"  ✓ Saved to {local_path}")

if __name__ == "__main__":
    print("=" * 70)
    print("Downloading models for Lesson 3")
    print("=" * 70)
    print("\nThis will download ~2-3GB of models. First time only.")
    print("Subsequent runs will use cached versions.\n")
    
    # Models to download
    models = [
        ("Qwen/Qwen2.5-0.5B", "./models/Qwen/Qwen3-0.6B-Base"),
        ("Qwen/Qwen2.5-0.5B-Instruct", "./models/banghua/Qwen3-0.6B-SFT"),
        ("HuggingFaceTB/SmolLM2-135M", "./models/HuggingFaceTB/SmolLM2-135M"),
    ]
    
    for repo_id, local_path in models:
        try:
            download_model(repo_id, local_path)
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    print("\n" + "=" * 70)
    print("✅ Download complete!")
    print("=" * 70)
    print("\nYou can now run the notebook with local model paths.")
