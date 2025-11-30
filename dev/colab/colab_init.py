#!/usr/bin/env python3
"""
Colab Hybrid Development Environment Initialization

This script sets up a Google Colab session for hybrid cloud-local development:
1. Mounts Google Drive for persistent storage
2. Configures HuggingFace cache to persist models
3. Sets up SSH tunnel for remote development
4. Navigates to project directory
5. Installs llm_eval package

USAGE IN COLAB:
  Copy this entire file into a Colab code cell and run it.
  Or use the formatted version below with @title magic.

REQUIREMENTS:
  - Google Colab environment
  - Google Drive with work_MBP_M1/llm-lab synced
  - SSH password (change SSH_PASSWORD below)

AFTER RUNNING:
  - Your project is at: /content/drive/MyDrive/work_MBP_M1/llm-lab
  - Models cache at: /content/drive/MyDrive/llm_cache
  - SSH tunnel ready for remote VSCode/terminal access
"""

# @title Hybrid Dev Setup: Drive, Cache, SSH & Project { display-mode: "form" }

import os
import sys
from pathlib import Path

print("=" * 70)
print("🚀 Initializing Hybrid Colab Development Environment")
print("=" * 70)
print()

# =============================================================================
# 1. MOUNT GOOGLE DRIVE
# =============================================================================
print("📂 Step 1/5: Mounting Google Drive...")

from google.colab import drive

if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')
    print("✅ Google Drive mounted successfully")
else:
    print("✅ Google Drive already mounted")

print()

# =============================================================================
# 2. CONFIGURE PERSISTENT LLM CACHE
# =============================================================================
print("💾 Step 2/5: Configuring persistent LLM cache...")

# This forces HuggingFace to save/load models from Drive instead of temp VM
CACHE_DIR = "/content/drive/MyDrive/llm_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Set all HuggingFace cache environment variables
os.environ['HF_HOME'] = CACHE_DIR
os.environ['HF_HUB_CACHE'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = os.path.join(CACHE_DIR, 'datasets')

print(f"✅ HuggingFace cache: {CACHE_DIR}")
print(f"   - Models will persist between sessions")
print(f"   - No need to re-download large models")

print()

# =============================================================================
# 3. NAVIGATE TO PROJECT DIRECTORY
# =============================================================================
print("📁 Step 3/5: Setting up project directory...")

PROJECT_DIR = "/content/drive/MyDrive/work_MBP_M1/llm-lab"

if os.path.exists(PROJECT_DIR):
    os.chdir(PROJECT_DIR)
    print(f"✅ Changed to project directory: {PROJECT_DIR}")
    
    # Add project to Python path
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
        print(f"✅ Added project to Python path")
else:
    print(f"⚠️  WARNING: Project directory not found: {PROJECT_DIR}")
    print(f"   Please sync your project using: ./scripts/sync_to_colab.py")
    print(f"   Continuing with setup anyway...")

print()

# =============================================================================
# 4. INSTALL LLM_EVAL PACKAGE
# =============================================================================
print("📦 Step 4/5: Installing llm_eval package...")

LLM_EVAL_PATH = os.path.join(PROJECT_DIR, "posttrain_llm/llm_eval")

if os.path.exists(LLM_EVAL_PATH):
    # Install in editable mode
    import subprocess
    result = subprocess.run(
        ["pip", "install", "-e", LLM_EVAL_PATH, "--quiet"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"✅ llm_eval package installed from: {LLM_EVAL_PATH}")
        
        # Verify installation
        try:
            import llm_eval
            print(f"✅ Package version: {llm_eval.__version__}")
        except ImportError:
            print("⚠️  Package installed but import failed (may need kernel restart)")
    else:
        print(f"❌ Failed to install llm_eval package")
        print(f"   Error: {result.stderr}")
else:
    print(f"⚠️  llm_eval package not found at: {LLM_EVAL_PATH}")
    print(f"   You can install it manually later with:")
    print(f"   !pip install -e {LLM_EVAL_PATH}")

print()

# =============================================================================
# 5. INSTALL & LAUNCH SSH TUNNEL
# =============================================================================
print("🔗 Step 5/5: Setting up SSH tunnel...")

# Install SSH tools
print("   Installing colab_ssh...")
import subprocess
subprocess.run(["pip", "install", "colab_ssh", "--quiet"], check=False)

from colab_ssh import launch_ssh_cloudflared

# ⚠️ CHANGE THIS PASSWORD for your session
SSH_PASSWORD = "change_me_123"  # TODO: Change this!

print(f"   Launching SSH tunnel with password: '{SSH_PASSWORD}'")
print(f"   ⚠️  Remember to change SSH_PASSWORD in this script!")
print()

launch_ssh_cloudflared(password=SSH_PASSWORD)

print()
print("=" * 70)
print("✅ SETUP COMPLETE!")
print("=" * 70)
print()
print("📍 Current directory:", os.getcwd())
print("🐍 Python path includes:", PROJECT_DIR if PROJECT_DIR in sys.path else "NOT SET")
print("💾 HuggingFace cache:", os.environ.get('HF_HOME', 'NOT SET'))
print()
print("🎯 Quick Start:")
print("   from llm_eval import ServeLLM, evaluate_math_reasoning")
print("   from datasets import load_dataset")
print()
print("👇 COPY THE SSH CONFIG BLOCK ABOVE INTO YOUR LOCAL ~/.ssh/config")
print("=" * 70)
