#!/usr/bin/env python3
"""Test script to verify all dependencies are compatible."""

import sys

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ torch: {torch.__version__}")
        
        import numpy as np
        print(f"✓ numpy: {np.__version__}")
        
        import pandas as pd
        print(f"✓ pandas: {pd.__version__}")
        
        import transformers
        print(f"✓ transformers: {transformers.__version__}")
        
        import datasets
        print(f"✓ datasets: {datasets.__version__}")
        
        import trl
        print(f"✓ trl: {trl.__version__}")
        
        import accelerate
        print(f"✓ accelerate: {accelerate.__version__}")
        
        import tabulate
        print(f"✓ tabulate: {tabulate.__version__}")
        
        import jinja2
        print(f"✓ jinja2: {jinja2.__version__}")
        
        import huggingface_hub
        print(f"✓ huggingface_hub: {huggingface_hub.__version__}")
        
        print("\n✅ All imports successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        return False


def test_trl_functionality():
    """Test that TRL components can be imported."""
    print("\nTesting TRL functionality...")
    
    try:
        from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
        print("✓ SFTTrainer imported")
        print("✓ DataCollatorForCompletionOnlyLM imported")
        print("✓ SFTConfig imported")
        
        print("\n✅ TRL functionality test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ TRL functionality test failed: {e}")
        return False


def test_transformers_compatibility():
    """Test transformers compatibility with torch."""
    print("\nTesting transformers-torch compatibility...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print("✓ AutoTokenizer imported")
        print("✓ AutoModelForCausalLM imported")
        
        print("\n✅ Transformers compatibility test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Transformers compatibility test failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("Dependency Compatibility Test")
    print("=" * 70)
    
    results = []
    results.append(test_imports())
    results.append(test_trl_functionality())
    results.append(test_transformers_compatibility())
    
    print("\n" + "=" * 70)
    if all(results):
        print("✅ ALL TESTS PASSED - Dependencies are compatible!")
        print("=" * 70)
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED - Check compatibility issues above")
        print("=" * 70)
        sys.exit(1)
