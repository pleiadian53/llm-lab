#!/usr/bin/env python3
"""
Quick test to verify model download is working.
Run this to check if models are downloading properly without UI errors.
"""

import os
import sys

# Disable progress bars to avoid UI issues
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

# Add utils to path
sys.path.insert(0, os.path.dirname(__file__))

from utils.serve_llm import ServeLLM

def test_model_download(model_id):
    """Test if a model can be downloaded and loaded."""
    print(f"\n{'='*70}")
    print(f"Testing: {model_id}")
    print(f"{'='*70}\n")
    
    try:
        with ServeLLM(model_id) as llm:
            print("\n✅ Model loaded successfully!")
            
            # Test a simple generation
            test_prompt = "What is 2+2?"
            print(f"\nTest prompt: {test_prompt}")
            response = llm.generate_response(test_prompt, max_tokens=50)
            print(f"Response: {response}\n")
            
            return True
            
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test with a small model first
    print("Testing model download and inference...")
    print(f"Cache location: {os.environ.get('HF_HOME', 'Not set')}")
    
    # Use the base model
    model_id = "deepseek-ai/deepseek-math-7b-base"
    
    success = test_model_download(model_id)
    
    if success:
        print("="*70)
        print("✅ TEST PASSED - Model download and inference working!")
        print("="*70)
    else:
        print("="*70)
        print("❌ TEST FAILED - Check errors above")
        print("="*70)
