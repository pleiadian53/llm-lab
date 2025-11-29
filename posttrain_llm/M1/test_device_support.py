#!/usr/bin/env python3
"""
Test script to verify device support and quantization options.
Run this to check what's available on your system.
"""

import torch
import sys
sys.path.append('..')

def check_device_support():
    """Check which devices are available on this system."""
    print("=" * 60)
    print("DEVICE SUPPORT CHECK")
    print("=" * 60)
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"\n✓ CUDA (NVIDIA GPU): {cuda_available}")
    if cuda_available:
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
        print(f"  - Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"  - CUDA Version: {torch.version.cuda}")
    
    # Check MPS (Apple Silicon)
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    print(f"\n✓ MPS (Apple Silicon): {mps_available}")
    if mps_available:
        print(f"  - PyTorch version: {torch.__version__}")
        print(f"  - MPS backend built: {torch.backends.mps.is_built()}")
    
    # CPU always available
    print(f"\n✓ CPU: True")
    
    # Check quantization support
    print(f"\n{'=' * 60}")
    print("QUANTIZATION SUPPORT CHECK")
    print("=" * 60)
    
    try:
        import bitsandbytes
        print(f"\n✓ bitsandbytes: Installed (version {bitsandbytes.__version__})")
        print("  - 4-bit quantization: Available")
        print("  - 8-bit quantization: Available")
    except ImportError:
        print(f"\n✗ bitsandbytes: Not installed")
        print("  - Install with: pip install bitsandbytes")
    
    # Recommended configuration
    print(f"\n{'=' * 60}")
    print("RECOMMENDED CONFIGURATION FOR YOUR SYSTEM")
    print("=" * 60)
    
    if cuda_available:
        print("\n✅ Use CUDA (default - best performance)")
        print("   with ServeLLM(model_name) as llm:")
        print("       # Auto-detects CUDA")
    elif mps_available:
        print("\n✅ Use MPS (Apple Silicon GPU)")
        print("   with ServeLLM(model_name, device='mps') as llm:")
        print("       # Or device='auto' for auto-detection")
        try:
            import bitsandbytes
            print("\n💡 For 16GB RAM, consider 4-bit quantization:")
            print("   with ServeLLM(model_name, device='mps', quantize='4bit') as llm:")
        except ImportError:
            print("\n💡 Install bitsandbytes for quantization support")
    else:
        print("\n⚠️  CPU only (slower)")
        print("   with ServeLLM(model_name, device='cpu') as llm:")
        try:
            import bitsandbytes
            print("\n💡 Use 4-bit quantization to reduce memory:")
            print("   with ServeLLM(model_name, device='cpu', quantize='4bit') as llm:")
        except ImportError:
            print("\n💡 Install bitsandbytes for quantization support")
    
    print(f"\n{'=' * 60}\n")


def test_model_loading():
    """Test loading a small model to verify everything works."""
    print("=" * 60)
    print("MODEL LOADING TEST")
    print("=" * 60)
    
    # Use a very small model for testing
    test_model = "gpt2"  # ~500MB, good for testing
    
    print(f"\nTesting with small model: {test_model}")
    print("This will verify that model loading works on your system.\n")
    
    try:
        from utils.utils import ServeLLM
        
        print("Loading model...")
        with ServeLLM(test_model) as llm:
            print("✅ Model loaded successfully!")
            
            # Test inference
            print("\nTesting inference...")
            response = llm.generate_response("Hello, world!", max_tokens=10)
            print(f"✅ Inference works! Response: {response[:50]}...")
            
        print("\n✅ All tests passed!")
        print("\nYou're ready to use the notebook!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease check:")
        print("1. transformers is installed: pip install transformers")
        print("2. torch is installed: pip install torch")
        print("3. You're in the correct directory")
        
    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test device support and model loading")
    parser.add_argument("--test-model", action="store_true", 
                       help="Test loading a small model (requires internet)")
    args = parser.parse_args()
    
    check_device_support()
    
    if args.test_model:
        test_model_loading()
    else:
        print("💡 Run with --test-model to test actual model loading")
        print("   python test_device_support.py --test-model\n")
