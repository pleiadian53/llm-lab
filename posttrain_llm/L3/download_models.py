#!/usr/bin/env python3
"""Download models to local directory. 

This script downloads pre-trained models from HuggingFace Hub and saves them locally.
Supports custom model directories, user-specified models, and validation.

Usage:
    # Download default models
    python download_models.py
    
    # Custom directory
    python download_models.py --models-dir /path/to/models
    
    # Download specific models
    python download_models.py --models Qwen/Qwen3-0.6B HuggingFaceTB/SmolLM2-135M
    
    # Dry run (check without downloading)
    python download_models.py --dry-run
    
    # Skip existing models
    python download_models.py --skip-existing
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from huggingface_hub import model_info, HfApi
except ImportError as e:
    print(f"Error: Missing required packages. Please install: pip install transformers huggingface_hub")
    sys.exit(1)

# Default models for Lesson 3
DEFAULT_MODELS = [
    ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-0.6B-Base"),
    ("banghua/Qwen3-0.6B-SFT", "banghua/Qwen3-0.6B-SFT"),
    ("HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-135M"),
]


def validate_repo_id(repo_id: str) -> Tuple[bool, Optional[str]]:
    """Validate if a HuggingFace repo ID exists.
    
    Args:
        repo_id: HuggingFace model repo ID (e.g., 'Qwen/Qwen3-0.6B')
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        api = HfApi()
        info = api.model_info(repo_id)
        return True, None
    except Exception as e:
        return False, str(e)


def model_exists_locally(local_path: str) -> bool:
    """Check if model already exists locally.
    
    Args:
        local_path: Path to local model directory
        
    Returns:
        True if model files exist, False otherwise
    """
    path = Path(local_path)
    if not path.exists():
        return False
    
    # Check for essential model files
    required_files = ['config.json', 'tokenizer_config.json']
    has_model_weights = any([
        (path / 'pytorch_model.bin').exists(),
        (path / 'model.safetensors').exists(),
        any(path.glob('model-*.safetensors')),
        any(path.glob('pytorch_model-*.bin')),
    ])
    
    has_required = all((path / f).exists() for f in required_files)
    return has_required and has_model_weights


def get_model_size(repo_id: str) -> Optional[str]:
    """Get approximate model size from HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace model repo ID
        
    Returns:
        Human-readable size string or None if unavailable
    """
    try:
        info = model_info(repo_id)
        if hasattr(info, 'safetensors') and info.safetensors:
            total_size = info.safetensors.get('total', 0)
        elif hasattr(info, 'siblings'):
            # Sum up file sizes
            total_size = sum(s.size for s in info.siblings if hasattr(s, 'size') and s.size)
        else:
            return None
            
        # Convert to human-readable format
        for unit in ['B', 'KB', 'MB', 'GB']:
            if total_size < 1024.0:
                return f"{total_size:.1f} {unit}"
            total_size /= 1024.0
        return f"{total_size:.1f} TB"
    except:
        return None


def download_model(repo_id: str, local_path: str, skip_existing: bool = False) -> bool:
    """Download model from HuggingFace and save locally.
    
    Args:
        repo_id: HuggingFace model repo ID
        local_path: Local directory to save model
        skip_existing: If True, skip download if model exists locally
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"Model: {repo_id}")
    print(f"Destination: {local_path}")
    
    # Check if model exists locally
    if skip_existing and model_exists_locally(local_path):
        print("  ⏭️  Model already exists locally, skipping...")
        return True
    
    # Get model size
    size = get_model_size(repo_id)
    if size:
        print(f"  Size: ~{size}")
    
    try:
        os.makedirs(local_path, exist_ok=True)
        
        # Download tokenizer
        print("  📥 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        tokenizer.save_pretrained(local_path)
        print("     ✓ Tokenizer saved")
        
        # Download model
        print("  📥 Downloading model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(repo_id)
        model.save_pretrained(local_path)
        print("     ✓ Model saved")
        
        print(f"  ✅ Successfully saved to {local_path}")
        return True
        
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download pre-trained models from HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download default models
  %(prog)s
  
  # Custom directory
  %(prog)s --models-dir /path/to/models
  
  # Download specific models
  %(prog)s --models Qwen/Qwen3-0.6B HuggingFaceTB/SmolLM2-135M
  
  # Dry run
  %(prog)s --dry-run
  
  # Skip existing models
  %(prog)s --skip-existing
        """
    )
    
    parser.add_argument(
        '--models-dir',
        type=str,
        default='./models',
        help='Base directory for saving models (default: ./models)'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        type=str,
        help='Specific model repo IDs to download (e.g., Qwen/Qwen3-0.6B)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate models without downloading'
    )
    
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip models that already exist locally'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate repo IDs without downloading'
    )
    
    parser.add_argument(
        '--list-default',
        action='store_true',
        help='List default models and exit'
    )
    
    return parser.parse_args()


def prepare_models_list(args) -> List[Tuple[str, str]]:
    """Prepare list of models to download.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        List of (repo_id, local_path) tuples
    """
    models_dir = Path(args.models_dir)
    
    if args.models:
        # User-specified models
        models = []
        for repo_id in args.models:
            # Use repo_id as subdirectory name
            local_name = repo_id.replace('/', '_')
            local_path = models_dir / local_name
            models.append((repo_id, str(local_path)))
    else:
        # Default models
        models = [
            (repo_id, str(models_dir / subdir))
            for repo_id, subdir in DEFAULT_MODELS
        ]
    
    return models


def main():
    """Main execution function."""
    args = parse_args()
    
    # List default models and exit
    if args.list_default:
        print("\nDefault models for Lesson 3:")
        print("=" * 70)
        for i, (repo_id, subdir) in enumerate(DEFAULT_MODELS, 1):
            size = get_model_size(repo_id)
            size_str = f" (~{size})" if size else ""
            print(f"{i}. {repo_id}{size_str}")
            print(f"   └─ Local: ./models/{subdir}")
        print()
        return
    
    # Prepare models list
    models = prepare_models_list(args)
    
    # Header
    print("\n" + "=" * 70)
    print("Model Download Script for Lesson 3")
    print("=" * 70)
    print(f"Models directory: {args.models_dir}")
    print(f"Number of models: {len(models)}")
    
    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No downloads will be performed")
    if args.skip_existing:
        print("\n⏭️  Skip existing models: Enabled")
    
    # Validate all models first
    print("\n" + "=" * 70)
    print("Validating model repo IDs...")
    print("=" * 70)
    
    invalid_models = []
    for repo_id, local_path in models:
        print(f"\n  Checking: {repo_id}")
        is_valid, error = validate_repo_id(repo_id)
        
        if is_valid:
            print("    ✅ Valid")
            size = get_model_size(repo_id)
            if size:
                print(f"    📊 Size: ~{size}")
        else:
            print(f"    ❌ Invalid: {error}")
            invalid_models.append((repo_id, error))
    
    if invalid_models:
        print("\n" + "=" * 70)
        print("❌ Validation failed for the following models:")
        print("=" * 70)
        for repo_id, error in invalid_models:
            print(f"  - {repo_id}")
            print(f"    Error: {error}")
        print("\nPlease check the repo IDs and try again.")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("✅ All models validated successfully!")
    print("=" * 70)
    
    # Exit if validate-only mode
    if args.validate_only or args.dry_run:
        print("\n✓ Validation complete. Exiting without download.")
        return
    
    # Download models
    print("\n" + "=" * 70)
    print("Downloading models...")
    print("=" * 70)
    print("\n⏳ This may take several minutes depending on your connection.")
    print("   Models will be cached by HuggingFace for faster subsequent downloads.\n")
    
    successful = []
    failed = []
    skipped = []
    
    for repo_id, local_path in models:
        if args.skip_existing and model_exists_locally(local_path):
            skipped.append(repo_id)
            print(f"\n⏭️  Skipping {repo_id} (already exists)")
            continue
            
        success = download_model(repo_id, local_path, args.skip_existing)
        if success:
            successful.append(repo_id)
        else:
            failed.append(repo_id)
    
    # Summary
    print("\n" + "=" * 70)
    print("Download Summary")
    print("=" * 70)
    print(f"✅ Successful: {len(successful)}")
    if successful:
        for repo_id in successful:
            print(f"   - {repo_id}")
    
    if skipped:
        print(f"\n⏭️  Skipped: {len(skipped)}")
        for repo_id in skipped:
            print(f"   - {repo_id}")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)}")
        for repo_id in failed:
            print(f"   - {repo_id}")
    
    print("\n" + "=" * 70)
    if failed:
        print("⚠️  Some downloads failed. Check errors above.")
        sys.exit(1)
    else:
        print("✅ All downloads complete!")
        print("=" * 70)
        print(f"\nModels saved to: {args.models_dir}")
        print("You can now run the notebook with local model paths.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
