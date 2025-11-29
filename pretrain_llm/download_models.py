#!/usr/bin/env python3
"""
Driver script for downloading pretrained models.

This script provides a convenient way to download models from HuggingFace Hub
to local storage for offline use or faster loading in experiments.

Usage:
    python download_models.py --model upstage/TinySolar-248m-4k --output ./models/TinySolar-248m-4k
    python download_models.py --model gpt2 --output ./models/gpt2 --device cpu
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_lab.pretrain_llm.model_loader import download_model_manually, load_model_with_fallback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download pretrained models from HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model identifier (e.g., 'upstage/TinySolar-248m-4k', 'gpt2')",
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Local directory to save the model",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for loading (default: auto)",
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for model weights (default: bfloat16)",
    )
    
    parser.add_argument(
        "--no-tokenizer",
        action="store_true",
        help="Skip downloading tokenizer",
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the download by loading the model",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Map dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]
    
    LOGGER.info("=" * 80)
    LOGGER.info("Model Download Configuration")
    LOGGER.info("=" * 80)
    LOGGER.info("Model: %s", args.model)
    LOGGER.info("Output Directory: %s", args.output)
    LOGGER.info("Device: %s", args.device)
    LOGGER.info("Data Type: %s", args.dtype)
    LOGGER.info("Include Tokenizer: %s", not args.no_tokenizer)
    LOGGER.info("=" * 80)
    
    try:
        # Download the model
        download_model_manually(
            model_name=args.model,
            output_dir=args.output,
            include_tokenizer=not args.no_tokenizer,
        )
        
        # Verify if requested
        if args.verify:
            LOGGER.info("Verifying download by loading model...")
            model, tokenizer = load_model_with_fallback(
                model_name=str(args.output),
                device_map=args.device,
                torch_dtype=torch_dtype,
            )
            LOGGER.info("✓ Model loaded successfully!")
            LOGGER.info("  - Model type: %s", type(model).__name__)
            LOGGER.info("  - Tokenizer type: %s", type(tokenizer).__name__)
            LOGGER.info("  - Vocab size: %d", len(tokenizer))
            
            # Test generation
            prompt = "Hello, I am"
            inputs = tokenizer(prompt, return_tensors="pt")
            LOGGER.info("Testing generation with prompt: '%s'", prompt)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=10)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            LOGGER.info("Generated: %s", generated_text)
        
        LOGGER.info("=" * 80)
        LOGGER.info("✓ Download completed successfully!")
        LOGGER.info("=" * 80)
        
    except Exception as e:
        LOGGER.error("Failed to download model: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
