#!/usr/bin/env python3
"""
Evaluation script for Online RL trained models.

This script evaluates a trained model on math datasets and reports accuracy.

Usage:
    # Evaluate a trained model
    python evaluate_model.py --model ./outputs/online_rl_math
    
    # Evaluate a HuggingFace model
    python evaluate_model.py --model Qwen/Qwen2.5-0.5B-Instruct --num-samples 10
    
    # Compare base vs trained model
    python evaluate_model.py --model Qwen/Qwen2.5-0.5B-Instruct --compare ./outputs/trained
"""

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from posttrain_llm.online_rl_trainer import (
    MathRewardFunction,
    GSM8KLoader,
    ModelEvaluator,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on math problems",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Model name or path to evaluate"
    )
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        help="Second model to compare against"
    )
    
    # Data arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        help="Dataset name"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=5,
        help="Number of samples to evaluate"
    )
    
    # Generation arguments
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    
    # Hardware arguments
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for inference"
    )
    
    # Output arguments
    parser.add_argument(
        "--show-examples",
        type=int,
        default=3,
        help="Number of examples to display"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Save results to JSON file"
    )
    
    return parser.parse_args()


def evaluate_model(
    model_path: str,
    dataset,
    reward_fn,
    use_gpu: bool,
    max_tokens: int,
    temperature: float,
    show_examples: int,
    model_name: str = "Model"
):
    """Evaluate a single model."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Path: {model_path}")
    print(f"{'='*60}")
    
    evaluator = ModelEvaluator(
        model_name_or_path=model_path,
        use_gpu=use_gpu,
        max_new_tokens=max_tokens,
        temperature=temperature,
    )
    
    results = evaluator.evaluate(
        dataset=dataset,
        reward_fn=reward_fn,
        show_examples=show_examples,
        verbose=True,
    )
    
    return results


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    
    # Load dataset
    print("\n📚 Loading dataset...")
    loader = GSM8KLoader()
    eval_dataset = loader.load(args.split, num_samples=args.num_samples)
    print(f"Loaded {len(eval_dataset)} evaluation examples")
    
    # Create reward function
    reward_fn = MathRewardFunction(normalize_numbers=True)
    
    # Evaluate main model
    results1 = evaluate_model(
        model_path=args.model,
        dataset=eval_dataset,
        reward_fn=reward_fn,
        use_gpu=args.use_gpu,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        show_examples=args.show_examples,
        model_name="Model 1"
    )
    
    # Evaluate comparison model if provided
    results2 = None
    if args.compare:
        results2 = evaluate_model(
            model_path=args.compare,
            dataset=eval_dataset,
            reward_fn=reward_fn,
            use_gpu=args.use_gpu,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            show_examples=args.show_examples,
            model_name="Model 2 (Comparison)"
        )
    
    # Print comparison
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nModel 1 ({args.model}):")
    print(f"  Accuracy: {results1['accuracy']:.2%}")
    
    if results2:
        print(f"\nModel 2 ({args.compare}):")
        print(f"  Accuracy: {results2['accuracy']:.2%}")
        
        diff = results2['accuracy'] - results1['accuracy']
        print(f"\nDifference: {diff:+.2%}")
        if diff > 0:
            print("  → Model 2 is better")
        elif diff < 0:
            print("  → Model 1 is better")
        else:
            print("  → Models perform equally")
    
    # Save results if requested
    if args.output_file:
        import json
        output = {
            "model1": {
                "path": args.model,
                "accuracy": results1["accuracy"],
                "num_samples": results1["num_samples"],
            }
        }
        if results2:
            output["model2"] = {
                "path": args.compare,
                "accuracy": results2["accuracy"],
                "num_samples": results2["num_samples"],
            }
        
        with open(args.output_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()
