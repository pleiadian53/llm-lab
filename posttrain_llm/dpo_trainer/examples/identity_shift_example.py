#!/usr/bin/env python
"""Example: Identity Shift Training with DPO.

This example demonstrates how to use DPO to change a model's self-identification,
replicating the functionality from Lesson_5.ipynb.

Usage:
    python -m dpo_trainer.examples.identity_shift_example
"""

import sys
from pathlib import Path

# Add parent to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dpo_trainer import (
    DPOTrainerWrapper,
    DPOTrainingConfig,
    build_identity_shift_dataset,
    load_model_and_tokenizer,
    test_model,
)


def main():
    # Configuration
    MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
    ORIGINAL_NAME = "Qwen"
    NEW_NAME = "Deep Qwen"
    OUTPUT_DIR = "./dpo_identity_shift_output"
    USE_GPU = False
    MAX_SAMPLES = 5  # Small for demo; increase for real training
    
    # Test questions
    questions = [
        "What is your name?",
        "Are you ChatGPT?",
        "Tell me about your name and organization.",
    ]
    
    print("=" * 60)
    print("  DPO Identity Shift Training Example")
    print("=" * 60)
    print(f"\nModel: {MODEL_NAME}")
    print(f"Original Name: {ORIGINAL_NAME}")
    print(f"New Name: {NEW_NAME}")
    print(f"Max Samples: {MAX_SAMPLES}")
    print()
    
    # Step 1: Load model and test before training
    print("Step 1: Loading model and testing before DPO...")
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, use_gpu=USE_GPU)
    
    test_model(
        model, tokenizer, questions,
        title="Before DPO Training",
    )
    
    # Step 2: Build identity shift dataset
    print("\nStep 2: Building identity shift dataset...")
    dpo_dataset = build_identity_shift_dataset(
        model, tokenizer,
        original_name=ORIGINAL_NAME,
        new_name=NEW_NAME,
        max_samples=MAX_SAMPLES,
    )
    print(f"Generated {len(dpo_dataset)} preference pairs")
    
    # Show a sample
    print("\nSample preference pair:")
    sample = dpo_dataset[0]
    print(f"  Chosen: {sample['chosen'][-1]['content'][:100]}...")
    print(f"  Rejected: {sample['rejected'][-1]['content'][:100]}...")
    
    # Clean up model used for generation
    del model
    
    # Step 3: Configure and run DPO training
    print("\nStep 3: Running DPO training...")
    
    training_config = DPOTrainingConfig(
        beta=0.2,
        learning_rate=5e-5,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        logging_steps=2,
        output_dir=OUTPUT_DIR,
    )
    
    trainer = DPOTrainerWrapper(
        model_name=MODEL_NAME,
        train_dataset=dpo_dataset,
        training_config=training_config,
        use_gpu=USE_GPU,
    )
    
    metrics = trainer.train()
    
    # Step 4: Save the model
    print("\nStep 4: Saving model...")
    trainer.save_model(OUTPUT_DIR)
    
    # Step 5: Test after training
    print("\nStep 5: Testing after DPO training...")
    test_model(
        trainer.model, trainer.tokenizer, questions,
        title="After DPO Training",
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"\nModel saved to: {OUTPUT_DIR}")
    print("\nTraining Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
