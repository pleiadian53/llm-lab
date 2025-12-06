"""Display utilities for SFT Trainer."""

from typing import Dict, Optional


def print_section(title: str, char: str = "=", width: int = 60) -> None:
    """Print a section header.
    
    Args:
        title: Section title
        char: Character for the border
        width: Width of the header
    """
    border = char * width
    print(f"\n{border}")
    print(f" {title}")
    print(f"{border}\n")


def print_training_summary(
    metrics: Dict[str, float],
    peft_method: Optional[str] = None,
    model_name: Optional[str] = None,
) -> None:
    """Print a training summary.
    
    Args:
        metrics: Training metrics dictionary
        peft_method: PEFT method used (if any)
        model_name: Model name
    """
    print_section("Training Summary")
    
    if model_name:
        print(f"Model: {model_name}")
    
    if peft_method:
        print(f"PEFT Method: {peft_method}")
    
    print("\nMetrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def print_model_info(model, tokenizer) -> None:
    """Print model and tokenizer information.
    
    Args:
        model: The model
        tokenizer: The tokenizer
    """
    print_section("Model Information")
    
    # Model info
    config = model.config
    print(f"Model Type: {getattr(config, 'model_type', 'Unknown')}")
    print(f"Hidden Size: {getattr(config, 'hidden_size', 'Unknown')}")
    print(f"Num Layers: {getattr(config, 'num_hidden_layers', 'Unknown')}")
    print(f"Num Attention Heads: {getattr(config, 'num_attention_heads', 'Unknown')}")
    print(f"Vocab Size: {getattr(config, 'vocab_size', 'Unknown')}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    # Tokenizer info
    print(f"\nTokenizer Vocab Size: {len(tokenizer)}")
    print(f"Pad Token: {tokenizer.pad_token}")
    print(f"EOS Token: {tokenizer.eos_token}")
    print(f"Has Chat Template: {tokenizer.chat_template is not None}")


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"
