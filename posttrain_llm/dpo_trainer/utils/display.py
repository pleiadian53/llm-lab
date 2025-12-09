"""Display utilities for DPO training."""

from typing import Any, Dict, Optional


def print_section(title: str, width: int = 60) -> None:
    """Print a section header.
    
    Args:
        title: Section title
        width: Total width of the header
    """
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)
    print()


def print_training_summary(
    metrics: Dict[str, Any],
    config: Optional[Any] = None,
    title: str = "Training Summary",
) -> None:
    """Print a training summary.
    
    Args:
        metrics: Training metrics dictionary
        config: Optional training configuration
        title: Summary title
    """
    print_section(title)
    
    if config:
        print("Configuration:")
        print(f"  Beta: {getattr(config, 'beta', 'N/A')}")
        print(f"  Learning Rate: {getattr(config, 'learning_rate', 'N/A')}")
        print(f"  Epochs: {getattr(config, 'num_train_epochs', 'N/A')}")
        print(f"  Batch Size: {getattr(config, 'per_device_train_batch_size', 'N/A')}")
        print()
    
    print("Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print()


def print_comparison_table(
    results: Dict[str, Dict[str, Any]],
    title: str = "Model Comparison",
) -> None:
    """Print a comparison table for multiple models.
    
    Args:
        results: Dict mapping model names to their results
        title: Table title
    """
    print_section(title)
    
    # Get all metric keys
    all_keys = set()
    for model_results in results.values():
        all_keys.update(model_results.keys())
    
    # Print header
    model_names = list(results.keys())
    header = "Metric".ljust(30) + "".join(name.ljust(20) for name in model_names)
    print(header)
    print("-" * len(header))
    
    # Print rows
    for key in sorted(all_keys):
        row = key.ljust(30)
        for model_name in model_names:
            value = results[model_name].get(key, "N/A")
            if isinstance(value, float):
                row += f"{value:.4f}".ljust(20)
            else:
                row += str(value).ljust(20)
        print(row)
    print()


def format_preference_pair(
    chosen: str,
    rejected: str,
    prompt: Optional[str] = None,
    max_length: int = 100,
) -> str:
    """Format a preference pair for display.
    
    Args:
        chosen: Chosen response
        rejected: Rejected response
        prompt: Optional prompt
        max_length: Maximum length for truncation
    
    Returns:
        Formatted string
    """
    def truncate(text: str) -> str:
        if len(text) > max_length:
            return text[:max_length] + "..."
        return text
    
    lines = []
    if prompt:
        lines.append(f"Prompt: {truncate(prompt)}")
    lines.append(f"✓ Chosen: {truncate(chosen)}")
    lines.append(f"✗ Rejected: {truncate(rejected)}")
    
    return "\n".join(lines)
