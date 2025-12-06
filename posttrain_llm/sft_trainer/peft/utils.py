"""PEFT utility functions."""

from typing import List, Optional

from transformers import PreTrainedModel


# Common target modules for different model architectures
TARGET_MODULES_MAP = {
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "phi": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
    "gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gpt2": ["c_attn", "c_proj", "c_fc"],
    "gpt_neox": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "bloom": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "falcon": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "opt": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    "default": ["q_proj", "k_proj", "v_proj", "o_proj"],
}


def get_target_modules(
    model: Optional[PreTrainedModel] = None,
    model_type: Optional[str] = None,
    include_mlp: bool = True,
) -> List[str]:
    """Get target modules for a model architecture.
    
    Args:
        model: The model (used to infer architecture)
        model_type: Explicit model type (e.g., 'llama', 'qwen')
        include_mlp: Whether to include MLP layers
        
    Returns:
        List of module names to target
    """
    # Determine model type
    if model_type is None and model is not None:
        config = model.config
        model_type = getattr(config, "model_type", "default").lower()
    
    model_type = model_type or "default"
    
    # Get modules for this architecture
    modules = TARGET_MODULES_MAP.get(model_type, TARGET_MODULES_MAP["default"])
    
    # Filter out MLP layers if not wanted
    if not include_mlp:
        mlp_keywords = ["gate", "up", "down", "fc", "dense_h", "dense_4h"]
        modules = [m for m in modules if not any(kw in m for kw in mlp_keywords)]
    
    return modules


def print_trainable_parameters(model: PreTrainedModel) -> dict:
    """Print and return trainable parameter statistics.
    
    Args:
        model: The model to analyze
        
    Returns:
        Dictionary with parameter counts
    """
    trainable_params = 0
    all_params = 0
    
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    trainable_percent = 100 * trainable_params / all_params if all_params > 0 else 0
    
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {all_params:,}")
    print(f"Trainable %: {trainable_percent:.4f}%")
    
    return {
        "trainable_params": trainable_params,
        "all_params": all_params,
        "trainable_percent": trainable_percent,
    }


def find_all_linear_names(model: PreTrainedModel, exclude: Optional[List[str]] = None) -> List[str]:
    """Find all linear layer names in the model.
    
    Useful for targeting all linear layers with LoRA.
    
    Args:
        model: The model to analyze
        exclude: Module names to exclude
        
    Returns:
        List of linear layer names
    """
    import torch.nn as nn
    
    exclude = exclude or ["lm_head"]
    linear_names = set()
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Get the last part of the name
            names = name.split(".")
            layer_name = names[-1]
            if layer_name not in exclude:
                linear_names.add(layer_name)
    
    return list(linear_names)
