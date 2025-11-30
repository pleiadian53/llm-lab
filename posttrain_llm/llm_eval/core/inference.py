"""
Inference utilities for batch processing and prompt handling.
"""

from typing import List
from llm_eval.core.model_service import ServeLLM


def process_prompts(model_name: str, prompts: List[str], **kwargs) -> List[str]:
    """
    Process a list of prompts with a given model and return responses.
    
    This is a convenience function that handles model loading and cleanup automatically.
    
    Args:
        model_name: Path or name of the model to use
        prompts: List of prompts to process
        **kwargs: Additional arguments to pass to generate_response()
        
    Returns:
        List of model responses
        
    Example:
        >>> prompts = ["What is 2+2?", "What is 3+3?"]
        >>> results = process_prompts("model-name", prompts)
    """
    results = []
    with ServeLLM(model_name) as llm:
        for prompt in prompts:
            response = llm.generate_response(prompt, **kwargs)
            results.append(response)
    return results


def generate_batch(
    llm: ServeLLM,
    prompts: List[str],
    **kwargs
) -> List[str]:
    """
    Generate responses for a batch of prompts using an already-loaded model.
    
    Use this when you want to reuse the same model instance for multiple batches.
    
    Args:
        llm: An initialized ServeLLM instance
        prompts: List of prompts to process
        **kwargs: Additional arguments to pass to generate_response()
        
    Returns:
        List of model responses
        
    Example:
        >>> with ServeLLM("model-name") as llm:
        ...     batch1 = generate_batch(llm, prompts1)
        ...     batch2 = generate_batch(llm, prompts2)
    """
    results = []
    for prompt in prompts:
        response = llm.generate_response(prompt, **kwargs)
        results.append(response)
    return results
