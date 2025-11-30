"""
LLM Evaluation Library

This package provides modular tools for evaluating language models on:
- Mathematical reasoning (GSM8K dataset)
- Safety classification (Llama Guard)
"""

from .model_evaluation import (
    process_prompts,
    extract_number,
    evaluate_model_correctness,
    score_response,
    score_all_responses
)

from .safety_evaluation import (
    parse_llama_guard_response,
    calculate_safety_metrics,
    analyze_safety_categories,
    evaluate_safety_model
)

__all__ = [
    # Model evaluation
    'process_prompts',
    'extract_number',
    'evaluate_model_correctness',
    'score_response',
    'score_all_responses',
    # Safety evaluation
    'parse_llama_guard_response',
    'calculate_safety_metrics',
    'analyze_safety_categories',
    'evaluate_safety_model',
]
