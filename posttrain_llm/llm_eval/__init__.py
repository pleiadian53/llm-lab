"""
LLM Eval - A Reusable LLM Evaluation Toolkit

This package provides modular tools for evaluating language models across
various tasks including mathematical reasoning, safety classification, and more.

Modules:
    - core: Core model serving and inference utilities
    - metrics: Evaluation metrics and scoring functions
    - utils: Helper utilities for display, validation, etc.

Example:
    >>> from llm_eval.core import ServeLLM
    >>> from llm_eval.metrics import evaluate_math_reasoning
    >>> 
    >>> with ServeLLM("model-name") as llm:
    ...     results = evaluate_math_reasoning(llm, dataset)
"""

__version__ = "0.1.0"
__author__ = "pleiadian53"

# Core imports
from llm_eval.core.model_service import ServeLLM
from llm_eval.core.inference import generate_batch, process_prompts

# Metrics imports
from llm_eval.metrics.math_reasoning import (
    extract_number,
    evaluate_math_reasoning,
    score_response,
    score_all_responses
)
from llm_eval.metrics.safety import (
    parse_llama_guard_response,
    calculate_safety_metrics,
    analyze_safety_categories,
    evaluate_safety
)

# Utils imports
from llm_eval.utils.display import (
    display_section_header,
    display_warning,
    display_success,
    display_info
)
from llm_eval.utils.huggingface import validate_token

__all__ = [
    # Core
    'ServeLLM',
    'generate_batch',
    'process_prompts',
    # Math reasoning
    'extract_number',
    'evaluate_math_reasoning',
    'score_response',
    'score_all_responses',
    # Safety
    'parse_llama_guard_response',
    'calculate_safety_metrics',
    'analyze_safety_categories',
    'evaluate_safety',
    # Utils
    'display_section_header',
    'display_warning',
    'display_success',
    'display_info',
    'validate_token',
]
