"""
Metrics module for LLM evaluation.

Provides evaluation metrics for various tasks including:
- Mathematical reasoning (GSM8K)
- Safety classification (Llama Guard)
- More metrics to be added as you progress through the course
"""

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

__all__ = [
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
]
