"""
Core module for LLM serving and inference.

This module provides the fundamental building blocks for loading and
running language models with proper memory management and error handling.
"""

from llm_eval.core.model_service import ServeLLM
from llm_eval.core.inference import generate_batch, process_prompts

__all__ = ['ServeLLM', 'generate_batch', 'process_prompts']
