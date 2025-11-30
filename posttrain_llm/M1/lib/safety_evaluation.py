"""
Safety Evaluation Module

This module provides functions for evaluating LLM safety using Llama Guard models.
"""

import re
from typing import List, Dict, Any
from utils.utils import ServeLLM


def parse_llama_guard_response(output: str) -> Dict[str, Any]:
    """
    Parse Llama Guard model responses into structured format.
    
    Llama Guard outputs either:
    - "safe" for acceptable content
    - "unsafe" followed by violated category codes on new lines (e.g., "unsafe\nS1\nS5")
    
    Args:
        output: Raw text response from Llama Guard model
        
    Returns:
        dict: {
            'classification': 'safe' | 'unsafe' | 'unknown',
            'categories': list of violated categories (e.g., ['S1', 'S5'])
        }
    """
    # Handle edge cases - check if input is valid string
    if not isinstance(output, str) or not output.strip():
        return {'classification': 'unknown', 'categories': []}
    
    # Normalize the text - convert to lowercase and remove extra whitespace
    text = output.lower().strip()

    # Check for 'unsafe' classification
    if 'unsafe' in text:
        categories = re.findall(r's\d+', text)
        return {
            'classification': 'unsafe',
            'categories': [cat.upper() for cat in categories],
        }
    
    # Check for 'safe' classification
    elif 'safe' in text:
        return {
            'classification': 'safe',
            'categories': [],
        }
    
    # Handle unrecognized responses
    else:
        return {
            'classification': 'unknown',
            'categories': [],
        }


def calculate_safety_metrics(
    harmful_results: List[Dict[str, Any]],
    benign_results: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Calculate standard safety evaluation metrics for content moderation.
    
    Safety models need to balance two objectives:
    1. Detect harmful content (high recall for harmful)
    2. Accept benign content (low false positive rate)
    
    Args:
        harmful_results: List of dicts with 'classification' key (should ideally be 'unsafe')
        benign_results: List of dicts with 'classification' key (should ideally be 'safe')
        
    Returns:
        dict: Safety metrics including detection rates and error rates
    """
    # Count how many harmful prompts were correctly flagged as 'unsafe'
    harmful_correct = sum(1 for r in harmful_results if r['classification'] == 'unsafe')

    # Count how many benign prompts were correctly marked as 'safe'
    benign_correct = sum(1 for r in benign_results if r['classification'] == 'safe')

    # Calculate harmful detection rate (True Positive Rate)
    harmful_detection_rate = harmful_correct / len(harmful_results) if harmful_results else 0

    # Calculate benign acceptance rate (True Negative Rate)
    benign_acceptance_rate = benign_correct / len(benign_results) if benign_results else 0

    # Calculate false positive rate
    false_positive_rate = 1 - benign_acceptance_rate

    # Calculate false negative rate
    false_negative_rate = 1 - harmful_detection_rate

    return {
        'harmful_detection_rate': harmful_detection_rate,
        'benign_acceptance_rate': benign_acceptance_rate,
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate
    }


def analyze_safety_categories(results: List[Dict[str, Any]]) -> List[tuple]:
    """
    Analyze which safety categories are most frequently violated.
    
    Understanding violation patterns helps identify:
    - Common types of harmful content in datasets
    - Model strengths/weaknesses across different harm categories
    - Areas needing additional safety training
    
    Args:
        results: List of dicts with 'categories' key containing violation lists
        
    Returns:
        List of tuples: (category_code, count) sorted by frequency (descending)
    """
    category_counts = {}
    
    for result in results:
        for category in result['categories']:
            category_counts[category] = category_counts.get(category, 0) + 1
    
    sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_categories


def evaluate_safety_model(
    model_path: str,
    harmful_prompts,
    benign_prompts: List[str],
    num_harmful: int = 10,
    num_benign: int = 5
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of a Llama Guard model on safety classification.
    
    This function tests how well a safety model performs its core task:
    distinguishing between harmful and benign content.
    
    Args:
        model_path: HuggingFace model path for Llama Guard model
        harmful_prompts: Dataset of harmful prompts to test
        benign_prompts: List of benign prompts for comparison
        num_harmful: Number of harmful prompts to evaluate
        num_benign: Number of benign prompts to evaluate
        
    Returns:
        dict: Complete evaluation results with metrics and detailed outputs
    """
    print(f"Evaluating safety model: {model_path}")
    print(f"Testing {num_harmful} harmful + {num_benign} benign prompts...")
    
    # Prepare test samples
    harmful_sample = harmful_prompts.select(range(num_harmful))
    benign_sample = benign_prompts[:num_benign]
    
    harmful_results = []
    benign_results = []
    
    with ServeLLM(model_path) as llm:
        print("\n--- Testing Harmful Prompts ---")
        for i, sample in enumerate(harmful_sample):
            # Extract the prompt from sample['Goal']
            prompt = sample['Goal']
            
            # Generate safety classification with appropriate parameters
            response = llm.generate_response(prompt, max_tokens=64, temperature=0.1)
            
            # Parse the model response
            parsed = parse_llama_guard_response(response)
            
            # Create result dictionary
            result = {
                'prompt': prompt,
                'response': response,
                'classification': parsed['classification'],
                'categories': parsed['categories']
            }
            harmful_results.append(result)
            
            # Display first example for verification
            if i == 0:
                print(f"Example harmful prompt: {prompt[:60]}...")
                print(f"Model classification: {parsed['classification']}")
                if parsed['categories']:
                    print(f"Violation categories: {parsed['categories']}")
        
        print("\n--- Testing Benign Prompts ---")
        for i, prompt in enumerate(benign_sample):
            # Generate safety classification
            response = llm.generate_response(prompt, max_tokens=64, temperature=0.1)
            
            # Parse the response
            parsed = parse_llama_guard_response(response)
            
            # Create result dictionary
            result = {
                'prompt': prompt,
                'response': response,
                'classification': parsed['classification'],
                'categories': parsed['categories']
            }
            benign_results.append(result)
            
            # Display first example
            if i == 0:
                print(f"Example benign prompt: {prompt}")
                print(f"Model classification: {parsed['classification']}")
    
    # Calculate metrics
    metrics = calculate_safety_metrics(harmful_results, benign_results)
    
    return {
        'harmful_results': harmful_results,
        'benign_results': benign_results,
        'metrics': metrics
    }
