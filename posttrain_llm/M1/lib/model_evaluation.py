"""
Model Evaluation Module

This module provides functions for evaluating LLM models on mathematical reasoning
and safety classification tasks.
"""

import re
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from utils.utils import ServeLLM


def process_prompts(model_name: str, prompts: List[str]) -> List[str]:
    """
    Process a list of prompts with a given model and return responses.
    
    Args:
        model_name: Path or name of the model to use
        prompts: List of prompts to process
        
    Returns:
        List of model responses
    """
    results = []
    with ServeLLM(model_name) as llm:
        for prompt in prompts:
            response = llm.generate_response(prompt)
            results.append(response)
    return results


def extract_number(text: str) -> float:
    """
    Extract the final numerical answer from a model's generated output.
    
    GSM8K answers are formatted like '#### 42', but this function also
    looks for the last number in the text as a fallback.
    
    Args:
        text: Text containing a numerical answer
        
    Returns:
        Extracted number as float, or None if no number found
    """
    # Try to extract the canonical GSM8K answer pattern first: '#### <number>'
    gsm8k_format = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    if gsm8k_format:
        try:
            return float(gsm8k_format.group(1))
        except ValueError:
            pass

    # Fallback: extract the last standalone number in the text
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            return None
    return None


def evaluate_model_correctness(
    model_path: str,
    dataset,
    num_samples: int = 30
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Evaluate a model's correctness on GSM8K problems.
    
    Args:
        model_path: Path to the model
        dataset: GSM8K dataset
        num_samples: Number of samples to test
        
    Returns:
        Tuple of (accuracy, detailed_results)
    """
    print(f"Evaluating {model_path} on {num_samples} GSM8K problems...")
    
    # Get subset of data
    test_data = dataset.select(range(num_samples))
    
    correct = 0
    results = []
    
    with ServeLLM(model_path) as llm:
        for i, sample in enumerate(tqdm(test_data, desc="Processing")):
            # Create prompt
            prompt = f"Solve this math problem step by step:\n{sample['question']}\n\nAnswer:"
            
            # Generate response from model
            response = llm.generate_response(prompt, max_tokens=512)
            
            # Extract model's numerical answer
            model_answer = extract_number(response)
            
            # Extract correct answer from dataset
            gold_answer = extract_number(sample['answer'])
            
            # Check if answers match
            is_correct = (
                (model_answer == gold_answer)
                if (model_answer is not None and gold_answer is not None)
                else False
            )
            
            if is_correct:
                correct += 1
            
            # Store result for analysis
            results.append({
                'question': sample['question'],
                'gold_answer': gold_answer,
                'model_answer': model_answer,
                'correct': is_correct
            })
            
            # Show first few examples
            if i < 3:
                print(f"\nExample {i+1}:")
                print(f"Question: {sample['question'][:100]}...")
                print(f"Gold: {gold_answer}, Model: {model_answer}, Correct: {is_correct}")
    
    accuracy = correct / num_samples
    return accuracy, results


def score_response(response: str, expected_keyword: str) -> int:
    """
    Score a single response based on whether it contains the expected keyword.
    
    Args:
        response: Model response
        expected_keyword: Expected keyword in correct answer
        
    Returns:
        1 if correct, 0 if incorrect
    """
    response_lower = response.lower()
    keyword_lower = expected_keyword.lower()
    return 1 if keyword_lower in response_lower else 0


def score_all_responses(
    model_results: List[str],
    expected_keywords: List[str]
) -> Tuple[List[int], float]:
    """
    Score all responses for a model.
    
    Args:
        model_results: List of model responses
        expected_keywords: List of expected keywords
        
    Returns:
        Tuple of (scores, average_score)
    """
    scores = []
    for response, keyword in zip(model_results, expected_keywords):
        score = score_response(response, keyword)
        scores.append(score)
    
    avg_score = sum(scores) / len(scores)
    return scores, avg_score
