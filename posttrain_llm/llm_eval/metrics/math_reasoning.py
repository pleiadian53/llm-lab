"""
Mathematical Reasoning Evaluation

Provides metrics and evaluation functions for mathematical reasoning tasks,
particularly focused on GSM8K-style problems.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
from llm_eval.core import ServeLLM


def extract_number(text: str) -> Optional[float]:
    """
    Extract the final numerical answer from a model's generated output.
    
    GSM8K answers are formatted like '#### 42', but this function also
    looks for the last number in the text as a fallback.
    
    Args:
        text: Text containing a numerical answer
        
    Returns:
        Extracted number as float, or None if no number found
        
    Example:
        >>> extract_number("The answer is #### 42")
        42.0
        >>> extract_number("We calculate 2 + 2 = 4")
        4.0
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


def evaluate_math_reasoning(
    model_path: str,
    dataset,
    num_samples: int = 30,
    verbose: bool = True
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Evaluate a model's correctness on GSM8K-style math problems.
    
    Args:
        model_path: Path to the model (HuggingFace ID or local path)
        dataset: GSM8K dataset (or compatible format with 'question' and 'answer' fields)
        num_samples: Number of samples to test
        verbose: Whether to print progress and examples
        
    Returns:
        Tuple of (accuracy, detailed_results)
        - accuracy: Float between 0 and 1
        - detailed_results: List of dicts with question, answers, and correctness
        
    Example:
        >>> from datasets import load_dataset
        >>> dataset = load_dataset("gsm8k", "main", split="test")
        >>> accuracy, results = evaluate_math_reasoning("model-name", dataset, 50)
        >>> print(f"Accuracy: {accuracy:.2%}")
    """
    if verbose:
        print(f"Evaluating {model_path} on {num_samples} GSM8K problems...")
    
    # Get subset of data
    test_data = dataset.select(range(num_samples))
    
    correct = 0
    results = []
    
    with ServeLLM(model_path) as llm:
        iterator = tqdm(test_data, desc="Processing") if verbose else test_data
        
        for i, sample in enumerate(iterator):
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
                'correct': is_correct,
                'response': response
            })
            
            # Show first few examples
            if verbose and i < 3:
                print(f"\nExample {i+1}:")
                print(f"Question: {sample['question'][:100]}...")
                print(f"Gold: {gold_answer}, Model: {model_answer}, Correct: {is_correct}")
    
    accuracy = correct / num_samples
    return accuracy, results


def score_response(response: str, expected_keyword: str) -> int:
    """
    Score a single response based on whether it contains the expected keyword.
    
    This is a simple keyword-matching metric useful for quick evaluations.
    
    Args:
        response: Model response text
        expected_keyword: Expected keyword in correct answer
        
    Returns:
        1 if correct (keyword found), 0 if incorrect
        
    Example:
        >>> score_response("The answer is 42", "42")
        1
        >>> score_response("The answer is 43", "42")
        0
    """
    response_lower = response.lower()
    keyword_lower = expected_keyword.lower()
    return 1 if keyword_lower in response_lower else 0


def score_all_responses(
    model_results: List[str],
    expected_keywords: List[str]
) -> Tuple[List[int], float]:
    """
    Score all responses for a model using keyword matching.
    
    Args:
        model_results: List of model responses
        expected_keywords: List of expected keywords (one per response)
        
    Returns:
        Tuple of (scores, average_score)
        - scores: List of binary scores (0 or 1)
        - average_score: Float between 0 and 1
        
    Example:
        >>> responses = ["The answer is 42", "The answer is cos(x)"]
        >>> keywords = ["42", "cos(x)"]
        >>> scores, avg = score_all_responses(responses, keywords)
        >>> print(f"Average: {avg:.2%}")
    """
    scores = []
    for response, keyword in zip(model_results, expected_keywords):
        score = score_response(response, keyword)
        scores.append(score)
    
    avg_score = sum(scores) / len(scores) if scores else 0.0
    return scores, avg_score
