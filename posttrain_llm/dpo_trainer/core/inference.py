"""Inference utilities for DPO-trained models."""

from typing import Dict, List, Optional, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    user_message: str,
    system_message: Optional[str] = None,
    max_new_tokens: int = 300,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> str:
    """Generate a response from the model.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        user_message: User's input message
        system_message: Optional system prompt
        max_new_tokens: Maximum tokens to generate
        do_sample: Whether to use sampling
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
    
    Returns:
        Generated response string
    """
    # Build messages
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})
    
    # Apply chat template
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback for models without chat template
        prompt = f"User: {user_message}\nAssistant:"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part
    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    return response


def generate_from_messages(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 300,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> str:
    """Generate a response from a list of messages.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        messages: List of message dicts with 'role' and 'content'
        max_new_tokens: Maximum tokens to generate
        do_sample: Whether to use sampling
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
    
    Returns:
        Generated response string
    """
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part
    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    return response


def test_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    questions: List[str],
    system_message: Optional[str] = None,
    title: str = "Model Output",
    max_new_tokens: int = 300,
) -> List[str]:
    """Test a model with a list of questions.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        questions: List of questions to ask
        system_message: Optional system prompt
        title: Title for the output section
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        List of generated responses
    """
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    
    responses = []
    for i, question in enumerate(questions, 1):
        response = generate_response(
            model, tokenizer, question, 
            system_message=system_message,
            max_new_tokens=max_new_tokens,
        )
        responses.append(response)
        
        print(f"\n[Question {i}]: {question}")
        print(f"[Response {i}]: {response}")
    
    print(f"\n{'='*60}\n")
    return responses


def batch_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    system_message: Optional[str] = None,
    max_new_tokens: int = 300,
    batch_size: int = 4,
    do_sample: bool = False,
) -> List[str]:
    """Generate responses for multiple prompts in batches.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts: List of prompts
        system_message: Optional system prompt
        max_new_tokens: Maximum tokens to generate
        batch_size: Batch size for generation
        do_sample: Whether to use sampling
    
    Returns:
        List of generated responses
    """
    responses = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_responses = []
        
        for prompt in batch_prompts:
            response = generate_response(
                model, tokenizer, prompt,
                system_message=system_message,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
            )
            batch_responses.append(response)
        
        responses.extend(batch_responses)
    
    return responses


def compare_models(
    models: Dict[str, PreTrainedModel],
    tokenizer: PreTrainedTokenizer,
    questions: List[str],
    system_message: Optional[str] = None,
    max_new_tokens: int = 300,
) -> Dict[str, List[str]]:
    """Compare responses from multiple models.
    
    Args:
        models: Dict mapping model names to model objects
        tokenizer: The tokenizer (shared across models)
        questions: List of questions to ask
        system_message: Optional system prompt
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        Dict mapping model names to lists of responses
    """
    results = {}
    
    for model_name, model in models.items():
        print(f"\n--- Testing: {model_name} ---")
        responses = test_model(
            model, tokenizer, questions,
            system_message=system_message,
            title=f"{model_name} Output",
            max_new_tokens=max_new_tokens,
        )
        results[model_name] = responses
    
    return results
