"""Inference utilities for SFT models."""

from typing import List, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    user_message: Optional[str] = None,
    system_message: Optional[str] = None,
    messages: Optional[List[dict]] = None,
    max_new_tokens: int = 300,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    enable_thinking: bool = False,
) -> str:
    """Generate a response from the model.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        user_message: User message (used if messages is None)
        system_message: Optional system message
        messages: Full message list (overrides user_message/system_message)
        max_new_tokens: Maximum tokens to generate
        do_sample: Whether to use sampling
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        enable_thinking: Enable thinking mode for compatible models
        
    Returns:
        Generated response string
    """
    # Build messages if not provided
    if messages is None:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        if user_message:
            messages.append({"role": "user", "content": user_message})
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
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
    """Test model with a list of questions.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        questions: List of questions to test
        system_message: Optional system message
        title: Title for the output section
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        List of generated responses
    """
    print(f"\n=== {title} ===")
    responses = []
    
    for i, question in enumerate(questions, 1):
        response = generate_response(
            model, tokenizer, question, system_message,
            max_new_tokens=max_new_tokens
        )
        print(f"\nModel Input {i}:\n{question}\nModel Output {i}:\n{response}\n")
        responses.append(response)
    
    return responses


def batch_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    max_new_tokens: int = 300,
    batch_size: int = 4,
    do_sample: bool = False,
) -> List[str]:
    """Generate responses for multiple prompts in batches.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts: List of prompts
        max_new_tokens: Maximum tokens to generate
        batch_size: Batch size for generation
        do_sample: Whether to use sampling
        
    Returns:
        List of generated responses
    """
    responses = []
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode each response
        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"][j].shape[0]
            generated_ids = output[input_len:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            responses.append(response)
    
    return responses
