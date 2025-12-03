# Add your utilities or helper functions to this file.

import os
import torch
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
import pandas as pd

def generate_responses(model, tokenizer, user_message=None, system_message=None, max_new_tokens=300, full_message=None):
    # Format chat using tokenizer's chat template
    if full_message:
        messages = full_message
    else:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})
        
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return response
    
def test_model_with_questions(model, tokenizer, questions, system_message=None, title="Model Output"):
    print(f"\n=== {title} ===")
    for i, question in enumerate(questions, 1):
        response = generate_responses(model, tokenizer, question, system_message)
        print(f"\nModel Input {i}:\n{question}\nModel Output {i}:\n{response}\n")

def load_model_and_tokenizer(model_name, use_gpu=False, device=None):
    """
    Load model and tokenizer with support for CUDA, MPS (Apple Silicon), and CPU.
    
    Args:
        model_name: HuggingFace model name or local path
        use_gpu: If True, automatically select best available device (cuda > mps > cpu)
        device: Explicit device string ('cuda', 'mps', 'cpu'). Overrides use_gpu.
    
    Returns:
        model, tokenizer
    """
    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Determine device
    if device is not None:
        # Explicit device specified
        target_device = device
    elif use_gpu:
        # Auto-select best available device
        if torch.cuda.is_available():
            target_device = "cuda"
            print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            target_device = "mps"
            print("Using Apple Silicon GPU (MPS)")
        else:
            target_device = "cpu"
            print("No GPU available, using CPU")
    else:
        target_device = "cpu"
        print("Using CPU (use_gpu=False)")
    
    # Move model to device
    model.to(target_device)
    print(f"Model loaded on device: {target_device}")
    
    if not tokenizer.chat_template:
        tokenizer.chat_template = """{% for message in messages %}
                {% if message['role'] == 'system' %}System: {{ message['content'] }}\n
                {% elif message['role'] == 'user' %}User: {{ message['content'] }}\n
                {% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }} <|endoftext|>
                {% endif %}
                {% endfor %}"""
    
    # Tokenizer config
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer



def display_dataset(dataset):
    # Visualize the dataset 
    rows = []
    for i in range(3):
        example = dataset[i]
        user_msg = next(m['content'] for m in example['messages'] if m['role'] == 'user')
        assistant_msg = next(m['content'] for m in example['messages'] if m['role'] == 'assistant')
        rows.append({
            'User Prompt': user_msg,
            'Assistant Response': assistant_msg
        })
    
    # Display as table
    df = pd.DataFrame(rows)
    pd.set_option('display.max_colwidth', None)  # Avoid truncating long strings
    display(df)
