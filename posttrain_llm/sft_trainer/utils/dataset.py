"""Dataset utilities for SFT training."""

from typing import Dict, List, Optional, Union

import pandas as pd
from datasets import Dataset, load_dataset


def load_sft_dataset(
    dataset_name: str,
    split: str = "train",
    max_samples: Optional[int] = None,
    streaming: bool = False,
) -> Dataset:
    """Load a dataset for SFT training.
    
    Args:
        dataset_name: HuggingFace dataset ID
        split: Dataset split to load
        max_samples: Maximum number of samples to load
        streaming: Whether to use streaming mode
        
    Returns:
        Loaded dataset
    """
    print(f"Loading dataset: {dataset_name} (split: {split})")
    
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    
    if max_samples and not streaming:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"Limited to {len(dataset)} samples")
    
    return dataset


def display_dataset(
    dataset: Dataset,
    num_examples: int = 3,
    max_content_length: Optional[int] = None,
) -> pd.DataFrame:
    """Display dataset examples in a formatted table.
    
    Args:
        dataset: The dataset to display
        num_examples: Number of examples to show
        max_content_length: Maximum content length to display
        
    Returns:
        DataFrame with examples
    """
    rows = []
    
    for i in range(min(num_examples, len(dataset))):
        example = dataset[i]
        
        # Handle different dataset formats
        if "messages" in example:
            # Chat format
            messages = example["messages"]
            user_msg = next(
                (m["content"] for m in messages if m["role"] == "user"),
                "N/A"
            )
            assistant_msg = next(
                (m["content"] for m in messages if m["role"] == "assistant"),
                "N/A"
            )
        elif "prompt" in example and "completion" in example:
            # Prompt-completion format
            user_msg = example["prompt"]
            assistant_msg = example["completion"]
        elif "instruction" in example and "output" in example:
            # Instruction format
            user_msg = example["instruction"]
            if "input" in example and example["input"]:
                user_msg = f"{user_msg}\n\nInput: {example['input']}"
            assistant_msg = example["output"]
        elif "text" in example:
            # Raw text format
            user_msg = example["text"][:200] + "..." if len(example["text"]) > 200 else example["text"]
            assistant_msg = "N/A (raw text format)"
        else:
            user_msg = str(example)[:200]
            assistant_msg = "Unknown format"
        
        # Truncate if needed
        if max_content_length:
            if len(user_msg) > max_content_length:
                user_msg = user_msg[:max_content_length] + "..."
            if len(assistant_msg) > max_content_length:
                assistant_msg = assistant_msg[:max_content_length] + "..."
        
        rows.append({
            "User Prompt": user_msg,
            "Assistant Response": assistant_msg,
        })
    
    df = pd.DataFrame(rows)
    pd.set_option("display.max_colwidth", None)
    
    return df


def format_chat_messages(
    user_message: str,
    assistant_message: Optional[str] = None,
    system_message: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Format messages into chat format.
    
    Args:
        user_message: The user's message
        assistant_message: Optional assistant response
        system_message: Optional system message
        
    Returns:
        List of message dictionaries
    """
    messages = []
    
    if system_message:
        messages.append({"role": "system", "content": system_message})
    
    messages.append({"role": "user", "content": user_message})
    
    if assistant_message:
        messages.append({"role": "assistant", "content": assistant_message})
    
    return messages


def convert_to_chat_format(
    dataset: Dataset,
    user_column: str = "instruction",
    assistant_column: str = "output",
    input_column: Optional[str] = "input",
    system_message: Optional[str] = None,
) -> Dataset:
    """Convert a dataset to chat format.
    
    Args:
        dataset: Input dataset
        user_column: Column containing user messages
        assistant_column: Column containing assistant responses
        input_column: Optional column for additional input
        system_message: Optional system message to add
        
    Returns:
        Dataset with 'messages' column
    """
    def convert_example(example):
        user_content = example[user_column]
        
        # Add input if present
        if input_column and input_column in example and example[input_column]:
            user_content = f"{user_content}\n\nInput: {example[input_column]}"
        
        messages = format_chat_messages(
            user_message=user_content,
            assistant_message=example[assistant_column],
            system_message=system_message,
        )
        
        return {"messages": messages}
    
    return dataset.map(convert_example)
