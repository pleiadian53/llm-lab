"""Dataset utilities for DPO training.

This module provides functions for building and loading DPO preference datasets,
including the identity shift dataset used in Lesson 5.
"""

from typing import Any, Callable, Dict, List, Optional, Union

import torch
from datasets import Dataset, load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

from .inference import generate_response


def build_identity_shift_dataset(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    original_name: str,
    new_name: str,
    raw_dataset: Optional[Dataset] = None,
    dataset_name: str = "mrfakename/identity",
    system_prompt: str = "You're a helpful assistant.",
    max_samples: Optional[int] = None,
    max_new_tokens: int = 300,
) -> Dataset:
    """Build a DPO dataset for identity shift training.
    
    This creates preference pairs where:
    - Chosen: Response with the new identity name
    - Rejected: Response with the original identity name
    
    Args:
        model: The model to generate responses from
        tokenizer: The tokenizer
        original_name: The original name to replace (e.g., "Qwen")
        new_name: The new name to use (e.g., "Deep Qwen")
        raw_dataset: Pre-loaded raw dataset (optional)
        dataset_name: HuggingFace dataset name if raw_dataset not provided
        system_prompt: System prompt to use
        max_samples: Limit number of samples
        max_new_tokens: Max tokens for generation
    
    Returns:
        Dataset with 'chosen' and 'rejected' columns in ChatML format
    
    Example:
        >>> dataset = build_identity_shift_dataset(
        ...     model, tokenizer,
        ...     original_name="Qwen",
        ...     new_name="Deep Qwen",
        ... )
    """
    # Load raw dataset if not provided
    if raw_dataset is None:
        raw_dataset = load_dataset(dataset_name, split="train")
    
    # Limit samples if specified
    if max_samples and len(raw_dataset) > max_samples:
        raw_dataset = raw_dataset.select(range(max_samples))
    
    def build_dpo_example(example: Dict) -> Dict:
        """Build a single DPO example from a conversation."""
        msgs = example["conversations"]
        
        # Extract the last human prompt
        prompt = next(
            m["value"] for m in reversed(msgs) 
            if m["from"] == "human"
        )
        
        # Generate response from model
        try:
            rejected_resp = generate_response(
                model, tokenizer, prompt,
                max_new_tokens=max_new_tokens,
            )
        except Exception as e:
            rejected_resp = "Error: failed to generate response."
            print(f"Generation error for prompt: {prompt}\n{e}")
        
        # Create chosen response by replacing the name
        chosen_resp = rejected_resp.replace(original_name, new_name)
        
        # Build ChatML format
        chosen = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen_resp},
        ]
        rejected = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected_resp},
        ]
        
        return {"chosen": chosen, "rejected": rejected}
    
    # Build the DPO dataset
    dpo_dataset = raw_dataset.map(
        build_dpo_example,
        remove_columns=raw_dataset.column_names,
    )
    
    return dpo_dataset


def build_dpo_dataset(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    raw_dataset: Dataset,
    prompt_column: str = "prompt",
    chosen_transform: Optional[Callable[[str], str]] = None,
    system_prompt: str = "You're a helpful assistant.",
    max_new_tokens: int = 300,
) -> Dataset:
    """Build a generic DPO dataset from prompts.
    
    Args:
        model: The model to generate responses from
        tokenizer: The tokenizer
        raw_dataset: Dataset with prompts
        prompt_column: Column name containing prompts
        chosen_transform: Function to transform rejected response to chosen
        system_prompt: System prompt to use
        max_new_tokens: Max tokens for generation
    
    Returns:
        Dataset with 'chosen' and 'rejected' columns
    """
    def build_example(example: Dict) -> Dict:
        prompt = example[prompt_column]
        
        # Generate response
        try:
            rejected_resp = generate_response(
                model, tokenizer, prompt,
                max_new_tokens=max_new_tokens,
            )
        except Exception as e:
            rejected_resp = "Error: failed to generate response."
            print(f"Generation error: {e}")
        
        # Apply transform to create chosen response
        if chosen_transform:
            chosen_resp = chosen_transform(rejected_resp)
        else:
            chosen_resp = rejected_resp
        
        # Build ChatML format
        chosen = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen_resp},
        ]
        rejected = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected_resp},
        ]
        
        return {"chosen": chosen, "rejected": rejected}
    
    return raw_dataset.map(build_example, remove_columns=raw_dataset.column_names)


def load_dpo_dataset(
    dataset_name: str,
    split: str = "train",
    max_samples: Optional[int] = None,
) -> Dataset:
    """Load a pre-built DPO dataset from HuggingFace.
    
    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split to load
        max_samples: Limit number of samples
    
    Returns:
        Dataset with 'chosen' and 'rejected' columns
    """
    dataset = load_dataset(dataset_name, split=split)
    
    if max_samples and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
    
    return dataset


class DPODatasetBuilder:
    """Builder class for creating DPO preference datasets.
    
    This class provides a flexible interface for building DPO datasets
    with various preference pair generation strategies.
    
    Example:
        >>> builder = DPODatasetBuilder(model, tokenizer)
        >>> builder.set_system_prompt("You are a helpful assistant.")
        >>> builder.set_transform(lambda x: x.replace("bad", "good"))
        >>> dataset = builder.build_from_prompts(prompts)
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ):
        """Initialize the dataset builder.
        
        Args:
            model: The model to generate responses from
            tokenizer: The tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = "You're a helpful assistant."
        self.transform = None
        self.max_new_tokens = 300
    
    def set_system_prompt(self, prompt: str) -> "DPODatasetBuilder":
        """Set the system prompt."""
        self.system_prompt = prompt
        return self
    
    def set_transform(self, transform: Callable[[str], str]) -> "DPODatasetBuilder":
        """Set the transform function for creating chosen responses."""
        self.transform = transform
        return self
    
    def set_max_tokens(self, max_tokens: int) -> "DPODatasetBuilder":
        """Set maximum tokens for generation."""
        self.max_new_tokens = max_tokens
        return self
    
    def build_from_prompts(self, prompts: List[str]) -> Dataset:
        """Build DPO dataset from a list of prompts.
        
        Args:
            prompts: List of user prompts
        
        Returns:
            Dataset with 'chosen' and 'rejected' columns
        """
        data = []
        
        for prompt in prompts:
            # Generate response
            try:
                rejected_resp = generate_response(
                    self.model, self.tokenizer, prompt,
                    max_new_tokens=self.max_new_tokens,
                )
            except Exception as e:
                rejected_resp = "Error: failed to generate response."
                print(f"Generation error: {e}")
            
            # Apply transform
            if self.transform:
                chosen_resp = self.transform(rejected_resp)
            else:
                chosen_resp = rejected_resp
            
            # Build ChatML format
            chosen = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": chosen_resp},
            ]
            rejected = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": rejected_resp},
            ]
            
            data.append({"chosen": chosen, "rejected": rejected})
        
        return Dataset.from_list(data)
    
    def build_identity_shift(
        self,
        original_name: str,
        new_name: str,
        raw_dataset: Optional[Dataset] = None,
        dataset_name: str = "mrfakename/identity",
        max_samples: Optional[int] = None,
    ) -> Dataset:
        """Build an identity shift dataset.
        
        Args:
            original_name: Name to replace
            new_name: New name to use
            raw_dataset: Pre-loaded dataset
            dataset_name: HuggingFace dataset name
            max_samples: Limit samples
        
        Returns:
            DPO dataset for identity shift
        """
        return build_identity_shift_dataset(
            self.model,
            self.tokenizer,
            original_name=original_name,
            new_name=new_name,
            raw_dataset=raw_dataset,
            dataset_name=dataset_name,
            system_prompt=self.system_prompt,
            max_samples=max_samples,
            max_new_tokens=self.max_new_tokens,
        )
