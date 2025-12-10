"""
Dataset loading and processing for Online RL training.
"""

import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
from datasets import load_dataset, Dataset


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders."""
    
    @abstractmethod
    def load(
        self, 
        split: str = "train",
        num_samples: Optional[int] = None
    ) -> Dataset:
        """
        Load and preprocess the dataset.
        
        Args:
            split: Dataset split ('train', 'test', 'validation')
            num_samples: Number of samples to load (None for all)
            
        Returns:
            HuggingFace Dataset object
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this dataset."""
        pass


class GSM8KLoader(DatasetLoader):
    """
    Loader for GSM8K (Grade School Math 8K) dataset.
    
    GSM8K contains grade school math word problems with step-by-step solutions.
    """
    
    def __init__(
        self, 
        system_prompt: Optional[str] = None,
        include_cot: bool = False
    ):
        """
        Args:
            system_prompt: System prompt to prepend to each example
            include_cot: Whether to include chain-of-thought in training
        """
        self.system_prompt = system_prompt or (
            "You are a helpful assistant that solves problems step-by-step. "
            "Always include the final numeric answer inside \\boxed{}."
        )
        self.include_cot = include_cot
    
    @property
    def name(self) -> str:
        return "gsm8k"
    
    def _extract_answer(self, answer_text: str) -> str:
        """Extract the final numeric answer from GSM8K format."""
        # GSM8K format: "... #### <answer>"
        match = re.search(r"####\s*(-?\d+)", answer_text)
        return match.group(1) if match else ""
    
    def _process_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single example."""
        # Extract ground truth answer
        ground_truth = self._extract_answer(example["answer"])
        
        # Create prompt in chat format
        prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": example["question"]}
        ]
        
        return {
            "prompt": prompt,
            "ground_truth": ground_truth,
        }
    
    def load(
        self, 
        split: str = "train",
        num_samples: Optional[int] = None
    ) -> Dataset:
        """Load GSM8K dataset."""
        dataset = load_dataset("openai/gsm8k", "main")[split]
        
        if num_samples is not None:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        # Process examples
        dataset = dataset.map(
            self._process_example,
            remove_columns=["question", "answer"]
        )
        
        return dataset


class MATHLoader(DatasetLoader):
    """
    Loader for MATH dataset (competition mathematics).
    
    More challenging than GSM8K, includes algebra, geometry, etc.
    """
    
    def __init__(
        self, 
        system_prompt: Optional[str] = None,
        difficulty_filter: Optional[List[int]] = None
    ):
        """
        Args:
            system_prompt: System prompt to prepend
            difficulty_filter: List of difficulty levels to include (1-5)
        """
        self.system_prompt = system_prompt or (
            "You are a helpful assistant that solves math problems step-by-step. "
            "Always include the final answer inside \\boxed{}."
        )
        self.difficulty_filter = difficulty_filter
    
    @property
    def name(self) -> str:
        return "math"
    
    def _process_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single example."""
        # Extract answer from solution (MATH uses \boxed{} format)
        match = re.search(r"\\boxed\{(.*?)\}", example.get("solution", ""))
        ground_truth = match.group(1) if match else ""
        
        prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": example["problem"]}
        ]
        
        return {
            "prompt": prompt,
            "ground_truth": ground_truth,
        }
    
    def load(
        self, 
        split: str = "train",
        num_samples: Optional[int] = None
    ) -> Dataset:
        """Load MATH dataset."""
        dataset = load_dataset("hendrycks/competition_math")[split]
        
        # Filter by difficulty if specified
        if self.difficulty_filter:
            dataset = dataset.filter(
                lambda x: x.get("level", 0) in self.difficulty_filter
            )
        
        if num_samples is not None:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        dataset = dataset.map(
            self._process_example,
            remove_columns=dataset.column_names
        )
        
        return dataset


class CustomDatasetLoader(DatasetLoader):
    """
    Loader for custom datasets.
    
    Expects dataset with 'prompt' and 'ground_truth' columns,
    or a custom processing function.
    """
    
    def __init__(
        self,
        dataset_name_or_path: str,
        system_prompt: Optional[str] = None,
        prompt_column: str = "prompt",
        answer_column: str = "answer",
        process_fn: Optional[Callable] = None
    ):
        """
        Args:
            dataset_name_or_path: HuggingFace dataset name or local path
            system_prompt: System prompt to prepend
            prompt_column: Name of the prompt column
            answer_column: Name of the answer column
            process_fn: Custom processing function
        """
        self.dataset_name_or_path = dataset_name_or_path
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.prompt_column = prompt_column
        self.answer_column = answer_column
        self.process_fn = process_fn
    
    @property
    def name(self) -> str:
        return f"custom:{self.dataset_name_or_path}"
    
    def _default_process(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Default processing function."""
        prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": example[self.prompt_column]}
        ]
        
        return {
            "prompt": prompt,
            "ground_truth": str(example.get(self.answer_column, "")),
        }
    
    def load(
        self, 
        split: str = "train",
        num_samples: Optional[int] = None
    ) -> Dataset:
        """Load custom dataset."""
        dataset = load_dataset(self.dataset_name_or_path)[split]
        
        if num_samples is not None:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        process_fn = self.process_fn or self._default_process
        dataset = dataset.map(process_fn)
        
        return dataset


# Factory function
def get_dataset_loader(
    dataset_name: str,
    system_prompt: Optional[str] = None,
    **kwargs
) -> DatasetLoader:
    """
    Get a dataset loader by name.
    
    Args:
        dataset_name: One of 'gsm8k', 'math', or a custom dataset path
        system_prompt: System prompt to use
        **kwargs: Additional arguments for the loader
        
    Returns:
        DatasetLoader instance
    """
    loaders = {
        "gsm8k": GSM8KLoader,
        "math": MATHLoader,
    }
    
    if dataset_name.lower() in loaders:
        return loaders[dataset_name.lower()](system_prompt=system_prompt, **kwargs)
    else:
        return CustomDatasetLoader(
            dataset_name_or_path=dataset_name,
            system_prompt=system_prompt,
            **kwargs
        )
