"""
Inference and evaluation module for Online RL trained models.
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from tqdm import tqdm

from .rewards import RewardFunction


class ModelEvaluator:
    """
    Evaluator for Online RL trained models.
    
    Handles model loading, generation, and evaluation.
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        use_gpu: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
    ):
        """
        Initialize the evaluator.
        
        Args:
            model_name_or_path: Path to model or HuggingFace model name
            use_gpu: Whether to use GPU
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling (vs greedy)
        """
        self.model_name_or_path = model_name_or_path
        self.use_gpu = use_gpu
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        
        self.model, self.tokenizer = self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer."""
        print(f"Loading model: {self.model_name_or_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Determine device
        if self.use_gpu and torch.cuda.is_available():
            device_map = "auto"
            torch_dtype = torch.float16
        else:
            device_map = "cpu"
            torch_dtype = torch.float32
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True
        )
        
        model.eval()
        return model, tokenizer
    
    def generate(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate a response for a single prompt.
        
        Args:
            prompt: Either a string or a list of chat messages
            max_new_tokens: Override default max tokens
            temperature: Override default temperature
            
        Returns:
            Generated response string
        """
        max_new_tokens = max_new_tokens or self.max_new_tokens
        temperature = temperature or self.temperature
        
        # Handle chat format
        if isinstance(prompt, list):
            input_text = self.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            input_text = prompt
        
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        
        if self.use_gpu and torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if self.do_sample else 1.0,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Decode only the generated part
        input_length = inputs["input_ids"].shape[1]
        generated = outputs[0][input_length:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        
        return response
    
    def evaluate(
        self,
        dataset: Dataset,
        reward_fn: RewardFunction,
        show_examples: int = 3,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate model on a dataset.
        
        Args:
            dataset: Evaluation dataset with 'prompt' and 'ground_truth' columns
            reward_fn: Reward function to compute accuracy
            show_examples: Number of examples to print
            verbose: Whether to print progress
            
        Returns:
            Dictionary with evaluation results
        """
        all_predictions = []
        all_ground_truths = []
        all_responses = []
        
        iterator = tqdm(dataset, desc="Evaluating") if verbose else dataset
        
        for i, example in enumerate(iterator):
            prompt = example["prompt"]
            ground_truth = example["ground_truth"]
            
            # Generate response
            response = self.generate(prompt)
            
            # Store for evaluation
            all_predictions.append([{"role": "assistant", "content": response}])
            all_ground_truths.append(ground_truth)
            all_responses.append(response)
            
            # Print examples
            if verbose and i < show_examples:
                print(f"\n--- Example {i+1} ---")
                if isinstance(prompt, list):
                    user_msg = next(
                        (m["content"] for m in prompt if m["role"] == "user"), 
                        str(prompt)
                    )
                    print(f"Question: {user_msg[:200]}...")
                else:
                    print(f"Prompt: {prompt[:200]}...")
                print(f"Response: {response[:300]}...")
                print(f"Ground Truth: {ground_truth}")
        
        # Compute rewards
        rewards = reward_fn(all_predictions, ground_truth=all_ground_truths)
        
        # Calculate metrics
        accuracy = sum(rewards) / len(rewards)
        
        results = {
            "accuracy": accuracy,
            "num_samples": len(rewards),
            "rewards": rewards,
            "responses": all_responses,
            "ground_truths": all_ground_truths,
        }
        
        if verbose:
            print(f"\n=== Evaluation Results ===")
            print(f"Accuracy: {accuracy:.2%}")
            print(f"Samples: {len(rewards)}")
        
        return results
    
    def batch_generate(
        self,
        prompts: List[Union[str, List[Dict[str, str]]]],
        batch_size: int = 4,
    ) -> List[str]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of prompts
            batch_size: Batch size for generation
            
        Returns:
            List of generated responses
        """
        responses = []
        for prompt in tqdm(prompts, desc="Generating"):
            response = self.generate(prompt)
            responses.append(response)
        return responses


def load_and_evaluate(
    model_path: str,
    dataset: Dataset,
    reward_fn: RewardFunction,
    use_gpu: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to load a model and evaluate it.
    
    Args:
        model_path: Path to model
        dataset: Evaluation dataset
        reward_fn: Reward function
        use_gpu: Whether to use GPU
        **kwargs: Additional arguments for ModelEvaluator
        
    Returns:
        Evaluation results dictionary
    """
    evaluator = ModelEvaluator(
        model_name_or_path=model_path,
        use_gpu=use_gpu,
        **kwargs
    )
    
    return evaluator.evaluate(dataset, reward_fn)
