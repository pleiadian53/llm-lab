"""
Model Service Module

Provides the ServeLLM class for loading and running language models
with proper memory management, retry logic, and device selection.
"""

import os
import torch
import gc
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM


class ServeLLM:
    """
    A service class for loading and running language models with proper memory management.
    
    Features:
        - Automatic device selection (CUDA/CPU)
        - Retry logic for network issues
        - Context manager support for automatic cleanup
        - Memory-efficient inference
    
    Example:
        >>> with ServeLLM("deepseek-ai/deepseek-math-7b-base") as llm:
        ...     response = llm.generate_response("What is 2+2?")
        ...     print(response)
    """
    
    def __init__(self, model_name: str, device: str = "auto"):
        """
        Initialize the ServeLLM instance.
        
        Args:
            model_name: Name/path of the model to load (HuggingFace model ID or local path)
            device: Device to load model on ('auto', 'cuda', 'mps', 'cpu')
                   'auto' will automatically select the best available device
        """
        self.model_name = model_name
        self.device = self._determine_device(device)
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _determine_device(self, device: str) -> str:
        """
        Determine the best device to use for model inference.
        
        Args:
            device: Requested device ('auto', 'cuda', 'mps', 'cpu')
            
        Returns:
            str: Selected device name
        """
        if device == "auto":
            # Check for CUDA (NVIDIA) or ROCm (AMD) availability
            if torch.cuda.is_available():
                return "cuda"
            # Check for Apple Silicon MPS
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device
    
    def _load_model(self, max_retries: int = 3):
        """
        Load the tokenizer and model with retry logic for network issues.
        
        Args:
            max_retries: Maximum number of retry attempts for network failures
        """
        import time
        
        print(f"Loading {self.model_name}...")
        
        # Load tokenizer with retry logic
        for attempt in range(max_retries):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                break  # Success, exit retry loop
            except Exception as e:
                error_msg = str(e)
                if "couldn't connect" in error_msg or "connection" in error_msg.lower() or "offline" in error_msg.lower():
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 10
                        print(f"⚠️  Tokenizer download failed (attempt {attempt + 1}/{max_retries})")
                        print(f"   Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"❌ Failed to load tokenizer after {max_retries} attempts")
                        print(f"   Error: {error_msg}")
                        raise
                else:
                    # Non-connection error, raise immediately
                    print(f"❌ Error loading tokenizer: {error_msg}")
                    raise
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with retry logic
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
        }
        
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"
        
        for attempt in range(max_retries):
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
                break  # Success, exit retry loop
            except Exception as e:
                error_msg = str(e)
                if "couldn't connect" in error_msg or "connection" in error_msg.lower() or "offline" in error_msg.lower():
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 10
                        print(f"⚠️  Model download failed (attempt {attempt + 1}/{max_retries})")
                        print(f"   Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"❌ Failed to load model after {max_retries} attempts")
                        print(f"   Error: {error_msg}")
                        raise
                else:
                    # Non-connection error, raise immediately
                    print(f"❌ Error loading model: {error_msg}")
                    raise
        
        if self.device in ["cpu", "mps"]:
            self.model = self.model.to(self.device)
            
        print(f"✅ Model loaded successfully on {self.device}")
    
    def generate_response(
        self, 
        prompt: str, 
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True, 
    ) -> str:
        """
        Generate a response to the given prompt.
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling (False = greedy decoding)
            
        Returns:
            Generated response text (excluding the input prompt)
            
        Raises:
            RuntimeError: If model is not loaded
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode response (exclude input tokens)
            response_tokens = outputs[0][inputs.input_ids.shape[1]:]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return f"Error: {str(e)}"
    
    def cleanup(self):
        """Clean up model and free GPU/CPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        print("🧹 Model cleaned up and memory freed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.cleanup()
    
    @staticmethod
    def cleanup_all():
        """Static method to clean up all GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("🧹 All GPU memory cleaned up")
