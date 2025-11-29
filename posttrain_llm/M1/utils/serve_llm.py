import time
import random
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 42

import torch
import random
import time
import gc

class ServeLLM:
    def __init__(self, model_name, seed=42, device="auto", dtype=torch.float16, quantize=None):
        """
        Initialize LLM serving wrapper with multi-backend support.
        
        Args:
            model_name: HuggingFace model path or local path
            seed: Random seed for reproducibility
            device: Device to use - "auto", "cuda", "mps", or "cpu"
                   "auto" will select best available: cuda > mps > cpu
            dtype: Model dtype (torch.float16, torch.float32, torch.bfloat16)
            quantize: Quantization config - None, "4bit", or "8bit"
        """
        self.model_name = model_name
        self.seed = seed
        self.dtype = dtype
        self.quantize = quantize
        self.model = None
        self.tokenizer = None
        self._initialized = False
        
        # Auto-detect best available device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            # Explicit device specified - validate and use
            if device == "cuda" and not torch.cuda.is_available():
                print(f"Warning: CUDA requested but not available. Falling back to CPU.")
                self.device = "cpu"
            elif device == "mps" and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                print(f"Warning: MPS requested but not available. Falling back to CPU.")
                self.device = "cpu"
            else:
                self.device = device
        
        print(f"Using device: {self.device}")
        
        # non with use
        if not hasattr(self, "_in_context"):
            self._initialize()
    
    def _initialize(self):
        if self._initialized:
            return
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        self.model, self.tokenizer = self.load_model()
        self._initialized = True
    
    def __enter__(self):
        self._in_context = True
        self._initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
    
    def load_model(self):
        print(f"Loading model: {self.model_name}")
        print(f"Device: {self.device}, Quantization: {self.quantize or 'None'}")
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Prepare model loading kwargs
        model_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,  # Important for memory efficiency
        }
        
        # Handle quantization (requires bitsandbytes)
        if self.quantize == "4bit":
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self.dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                print("Using 4-bit quantization")
            except ImportError:
                print("Warning: bitsandbytes not installed. Install with: pip install bitsandbytes")
                print("Falling back to non-quantized loading")
                model_kwargs["torch_dtype"] = self.dtype
        elif self.quantize == "8bit":
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True
                )
                print("Using 8-bit quantization")
            except ImportError:
                print("Warning: bitsandbytes not installed. Install with: pip install bitsandbytes")
                print("Falling back to non-quantized loading")
                model_kwargs["torch_dtype"] = self.dtype
        else:
            model_kwargs["torch_dtype"] = self.dtype
        
        # Handle device mapping
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Move to device if not using CUDA device_map or quantization
        if self.device != "cuda" and self.quantize is None:
            print(f"Moving model to {self.device}...")
            model = model.to(self.device)
        
        model.eval()
        print("Model loaded successfully!")
        return model, tokenizer
    
    def generate_response(self, prompts, temperature=0.0, top_p=1.0, max_tokens=100):
        if isinstance(prompts, str):
            prompts = [prompts]
        
        responses = []
        
        for prompt in prompts:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else 1e-7,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode only the generated part
            input_length = inputs['input_ids'].shape[1]
            generated_ids = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            responses.append(response)
        
        # Return single string if input was single string
        if len(responses) == 1:
            return responses[0]
        return responses
    
    def cleanup(self):
        print("Cleaning up memory...")
        try:
            del self.model
            del self.tokenizer
            gc.collect()
            
            # Clear cache for all supported backends
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            time.sleep(2)
            print("Cleanup complete!")
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    @staticmethod
    def cleanup_all():
        """Static method to clean up all GPU/MPS memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print("All device memory cleared.")




if __name__ == "__main__":
    base_model = "deepseek-ai/deepseek-math-7b-base"
    sft_model = "deepseek-ai/deepseek-math-7b-instruct"
    rl_model = "deepseek-ai/deepseek-math-7b-rl"

    llm = ServeLLM(sft_model)
    while True:
        print("--------------------------------")
        print("\nPlease input your Prompt (type 'END' on a new line to finish, or 'exit'/'quit' to break the loop):")
        
        lines = []
        while True:
            line = input()
            if line.lower() in ["exit", "quit"]:
                print("Exiting bot...")
                llm.cleanup()
                exit()
            if line.strip() == "END":
                break
            lines.append(line)
        
        if not lines:
            print("No input provided. Please try again.")
            continue
            
        user_input = "\n".join(lines)
        print(f"\nProcessing prompt:\n{user_input}\n")
        
        response = llm.generate_response(user_input, temperature=0.0, top_p=1.0, max_tokens=5000)
        print(f"Result: {response}")
    llm.cleanup()

