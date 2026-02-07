"""Model Loading Utilities for SFT Training."""

import os
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


def _get_default_chat_template() -> str:
    """Return a simple default chat template."""
    lines = [
        "{% for message in messages %}",
        "{% if message['role'] == 'system' %}System: {{ message['content'] }}",
        "{% elif message['role'] == 'user' %}User: {{ message['content'] }}",
        "{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}",
        "{% endif %}",
        "{% endfor %}",
    ]
    return "\n".join(lines)


def determine_device(use_gpu: bool = False, device: Optional[str] = None) -> str:
    """Determine the target device for model loading."""
    if device is not None:
        return device
    
    if use_gpu:
        if torch.cuda.is_available():
            print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            return "cuda"
        elif torch.backends.mps.is_available():
            print("Using Apple Silicon GPU (MPS)")
            return "mps"
        else:
            print("No GPU available, using CPU")
            return "cpu"
    
    print("Using CPU (use_gpu=False)")
    return "cpu"


def load_model_and_tokenizer(
    model_name: str,
    use_gpu: bool = False,
    device: Optional[str] = None,
    trust_remote_code: bool = False,
    token: Optional[str] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load model and tokenizer with device support."""
    # Resolve relative paths
    if model_name.startswith('./') or model_name.startswith('../'):
        model_name = os.path.abspath(model_name)
        print(f"Resolved local model path: {model_name}")
    
    target_device = determine_device(use_gpu, device)
    
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code, token=token,
    )
    
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    if not tokenizer.chat_template:
        tokenizer.chat_template = _get_default_chat_template()
    
    print(f"Loading model from {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=trust_remote_code, token=token,
    )
    
    model.to(target_device)
    print(f"Model loaded on device: {target_device}")
    
    return model, tokenizer


class ModelLoader:
    """Class-based model loader with additional features."""
    
    def __init__(
        self,
        model_name: str,
        use_gpu: bool = False,
        device: Optional[str] = None,
        trust_remote_code: bool = False,
        token: Optional[str] = None,
    ):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.token = token
        self._model = None
        self._tokenizer = None
    
    def load(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load and return model and tokenizer."""
        self._model, self._tokenizer = load_model_and_tokenizer(
            self.model_name,
            use_gpu=self.use_gpu,
            device=self.device,
            trust_remote_code=self.trust_remote_code,
            token=self.token,
        )
        return self._model, self._tokenizer
    
    @property
    def model(self) -> Optional[PreTrainedModel]:
        return self._model
    
    @property
    def tokenizer(self) -> Optional[PreTrainedTokenizer]:
        return self._tokenizer
