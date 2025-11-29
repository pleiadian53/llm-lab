"""Model and tokenizer loading utilities with smart caching and download management."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

LOGGER = logging.getLogger(__name__)


class ModelLoader:
    """Handles model and tokenizer loading with automatic download and caching."""

    def __init__(
        self,
        model_name_or_path: str,
        local_cache_dir: Optional[Path] = None,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        use_fast_tokenizer: bool = True,
    ) -> None:
        """
        Initialize the model loader.

        Args:
            model_name_or_path: HuggingFace model identifier or local path
            local_cache_dir: Optional local directory to cache models
            device_map: Device mapping strategy ('auto', 'cpu', 'cuda', etc.)
            torch_dtype: Torch dtype for model weights
            use_fast_tokenizer: Whether to use fast tokenizer implementation
        """
        self.model_name_or_path = model_name_or_path
        self.local_cache_dir = Path(local_cache_dir) if local_cache_dir else None
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.use_fast_tokenizer = use_fast_tokenizer

    def _get_load_path(self) -> str:
        """Determine the path to load from (local cache or HF Hub)."""
        if self.local_cache_dir and self.local_cache_dir.exists():
            LOGGER.info("Loading from local cache: %s", self.local_cache_dir)
            return str(self.local_cache_dir)
        
        # Check if it's already a local path
        if Path(self.model_name_or_path).exists():
            LOGGER.info("Loading from existing local path: %s", self.model_name_or_path)
            return self.model_name_or_path
        
        LOGGER.info("Will download from HuggingFace Hub: %s", self.model_name_or_path)
        return self.model_name_or_path

    def load_model(self) -> PreTrainedModel:
        """
        Load the model with smart caching.

        Returns:
            Loaded pretrained model
        """
        load_path = self._get_load_path()
        
        LOGGER.info("Loading model from: %s", load_path)
        model = AutoModelForCausalLM.from_pretrained(
            load_path,
            device_map=self.device_map,
            torch_dtype=self.torch_dtype,
            trust_remote_code=False,  # Security best practice
        )
        
        # Save to local cache if downloading from HF Hub
        if self.local_cache_dir and not self.local_cache_dir.exists():
            LOGGER.info("Saving model to local cache: %s", self.local_cache_dir)
            self.local_cache_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(self.local_cache_dir)
        
        return model

    def load_tokenizer(self) -> PreTrainedTokenizerBase:
        """
        Load the tokenizer with smart caching.

        Returns:
            Loaded tokenizer
        """
        load_path = self._get_load_path()
        
        LOGGER.info("Loading tokenizer from: %s", load_path)
        tokenizer = AutoTokenizer.from_pretrained(
            load_path,
            use_fast=self.use_fast_tokenizer,
            trust_remote_code=False,
        )
        
        # Save to local cache if downloading from HF Hub
        if self.local_cache_dir and not self.local_cache_dir.exists():
            LOGGER.info("Saving tokenizer to local cache: %s", self.local_cache_dir)
            self.local_cache_dir.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(self.local_cache_dir)
        
        return tokenizer

    def load_model_and_tokenizer(self) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """
        Load both model and tokenizer together.

        Returns:
            Tuple of (model, tokenizer)
        """
        model = self.load_model()
        tokenizer = self.load_tokenizer()
        
        # Set pad token if not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            LOGGER.info("Set pad_token to eos_token")
        
        return model, tokenizer


def download_model_manually(
    model_name: str,
    output_dir: Path,
    include_tokenizer: bool = True,
) -> None:
    """
    Manually download a model from HuggingFace Hub to a local directory.

    This is useful for pre-downloading models before running experiments,
    especially in offline or restricted network environments.

    Args:
        model_name: HuggingFace model identifier (e.g., 'upstage/TinySolar-248m-4k')
        output_dir: Local directory to save the model
        include_tokenizer: Whether to also download the tokenizer

    Example:
        >>> download_model_manually(
        ...     "upstage/TinySolar-248m-4k",
        ...     Path("./models/TinySolar-248m-4k")
        ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    LOGGER.info("Downloading model '%s' to %s", model_name, output_dir)
    
    # Download model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.save_pretrained(output_dir)
    LOGGER.info("Model saved to %s", output_dir)
    
    # Download tokenizer
    if include_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(output_dir)
        LOGGER.info("Tokenizer saved to %s", output_dir)
    
    LOGGER.info("Download complete!")


def load_model_with_fallback(
    model_name: str,
    local_dir: Optional[Path] = None,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load model with automatic fallback between local and remote sources.

    This is the recommended high-level function for most use cases.

    Args:
        model_name: HuggingFace model identifier or local path
        local_dir: Optional local cache directory
        device_map: Device mapping strategy
        torch_dtype: Torch dtype for model weights

    Returns:
        Tuple of (model, tokenizer)

    Example:
        >>> model, tokenizer = load_model_with_fallback(
        ...     "upstage/TinySolar-248m-4k",
        ...     local_dir=Path("./models/TinySolar-248m-4k")
        ... )
    """
    loader = ModelLoader(
        model_name_or_path=model_name,
        local_cache_dir=local_dir,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    return loader.load_model_and_tokenizer()
