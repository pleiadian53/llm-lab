"""SFT Trainer Wrapper with PEFT support."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
from datasets import Dataset, load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizer
from trl import SFTConfig, SFTTrainer

from .model_loader import load_model_and_tokenizer


def format_chat_example(example: Dict, tokenizer) -> str:
    """Format a chat example into a string for training.
    
    Handles datasets with 'messages' column (chat format).
    """
    if "messages" in example:
        # Use tokenizer's chat template if available
        try:
            return tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            # Fallback: simple formatting
            text_parts = []
            for msg in example["messages"]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                text_parts.append(f"{role.capitalize()}: {content}")
            return "\n".join(text_parts)
    elif "text" in example:
        return example["text"]
    elif "prompt" in example and "completion" in example:
        return f"{example['prompt']}\n{example['completion']}"
    else:
        raise ValueError(f"Unknown dataset format. Keys: {list(example.keys())}")


@dataclass
class TrainingConfig:
    """Configuration for SFT training.
    
    Attributes:
        learning_rate: Learning rate for training
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        gradient_accumulation_steps: Steps to accumulate gradients
        gradient_checkpointing: Enable gradient checkpointing
        logging_steps: Frequency of logging
        save_steps: Frequency of saving checkpoints
        eval_steps: Frequency of evaluation
        warmup_ratio: Warmup ratio for learning rate scheduler
        weight_decay: Weight decay for optimizer
        max_seq_length: Maximum sequence length
        output_dir: Directory for saving outputs
        report_to: Reporting integrations (e.g., 'wandb', 'tensorboard')
    """
    learning_rate: float = 8e-5
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = False
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_seq_length: int = 2048
    output_dir: str = "./sft_output"
    report_to: str = "none"
    
    def to_sft_config(self) -> SFTConfig:
        """Convert to TRL SFTConfig."""
        return SFTConfig(
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            gradient_checkpointing=self.gradient_checkpointing,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            eval_steps=self.eval_steps,
            warmup_ratio=self.warmup_ratio,
            weight_decay=self.weight_decay,
            max_seq_length=self.max_seq_length,
            output_dir=self.output_dir,
            report_to=self.report_to,
        )


class SFTTrainerWrapper:
    """High-level wrapper for SFT training with PEFT support.
    
    This class provides a simplified interface for supervised fine-tuning
    of language models with optional PEFT (LoRA, VeRA, DoRA, etc.) support.
    
    Example:
        >>> from sft_trainer import SFTTrainerWrapper
        >>> 
        >>> # Full fine-tuning
        >>> trainer = SFTTrainerWrapper(
        ...     model_name="HuggingFaceTB/SmolLM2-135M",
        ...     dataset_name="banghua/DL-SFT-Dataset",
        ... )
        >>> trainer.train()
        >>> 
        >>> # With LoRA
        >>> from sft_trainer.peft import PEFTConfig
        >>> peft_config = PEFTConfig.from_preset("lora_default")
        >>> trainer = SFTTrainerWrapper(
        ...     model_name="HuggingFaceTB/SmolLM2-135M",
        ...     dataset_name="banghua/DL-SFT-Dataset",
        ...     peft_config=peft_config,
        ... )
        >>> trainer.train()
    """
    
    def __init__(
        self,
        model_name: str,
        dataset_name: Optional[str] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        training_config: Optional[TrainingConfig] = None,
        peft_config: Optional[Any] = None,
        use_gpu: bool = False,
        device: Optional[str] = None,
        trust_remote_code: bool = False,
        token: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        """Initialize SFTTrainerWrapper.
        
        Args:
            model_name: HuggingFace model ID or local path
            dataset_name: HuggingFace dataset ID (optional if train_dataset provided)
            train_dataset: Pre-loaded training dataset
            eval_dataset: Pre-loaded evaluation dataset
            training_config: Training configuration
            peft_config: PEFT configuration (from sft_trainer.peft)
            use_gpu: If True, auto-select best available GPU
            device: Explicit device string
            trust_remote_code: Whether to trust remote code
            token: HuggingFace token
            max_samples: Limit dataset size (useful for testing)
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.training_config = training_config or TrainingConfig()
        self.peft_config = peft_config
        self.use_gpu = use_gpu
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.token = token
        self.max_samples = max_samples
        
        self._model = None
        self._tokenizer = None
        self._train_dataset = train_dataset
        self._eval_dataset = eval_dataset
        self._trainer = None
        self._is_peft_model = False
    
    def load_model(self) -> None:
        """Load model and tokenizer."""
        print(f"Loading model: {self.model_name}")
        self._model, self._tokenizer = load_model_and_tokenizer(
            self.model_name,
            use_gpu=self.use_gpu,
            device=self.device,
            trust_remote_code=self.trust_remote_code,
            token=self.token,
        )
        
        # Apply PEFT if configured
        if self.peft_config is not None:
            self._apply_peft()
    
    def _apply_peft(self) -> None:
        """Apply PEFT adapter to the model."""
        from peft import get_peft_model, prepare_model_for_kbit_training
        
        print(f"Applying PEFT: {self.peft_config.method.value}")
        
        # Get the PEFT library config
        peft_lib_config = self.peft_config.to_peft_config()
        
        # Prepare for quantized training if using QLoRA
        if self.peft_config.method.value == "qlora":
            self._model = prepare_model_for_kbit_training(self._model)
        
        # Apply PEFT
        self._model = get_peft_model(self._model, peft_lib_config)
        self._is_peft_model = True
        
        # Print trainable parameters
        self._model.print_trainable_parameters()
    
    def load_dataset(self) -> None:
        """Load training dataset."""
        if self._train_dataset is not None:
            print("Using provided training dataset")
        elif self.dataset_name:
            print(f"Loading dataset: {self.dataset_name}")
            dataset = load_dataset(self.dataset_name)
            self._train_dataset = dataset["train"]
            if "test" in dataset:
                self._eval_dataset = dataset["test"]
            elif "validation" in dataset:
                self._eval_dataset = dataset["validation"]
        else:
            raise ValueError("Either dataset_name or train_dataset must be provided")
        
        # Limit samples if specified
        if self.max_samples and len(self._train_dataset) > self.max_samples:
            print(f"Limiting dataset to {self.max_samples} samples")
            self._train_dataset = self._train_dataset.select(range(self.max_samples))
        
        print(f"Training samples: {len(self._train_dataset)}")
        if self._eval_dataset:
            print(f"Evaluation samples: {len(self._eval_dataset)}")
    
    def setup_trainer(self) -> None:
        """Set up the SFTTrainer."""
        if self._model is None:
            self.load_model()
        if self._train_dataset is None:
            self.load_dataset()
        
        sft_config = self.training_config.to_sft_config()
        
        # Create formatting function that captures tokenizer
        tokenizer = self._tokenizer
        def formatting_func(example):
            return format_chat_example(example, tokenizer)
        
        # Check if dataset needs formatting (has 'messages' but no 'text')
        sample = self._train_dataset[0]
        needs_formatting = "messages" in sample and "text" not in sample
        
        trainer_kwargs = {
            "model": self._model,
            "args": sft_config,
            "train_dataset": self._train_dataset,
            "eval_dataset": self._eval_dataset,
            "processing_class": self._tokenizer,
        }
        
        if needs_formatting:
            trainer_kwargs["formatting_func"] = formatting_func
        
        self._trainer = SFTTrainer(**trainer_kwargs)
    
    def train(self) -> Dict[str, float]:
        """Run training.
        
        Returns:
            Training metrics dictionary
        """
        if self._trainer is None:
            self.setup_trainer()
        
        print("Starting training...")
        result = self._trainer.train()
        print("Training complete!")
        
        return result.metrics
    
    def save_model(self, output_dir: Optional[str] = None) -> None:
        """Save the trained model.
        
        Args:
            output_dir: Directory to save to (defaults to training_config.output_dir)
        """
        save_dir = output_dir or self.training_config.output_dir
        
        if self._is_peft_model:
            # Save only the adapter
            print(f"Saving PEFT adapter to {save_dir}")
            self._model.save_pretrained(save_dir)
        else:
            # Save full model
            print(f"Saving full model to {save_dir}")
            self._trainer.save_model(save_dir)
        
        # Save tokenizer
        self._tokenizer.save_pretrained(save_dir)
        print(f"Model saved to {save_dir}")
    
    def push_to_hub(self, repo_id: str, private: bool = True) -> None:
        """Push model to HuggingFace Hub.
        
        Args:
            repo_id: Repository ID on HuggingFace Hub
            private: Whether to make the repo private
        """
        print(f"Pushing model to {repo_id}...")
        self._trainer.push_to_hub(repo_id=repo_id, private=private)
        print(f"Model pushed to {repo_id}")
    
    @property
    def model(self) -> Optional[PreTrainedModel]:
        """Get the model."""
        return self._model
    
    @property
    def tokenizer(self) -> Optional[PreTrainedTokenizer]:
        """Get the tokenizer."""
        return self._tokenizer
    
    @property
    def trainer(self) -> Optional[SFTTrainer]:
        """Get the underlying SFTTrainer."""
        return self._trainer
