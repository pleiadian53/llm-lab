"""DPO Trainer Wrapper for Direct Preference Optimization."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from datasets import Dataset, load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizer
from trl import DPOConfig, DPOTrainer

from .model_loader import load_model_and_tokenizer


@dataclass
class DPOTrainingConfig:
    """Configuration for DPO training.
    
    Attributes:
        beta: DPO beta parameter (controls deviation from reference model)
        learning_rate: Learning rate for training
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        gradient_accumulation_steps: Steps to accumulate gradients
        logging_steps: Frequency of logging
        save_steps: Frequency of saving checkpoints
        warmup_ratio: Warmup ratio for learning rate scheduler
        weight_decay: Weight decay for optimizer
        max_length: Maximum sequence length
        max_prompt_length: Maximum prompt length
        output_dir: Directory for saving outputs
        report_to: Reporting integrations (e.g., 'wandb', 'tensorboard')
    """
    beta: float = 0.2
    learning_rate: float = 5e-5
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    logging_steps: int = 2
    save_steps: int = 500
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_length: int = 1024
    max_prompt_length: int = 512
    output_dir: str = "./dpo_output"
    report_to: str = "none"
    
    def to_dpo_config(self) -> DPOConfig:
        """Convert to TRL DPOConfig."""
        return DPOConfig(
            beta=self.beta,
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            warmup_ratio=self.warmup_ratio,
            weight_decay=self.weight_decay,
            max_length=self.max_length,
            max_prompt_length=self.max_prompt_length,
            output_dir=self.output_dir,
            report_to=self.report_to,
        )


class DPOTrainerWrapper:
    """High-level wrapper for DPO training.
    
    This class provides a simplified interface for Direct Preference Optimization
    training of language models.
    
    DPO trains models to prefer "chosen" responses over "rejected" responses
    without needing an explicit reward model.
    
    Example - Basic DPO Training:
        >>> from dpo_trainer import DPOTrainerWrapper
        >>> 
        >>> trainer = DPOTrainerWrapper(
        ...     model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        ...     dataset_name="banghua/DL-DPO-Dataset",
        ... )
        >>> trainer.train()
    
    Example - Custom Dataset:
        >>> from dpo_trainer import DPOTrainerWrapper, build_identity_shift_dataset
        >>> 
        >>> # Build custom preference dataset
        >>> dataset = build_identity_shift_dataset(
        ...     model, tokenizer,
        ...     original_name="Qwen",
        ...     new_name="Deep Qwen",
        ... )
        >>> 
        >>> trainer = DPOTrainerWrapper(
        ...     model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        ...     train_dataset=dataset,
        ... )
        >>> trainer.train()
    
    Example - With Custom Config:
        >>> config = DPOTrainingConfig(
        ...     beta=0.1,
        ...     learning_rate=1e-5,
        ...     num_train_epochs=3,
        ... )
        >>> trainer = DPOTrainerWrapper(
        ...     model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        ...     dataset_name="banghua/DL-DPO-Dataset",
        ...     training_config=config,
        ... )
        >>> trainer.train()
    """
    
    def __init__(
        self,
        model_name: str,
        dataset_name: Optional[str] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        training_config: Optional[DPOTrainingConfig] = None,
        ref_model: Optional[PreTrainedModel] = None,
        use_gpu: bool = False,
        device: Optional[str] = None,
        trust_remote_code: bool = False,
        token: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        """Initialize DPOTrainerWrapper.
        
        Args:
            model_name: HuggingFace model ID or local path
            dataset_name: HuggingFace dataset ID (optional if train_dataset provided)
            train_dataset: Pre-loaded training dataset with 'chosen' and 'rejected' columns
            eval_dataset: Pre-loaded evaluation dataset
            training_config: DPO training configuration
            ref_model: Reference model for DPO (None = implicit reference)
            use_gpu: If True, auto-select best available GPU
            device: Explicit device string
            trust_remote_code: Whether to trust remote code
            token: HuggingFace token
            max_samples: Limit dataset size (useful for testing)
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.training_config = training_config or DPOTrainingConfig()
        self.ref_model = ref_model
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
        """Set up the DPOTrainer."""
        if self._model is None:
            self.load_model()
        if self._train_dataset is None:
            self.load_dataset()
        
        dpo_config = self.training_config.to_dpo_config()
        
        self._trainer = DPOTrainer(
            model=self._model,
            ref_model=self.ref_model,
            args=dpo_config,
            processing_class=self._tokenizer,
            train_dataset=self._train_dataset,
            eval_dataset=self._eval_dataset,
        )
    
    def train(self) -> Dict[str, float]:
        """Run DPO training.
        
        Returns:
            Training metrics dictionary
        """
        if self._trainer is None:
            self.setup_trainer()
        
        print("Starting DPO training...")
        print(f"  Beta: {self.training_config.beta}")
        print(f"  Learning rate: {self.training_config.learning_rate}")
        print(f"  Epochs: {self.training_config.num_train_epochs}")
        print()
        
        result = self._trainer.train()
        print("DPO training complete!")
        
        return result.metrics
    
    def save_model(self, output_dir: Optional[str] = None) -> None:
        """Save the trained model.
        
        Args:
            output_dir: Directory to save to (defaults to training_config.output_dir)
        """
        save_dir = output_dir or self.training_config.output_dir
        
        print(f"Saving model to {save_dir}")
        self._trainer.save_model(save_dir)
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
    def trainer(self) -> Optional[DPOTrainer]:
        """Get the underlying DPOTrainer."""
        return self._trainer
