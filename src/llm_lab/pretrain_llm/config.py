"""Configuration objects for language model pre-training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from hydra.utils import instantiate
from omegaconf import OmegaConf
from pydantic import BaseModel, Field, validator


class DataConfig(BaseModel):
    """Dataset configuration."""

    dataset_name: str = Field("wikitext", description="Hugging Face dataset identifier.")
    dataset_config: str = Field("wikitext-2-raw-v1", description="Dataset config name.")
    text_column: str = Field("text", description="Column containing text data.")
    streaming: bool = Field(False, description="Enable streaming mode for large datasets.")


class ModelConfig(BaseModel):
    """Model definition."""

    pretrained_model_name: str = Field("gpt2", description="Hugging Face model identifier.")
    gradient_checkpointing: bool = Field(
        True, description="Enable gradient checkpointing for larger models."
    )


class OptimizerConfig(BaseModel):
    """Optimizer hyper-parameters."""

    learning_rate: float = Field(5e-5, description="Learning rate for the optimizer.")
    weight_decay: float = Field(0.01, description="Weight decay factor.")
    betas: tuple[float, float] = Field((0.9, 0.95), description="Adam beta parameters.")


class TrainerConfig(BaseModel):
    """Training loop parameters."""

    num_train_steps: int = Field(1000, description="Number of optimizer steps.")
    gradient_accumulation_steps: int = Field(
        1, description="Number of gradient accumulation steps."
    )
    eval_interval: int = Field(200, description="How often to run evaluation.")
    log_interval: int = Field(50, description="How often to emit logs.")
    precision: str = Field("bf16", description="Training precision (fp32, bf16, etc.).")


class PretrainingConfig(BaseModel):
    """Aggregate configuration for pre-training experiments."""

    seed: int = Field(17, description="Random seed for reproducibility.")
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    output_dir: Path = Field(
        default_factory=lambda: Path.cwd() / "artifacts" / "pretraining",
        description="Directory to store checkpoints and logs.",
    )

    class Config:
        arbitrary_types_allowed = True

    @validator("output_dir")
    def _ensure_output_dir(cls, value: Path) -> Path:
        value.mkdir(parents=True, exist_ok=True)
        return value

    def to_dict(self) -> Dict[str, Any]:
        """Return a dict representation compatible with Hydra/OmegaConf."""
        return OmegaConf.to_container(OmegaConf.create(self.dict()), resolve=True)  # type: ignore[return-value]

    @classmethod
    def from_file(cls, config_path: Path) -> "PretrainingConfig":
        """Create a config from a YAML file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        conf = OmegaConf.load(config_path)
        data = OmegaConf.to_container(conf, resolve=True)
        return cls.parse_obj(data)

    def instantiate(self, target: str, **overrides: Any) -> Any:
        """Instantiate a Hydra object using this configuration."""
        cfg = OmegaConf.create(self.to_dict())
        cfg._target_ = target
        cfg.update(overrides)
        return instantiate(cfg)
