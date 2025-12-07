"""Configuration for post-training experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from pydantic import BaseModel, Field, model_validator


class PreferenceDatasetConfig(BaseModel):
    """Preference data specification."""

    dataset_name: str = Field(
        "Anthropic/hh-rlhf", description="Preference dataset name."
    )
    split: str = Field("train", description="Split to load.")
    text_key_chosen: str = Field(
        "chosen", description="Column containing preferred response."
    )
    text_key_rejected: str = Field(
        "rejected", description="Column containing rejected response."
    )


class RewardModelConfig(BaseModel):
    """Reward model parameters."""

    base_model_name: str = Field(
        "distilbert-base-uncased",
        description="Base transformer used for reward modeling.",
    )
    learning_rate: float = Field(1e-5, description="Optimizer learning rate.")
    batch_size: int = Field(8, description="Batch size for fine-tuning.")
    num_epochs: int = Field(1, description="Epochs for quick smoke testing.")


class AlignmentConfig(BaseModel):
    """Alignment strategy configuration."""

    strategy: str = Field("dpo", description="Alignment algorithm (dpo/ppo/rm).")
    kl_coefficient: float = Field(0.1, description="KL penalty for RLHF variants.")


class PostTrainingConfig(BaseModel):
    """Aggregate configuration for post-training alignment."""

    output_dir: Path = Field(
        default_factory=lambda: Path.cwd() / "artifacts" / "posttraining",
        description="Directory for alignment outputs.",
    )
    reward_model: RewardModelConfig = Field(default_factory=RewardModelConfig)
    dataset: PreferenceDatasetConfig = Field(default_factory=PreferenceDatasetConfig)
    alignment: AlignmentConfig = Field(default_factory=AlignmentConfig)

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="after")
    def _ensure_output_dir(self) -> PostTrainingConfig:
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        return self

    def to_dict(self) -> dict[str, Any]:
        conf = OmegaConf.create(self.dict())
        result = OmegaConf.to_container(conf, resolve=True)
        return dict(result)  # type: ignore[arg-type]

    @classmethod
    def from_file(cls, config_path: Path) -> PostTrainingConfig:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        conf = OmegaConf.load(config_path)
        data = OmegaConf.to_container(conf, resolve=True)
        return cls.model_validate(data)
