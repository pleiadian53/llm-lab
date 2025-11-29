"""Shared experiment configuration primitives."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, validator


class ExperimentPaths(BaseModel):
    """Convenience container for common experiment directories."""

    project_root: Path = Field(
        default_factory=lambda: Path.cwd(),
        description="Root directory of the project repository.",
    )
    data_dir: Path = Field(
        default_factory=lambda: Path.cwd() / "data",
        description="Directory that stores raw and processed datasets.",
    )
    artifacts_dir: Path = Field(
        default_factory=lambda: Path.cwd() / "artifacts",
        description="Directory used for model checkpoints and logs.",
    )

    class Config:
        arbitrary_types_allowed = True

    @validator("data_dir", "artifacts_dir")
    def _ensure_exists(cls, value: Path) -> Path:
        """Create directories on demand to avoid missing-path errors."""
        value.mkdir(parents=True, exist_ok=True)
        return value
