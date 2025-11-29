"""Simple Typer-based command line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from llm_lab.posttrain_llm.alignment import PostTrainingPipeline
from llm_lab.posttrain_llm.config import PostTrainingConfig
from llm_lab.pretrain_llm.config import PretrainingConfig
from llm_lab.pretrain_llm.trainer import LanguageModelTrainer

app = typer.Typer(help="Command line utilities for LLM Lab.")


@app.command()
def pretrain(
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Optional path to a Hydra YAML configuration file.",
    ),
) -> None:
    """Run a lightweight pre-training experiment."""
    config = PretrainingConfig.from_file(config_path) if config_path else PretrainingConfig()
    trainer = LanguageModelTrainer.from_config(config)
    trainer.run()


@app.command()
def posttrain(
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Optional path to a Hydra YAML configuration file.",
    ),
) -> None:
    """Run a post-training alignment experiment."""
    config = PostTrainingConfig.from_file(config_path) if config_path else PostTrainingConfig()
    pipeline = PostTrainingPipeline.from_config(config)
    pipeline.run()


if __name__ == "__main__":
    app()
