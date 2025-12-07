"""High-level pipeline for post-training alignment experiments."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset

from llm_lab.posttrain_llm.config import PostTrainingConfig
from llm_lab.posttrain_llm.evaluation import RewardModelEvaluator

LOGGER = logging.getLogger(__name__)


def _load_preferences(config: PostTrainingConfig) -> Iterable[dict[Any, Any]]:
    try:
        dataset = load_dataset(
            config.dataset.dataset_name,
            split=config.dataset.split,
        )
        return dataset  # type: ignore[no-any-return]
    except Exception as exc:  # pragma: no cover - network dependent
        LOGGER.warning(
            "Falling back to synthetic preference dataset because load_dataset failed: %s",
            exc,
        )
        return [
            {
                config.dataset.text_key_chosen: "Thank you for asking! Here is a helpful reply.",
                config.dataset.text_key_rejected: "No.",
            },
            {
                config.dataset.text_key_chosen: "Certainly, let's walk through the solution.",
                config.dataset.text_key_rejected: "I will not help you.",
            },
        ]


@dataclass
class PostTrainingPipeline:
    """Coordinates preference data, reward modeling, and evaluation."""

    config: PostTrainingConfig
    evaluator: RewardModelEvaluator
    artifacts_dir: Path

    @classmethod
    def from_config(cls, config: PostTrainingConfig) -> "PostTrainingPipeline":
        artifacts_dir = config.output_dir
        evaluator = RewardModelEvaluator(config=config)
        return cls(config=config, evaluator=evaluator, artifacts_dir=artifacts_dir)

    def run(self) -> None:
        """Execute a placeholder post-training routine with metrics logging."""
        preferences = _load_preferences(self.config)
        scores = []
        for sample in preferences:
            chosen = sample[self.config.dataset.text_key_chosen]
            rejected = sample[self.config.dataset.text_key_rejected]
            score = self.evaluator.score_pair(chosen, rejected)
            scores.append(score)
        if not scores:
            LOGGER.warning("No preference samples were processed.")
            return

        summary = {
            "mean_score": sum(scores) / len(scores),
            "num_samples": len(scores),
            "strategy": self.config.alignment.strategy,
        }
        self._write_summary(summary)
        LOGGER.info("Post-training pipeline finished with summary: %s", summary)

    def _write_summary(self, summary: dict) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        summary_path = self.artifacts_dir / "posttraining_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
