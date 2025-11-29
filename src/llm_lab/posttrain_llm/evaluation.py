"""Reward model evaluation helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from llm_lab.posttrain_llm.config import PostTrainingConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class RewardModelEvaluator:
    """Wraps a lightweight reward scoring model with safe fallbacks."""

    config: PostTrainingConfig
    _model: AutoModelForSequenceClassification | None = None
    _tokenizer: AutoTokenizer | None = None

    def __post_init__(self) -> None:
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.reward_model.base_model_name
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.config.reward_model.base_model_name
            )
        except Exception as exc:  # pragma: no cover - network dependent
            LOGGER.warning(
                "Falling back to lexical reward model because pretrained model load failed: %s",
                exc,
            )
            self._tokenizer = None
            self._model = None

    def score_pair(self, chosen: str, rejected: str) -> float:
        """Return reward advantage for a chosen vs rejected response."""
        if self._model is None or self._tokenizer is None:
            return self._lexical_score(chosen, rejected)
        inputs = self._tokenizer(
            [chosen, rejected],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self._model(**inputs).logits.squeeze()
        if outputs.dim() == 0:
            chosen_score, rejected_score = outputs, torch.tensor(0.0)
        else:
            chosen_score, rejected_score = outputs[0], outputs[1]
        return float(chosen_score - rejected_score)

    @staticmethod
    def _lexical_score(chosen: str, rejected: str) -> float:
        chosen_tokens = set(word.lower() for word in chosen.split())
        rejected_tokens = set(word.lower() for word in rejected.split())
        return float(len(chosen_tokens) - len(rejected_tokens))
