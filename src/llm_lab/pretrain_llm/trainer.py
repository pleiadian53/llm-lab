"""Training loop primitives for autoregressive language models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import cycle
from typing import Iterator

import torch
from torch.optim import AdamW

from llm_lab.pretrain_llm.config import PretrainingConfig
from llm_lab.pretrain_llm.data import PretrainingDataModule
from llm_lab.pretrain_llm.model import AutoregressiveLM

LOGGER = logging.getLogger(__name__)


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class LanguageModelTrainer:
    """Co-ordinates model, optimizer, and data loader."""

    config: PretrainingConfig
    model_bundle: AutoregressiveLM
    data_module: PretrainingDataModule
    device: torch.device

    @classmethod
    def from_config(cls, config: PretrainingConfig) -> "LanguageModelTrainer":
        model_bundle = AutoregressiveLM.from_config(config)
        data_module = PretrainingDataModule(config=config, tokenizer=model_bundle.tokenizer)
        device = _resolve_device()
        model_bundle.to(device)
        return cls(
            config=config,
            model_bundle=model_bundle,
            data_module=data_module,
            device=device,
        )

    def _build_optimizer(self) -> AdamW:
        return AdamW(
            params=self.model_bundle.model.parameters(),
            lr=self.config.optimizer.learning_rate,
            betas=self.config.optimizer.betas,
            weight_decay=self.config.optimizer.weight_decay,
        )

    def run(self) -> None:
        """Execute a tiny supervised fine-tuning loop suitable for smoke tests."""
        torch.manual_seed(self.config.seed)
        optimizer = self._build_optimizer()
        dataloader = self.data_module.build_dataloader()
        batches: Iterator = cycle(dataloader)

        for step in range(1, self.config.trainer.num_train_steps + 1):
            batch = next(batches)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model_bundle.forward(batch)
            loss = outputs["loss"]
            loss.backward()
            if step % self.config.trainer.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if step % self.config.trainer.log_interval == 0:
                LOGGER.info("step=%d loss=%.4f", step, loss.item())

            if step % self.config.trainer.eval_interval == 0:
                self._evaluate()

        LOGGER.info("Training run completed.")

    def _evaluate(self) -> None:
        """Run a dummy evaluation pass."""
        self.model_bundle.model.eval()
        self.model_bundle.model.train()
