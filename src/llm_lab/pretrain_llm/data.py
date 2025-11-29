"""Dataset helpers for pre-training experiments."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List

from datasets import IterableDataset, load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from llm_lab.pretrain_llm.config import PretrainingConfig

LOGGER = logging.getLogger(__name__)


class TokenizedTextDataset(Dataset):
    """A tiny in-memory dataset built from tokenized text examples."""

    def __init__(self, token_ids: List[List[int]]) -> None:
        self._token_ids = token_ids

    def __len__(self) -> int:
        return len(self._token_ids)

    def __getitem__(self, idx: int) -> List[int]:
        return self._token_ids[idx]


@dataclass
class PretrainingDataModule:
    """Thin wrapper that produces PyTorch dataloaders from a Hugging Face dataset."""

    config: PretrainingConfig
    tokenizer: PreTrainedTokenizerBase
    batch_size: int = 8
    max_seq_length: int = 256

    def _load_dataset(self) -> Iterable[dict]:
        try:
            dataset = load_dataset(
                path=self.config.data.dataset_name,
                name=self.config.data.dataset_config,
                split="train",
                streaming=self.config.data.streaming,
            )
            if isinstance(dataset, IterableDataset):
                return dataset
            return iter(dataset)
        except Exception as exc:  # pragma: no cover - network dependent
            LOGGER.warning(
                "Falling back to synthetic dataset because load_dataset failed: %s", exc
            )
            return [
                {self.config.data.text_column: "Language models are few-shot learners."},
                {self.config.data.text_column: "LLM Lab fallback sample."},
            ]

    def _tokenize_batch(self, batch: Iterable[dict]) -> List[List[int]]:
        examples: List[List[int]] = []
        for sample in batch:
            text = sample[self.config.data.text_column]
            encoded = self.tokenizer.encode(
                text,
                truncation=True,
                max_length=self.max_seq_length,
            )
            examples.append(encoded)
        return examples

    def _collate(self, batch: List[List[int]]) -> dict:
        return self.tokenizer.pad(
            {"input_ids": batch},
            padding=True,
            return_tensors="pt",
        )

    def build_dataloader(self) -> DataLoader:
        """Create a torch DataLoader object."""
        dataset_iter = self._load_dataset()
        tokenized_examples = self._tokenize_batch(dataset_iter)
        dataset = TokenizedTextDataset(tokenized_examples)
        return DataLoader(dataset, batch_size=self.batch_size, collate_fn=self._collate)
