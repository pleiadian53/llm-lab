"""Model helpers for pre-training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from llm_lab.pretrain_llm.config import PretrainingConfig


@dataclass
class AutoregressiveLM:
    """Bundle a language model and tokenizer with a tiny inference helper."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase

    @classmethod
    def from_config(cls, config: PretrainingConfig) -> "AutoregressiveLM":
        tokenizer = AutoTokenizer.from_pretrained(config.model.pretrained_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(config.model.pretrained_model_name)
        if config.model.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        return cls(model=model, tokenizer=tokenizer)

    def to(self, device: torch.device) -> "AutoregressiveLM":
        self.model.to(device)
        return self

    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = self.model.generate(**encoded, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs = self.model(**batch, labels=batch["input_ids"])
        return {"loss": outputs.loss, "logits": outputs.logits}
