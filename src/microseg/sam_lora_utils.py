from __future__ import annotations

from typing import Sequence

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import SamConfig, SamModel


def load_lora_sam_model(
    model_name: str = "facebook/sam-vit-base",
    *,
    rank: int = 4,
    alpha: float = 16.0,
    dropout: float = 0.1,
    target_modules: Sequence[str] = ("q_proj", "k_proj", "v_proj", "out_proj", "qkv", "proj"),
):
    config = SamConfig.from_pretrained(model_name)
    model = SamModel.from_pretrained(model_name, config=config)
    model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=list(target_modules),
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    return get_peft_model(model, lora_cfg)


def to_b1hw(t: torch.Tensor) -> torch.Tensor:
    if isinstance(t, list):
        t = t[0]

    while t.dim() > 4:
        t = t[:, 0, ...]

    if t.dim() == 4 and t.shape[1] != 1:
        t = t[:, :1, ...]
    elif t.dim() == 3:
        t = t.unsqueeze(1)
    elif t.dim() == 2:
        t = t.unsqueeze(0).unsqueeze(0)

    if t.dim() != 4:
        raise ValueError(f"Unexpected dim={t.dim()} after squeeze")
    return t.contiguous()
