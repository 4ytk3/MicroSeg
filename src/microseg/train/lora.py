from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import torch
import torch.nn as nn

DEFAULT_LORA_TARGETS: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "qkv",
    "proj",
)


class LoRALinear(nn.Module):
    """Minimal LoRA wrapper for nn.Linear matching PEFT semantics."""

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")

        self.base_layer = base_layer
        self.rank = int(rank)
        self.lora_alpha = float(alpha)
        self.scaling = self.lora_alpha / float(self.rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(self.rank, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, self.rank))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        delta_input = self.dropout(x) if self.dropout is not None and self.training else x
        delta = (delta_input @ self.lora_A.t()) @ self.lora_B.t()
        return result + delta * self.scaling


def _matches_target(name: str, targets: Sequence[str]) -> bool:
    suffix = name.split(".")[-1]
    for target in targets:
        if suffix == target or name.endswith(target):
            return True
    return False


def apply_lora_to_module(
    module: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    target_modules: Sequence[str] = DEFAULT_LORA_TARGETS,
    prefix: str = "",
) -> List[str]:
    """Recursively wrap matching Linear layers with LoRA adapters."""

    replaced: List[str] = []
    for child_name, child in module.named_children():
        child_full = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Linear) and _matches_target(child_full, target_modules):
            wrapped = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
            setattr(module, child_name, wrapped)
            replaced.append(child_full)
        else:
            replaced.extend(
                apply_lora_to_module(
                    child,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    target_modules=target_modules,
                    prefix=child_full,
                )
            )
    return replaced


def collect_lora_parameters(module: nn.Module) -> List[nn.Parameter]:
    """Return the trainable LoRA parameters from a model."""

    params: List[nn.Parameter] = []
    for child in module.modules():
        if isinstance(child, LoRALinear):
            params.extend([child.lora_A, child.lora_B])
    return params


def load_meta_sam_with_lora(
    *,
    checkpoint: Path | str,
    model_type: str,
    device: Optional[torch.device] = None,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float = 0.0,
    target_modules: Sequence[str] = DEFAULT_LORA_TARGETS,
    lora_checkpoint: Optional[Path | str] = None,
) -> tuple[nn.Module, List[str]]:
    """Load a Meta SAM checkpoint and attach LoRA adapters."""

    from segment_anything import sam_model_registry

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    replaced = apply_lora_to_module(
        sam,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        target_modules=target_modules,
    )

    if lora_checkpoint:
        state = torch.load(str(lora_checkpoint), map_location="cpu")
        state_dict = state.get("state_dict", state)
        missing, unexpected = sam.load_state_dict(state_dict, strict=False)
        print(
            f"[INFO] Loaded LoRA weights from {lora_checkpoint} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )

    if device is not None:
        sam = sam.to(device)

    return sam, replaced


def load_hf_lora_model(
    model_name: str = "facebook/sam-vit-base",
    *,
    rank: int = 4,
    alpha: float = 16,
    dropout: float = 0.1,
    target_modules: Sequence[str] = DEFAULT_LORA_TARGETS,
):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import SamConfig, SamModel

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
    model = get_peft_model(model, lora_cfg)
    return model


def to_b1hw(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, list):
        tensor = tensor[0]

    while tensor.dim() > 4:
        tensor = tensor[:, 0, ...]

    if tensor.dim() == 4 and tensor.shape[1] != 1:
        tensor = tensor[:, :1, ...]
    elif tensor.dim() == 3:
        tensor = tensor.unsqueeze(1)
    elif tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)

    if tensor.dim() != 4:
        raise ValueError(f"Unexpected dim={tensor.dim()} after squeeze")

    return tensor.contiguous()
