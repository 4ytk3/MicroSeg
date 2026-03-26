from __future__ import annotations

import argparse
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .data import (
    create_dataloader,
    list_image_mask_files,
    make_kfold_splits,
    split_indices,
    split_train_val_indices,
)
from .lora import collect_lora_parameters, load_hf_lora_model, load_meta_sam_with_lora, to_b1hw
from .runio import (
    append_metrics,
    create_run_dir,
    write_named_metrics,
    write_run_metadata,
    write_split_metadata,
    write_cv_summary,
)

try:
    from segment_anything.utils.transforms import ResizeLongestSide
except Exception:
    ResizeLongestSide = None  # type: ignore[assignment]


def _require_resize_longest_side():
    if ResizeLongestSide is None:
        raise ImportError(
            "segment_anything is required for backend=meta. "
            "Install segment-anything or use backend=hf."
        )
    return ResizeLongestSide


@dataclass(frozen=True)
class FreezeConfig:
    image_encoder: bool
    prompt_encoder: bool
    mask_decoder: bool

    def describe(self) -> str:
        parts: List[str] = []
        if self.image_encoder:
            parts.append("image_encoder")
        if self.prompt_encoder:
            parts.append("prompt_encoder")
        if self.mask_decoder:
            parts.append("mask_decoder")
        return ", ".join(parts) if parts else "none"


@dataclass
class SamTrainingSample:
    image_embedding: torch.Tensor
    sparse_embeddings: torch.Tensor
    dense_embeddings: torch.Tensor
    target_mask: torch.Tensor
    input_size: Tuple[int, int]
    original_size: Tuple[int, int]


@dataclass
class EvalMetrics:
    loss: float
    iou: float
    dice: float
    samples: int


def resolve_freeze_config(args: argparse.Namespace) -> FreezeConfig:
    if args.freeze_config == "vit_prompt":
        return FreezeConfig(image_encoder=True, prompt_encoder=True, mask_decoder=False)
    if args.freeze_config == "prompt_mask":
        return FreezeConfig(image_encoder=False, prompt_encoder=True, mask_decoder=True)
    if args.freeze_config == "none":
        return FreezeConfig(image_encoder=False, prompt_encoder=False, mask_decoder=False)
    return FreezeConfig(
        image_encoder=args.freeze_image_encoder,
        prompt_encoder=args.freeze_prompt_encoder,
        mask_decoder=args.freeze_mask_decoder,
    )


def infer_lora_scope_tag(targets: List[str], freeze_config: FreezeConfig) -> str:
    image_targets = {"qkv", "proj"}
    mask_targets = {"q_proj", "k_proj", "v_proj", "out_proj"}
    has_image = False
    has_mask = False
    for target in targets:
        suffix = target.split(".")[-1].lower()
        if suffix in image_targets:
            has_image = True
        if suffix in mask_targets:
            has_mask = True
    if freeze_config.image_encoder:
        has_image = False
    if freeze_config.mask_decoder:
        has_mask = False
    if has_image and has_mask:
        return "all"
    if has_image:
        return "ie"
    if has_mask:
        return "md"
    return "unk"


def infer_hf_model_tag(model_id: str) -> str:
    lowered = model_id.lower()
    if "vit-base" in lowered or "vit_b" in lowered or "vitb" in lowered:
        return "hf_vit_b"
    if "vit-large" in lowered or "vit_l" in lowered or "vitl" in lowered:
        return "hf_vit_l"
    if "vit-huge" in lowered or "vit_h" in lowered or "vith" in lowered:
        return "hf_vit_h"
    short = model_id.split("/")[-1]
    short = short.replace("sam-", "")
    short = short.replace("vit-", "vit_")
    return f"hf_{short}"


def build_run_tag(args: argparse.Namespace, freeze_config: FreezeConfig) -> str:
    scope_tag = infer_lora_scope_tag(list(args.lora_targets), freeze_config)
    if args.backend == "meta":
        model_tag = f"mt_{args.sam_model_type}"
    else:
        model_tag = infer_hf_model_tag(args.hf_model_id)
    sample_tag = "img" if args.sample_mode == "image" else "inst"
    return f"{model_tag}_{scope_tag}_{sample_tag}"


def resolve_log_dir(args: argparse.Namespace, run_dir: Path, log_suffix: Optional[str] = None) -> Path:
    if args.log_dir is None:
        return run_dir / "tensorboard"
    log_dir = Path(args.log_dir)
    if log_dir.is_absolute():
        return log_dir / log_suffix if log_suffix else log_dir
    return run_dir / log_dir


def select_fold_indices(k_fold: int, fold_index: Optional[int]) -> List[int]:
    if fold_index is None:
        return list(range(k_fold))
    if fold_index < 1 or fold_index > k_fold:
        raise ValueError(f"fold_index must be in [1, {k_fold}]")
    return [fold_index - 1]


def compute_cv_summary(fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not fold_results:
        return {"folds": [], "summary": {}}
    metrics = {
        "loss": [],
        "iou": [],
        "dice": [],
        "samples": [],
    }
    serial_folds: List[Dict[str, Any]] = []
    for entry in fold_results:
        result = entry.get("test")
        fold_id = entry.get("fold")
        if result is None:
            serial_folds.append({"fold": fold_id, "test": None})
            continue
        metrics["loss"].append(result.loss)
        metrics["iou"].append(result.iou)
        metrics["dice"].append(result.dice)
        metrics["samples"].append(result.samples)
        serial_folds.append(
            {
                "fold": fold_id,
                "test": {
                    "loss": result.loss,
                    "iou": result.iou,
                    "dice": result.dice,
                    "samples": result.samples,
                },
            }
        )
    summary: Dict[str, Dict[str, float]] = {}
    for key, values in metrics.items():
        if not values:
            continue
        arr = np.array(values, dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        summary[key] = {"mean": mean, "std": std}
    return {"folds": serial_folds, "summary": summary}


def set_global_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def as_numpy_image(image: object) -> np.ndarray:
    if isinstance(image, np.ndarray):
        img = image
    elif torch.is_tensor(image):
        tensor = image.detach().cpu()
        if tensor.dim() == 4 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.dim() == 3 and tensor.shape[0] in (1, 3):
            tensor = tensor.permute(1, 2, 0)
        elif tensor.dim() == 3 and tensor.shape[-1] in (1, 3):
            tensor = tensor
        else:
            raise ValueError(f"Unexpected image tensor shape {tuple(tensor.shape)}")
        img = tensor.numpy()
    elif hasattr(image, "mode") and hasattr(image, "size"):
        img = np.array(image)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got {img.shape}")

    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.floating):
            if img.max() <= 1.0:
                img = img * 255.0
            img = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    return np.ascontiguousarray(img)


def normalize_points(points: torch.Tensor) -> np.ndarray:
    if not torch.is_tensor(points):
        points = torch.as_tensor(points)
    pts = points.detach().cpu()
    if pts.dim() == 4 and pts.shape[0] == 1:
        pts = pts[0]
    if pts.dim() == 3:
        pts = pts[0]
    elif pts.dim() == 2:
        pass
    elif pts.dim() == 1:
        pts = pts.view(1, 2)
    else:
        raise ValueError(f"Unexpected point shape {tuple(pts.shape)}")
    if pts.shape[-1] != 2:
        raise ValueError(f"Expected point shape (*,2), got {tuple(pts.shape)}")
    return pts.numpy()


def prepare_sam_sample(
    sam_model,
    image: object,
    mask_tensor: torch.Tensor,
    points: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    transform: ResizeLongestSide,
    allow_image_grad: bool,
    allow_prompt_grad: bool,
) -> Optional[SamTrainingSample]:
    image_np = as_numpy_image(image)
    mask_np = mask_tensor.detach().squeeze().cpu().numpy()
    if mask_np.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {mask_np.shape}")
    if mask_np.sum() <= 0:
        return None

    image_resized = transform.apply_image(image_np)
    image_tensor = torch.as_tensor(image_resized, device=device)
    image_tensor = image_tensor.permute(2, 0, 1).contiguous().float() / 255.0
    input_tensor = sam_model.preprocess(image_tensor.unsqueeze(0))

    with torch.set_grad_enabled(allow_image_grad):
        image_embedding = sam_model.image_encoder(input_tensor)

    original_size = image_np.shape[:2]
    input_size = tuple(input_tensor.shape[-2:])

    point_xy = normalize_points(points)
    point_tf = transform.apply_coords(point_xy, original_size)
    point_coords = torch.as_tensor(point_tf, dtype=torch.float32, device=device).unsqueeze(0)

    point_labels = labels.to(device=device, dtype=torch.float32).view(1, -1)
    if point_labels.shape[1] != point_coords.shape[1]:
        if point_labels.shape[1] == 1:
            point_labels = point_labels.repeat(1, point_coords.shape[1])
        else:
            raise ValueError(
                f"point_labels shape {tuple(point_labels.shape)} does not match points {tuple(point_coords.shape)}"
            )

    ys, xs = np.where(mask_np > 0.5)
    if ys.size == 0 or xs.size == 0:
        bbox = np.array([[0, 0, original_size[1], original_size[0]]], dtype=np.float32)
    else:
        bbox = np.array([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]], dtype=np.float32)
    box_tf = transform.apply_boxes(bbox, original_size)
    box_t = torch.as_tensor(box_tf, dtype=torch.float32, device=device)

    with torch.set_grad_enabled(allow_prompt_grad):
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=(point_coords, point_labels),
            boxes=box_t,
            masks=None,
        )

    target_mask = torch.as_tensor(mask_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    return SamTrainingSample(
        image_embedding=image_embedding,
        sparse_embeddings=sparse_embeddings,
        dense_embeddings=dense_embeddings,
        target_mask=target_mask,
        input_size=input_size,
        original_size=original_size,
    )


def predict_meta_logits(sam_model, sample: SamTrainingSample) -> torch.Tensor:
    low_res_masks, _ = sam_model.mask_decoder(
        image_embeddings=sample.image_embedding,
        image_pe=sam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sample.sparse_embeddings,
        dense_prompt_embeddings=sample.dense_embeddings,
        multimask_output=False,
    )
    upscaled = sam_model.postprocess_masks(
        low_res_masks,
        sample.input_size,
        sample.original_size,
    )
    upscaled = F.interpolate(
        upscaled,
        size=sample.target_mask.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    return upscaled


def compute_mask_loss(logits: torch.Tensor, target: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    return criterion(logits, target)


def compute_iou_dice(logits: torch.Tensor, target: torch.Tensor, threshold: float) -> Tuple[float, float]:
    probs = torch.sigmoid(logits)
    pred = (probs > threshold).float()
    target_bin = (target > 0.5).float()
    dims = tuple(range(1, pred.dim()))
    intersection = (pred * target_bin).sum(dim=dims)
    union = pred.sum(dim=dims) + target_bin.sum(dim=dims) - intersection
    iou = (intersection / (union + 1e-6)).mean().item()
    dice = (2 * intersection / (pred.sum(dim=dims) + target_bin.sum(dim=dims) + 1e-6)).mean().item()
    return iou, dice


def _freeze_module(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def apply_meta_freeze(model, freeze_config: FreezeConfig) -> None:
    if freeze_config.image_encoder:
        _freeze_module(model.image_encoder)
    if freeze_config.prompt_encoder:
        _freeze_module(model.prompt_encoder)
    if freeze_config.mask_decoder:
        _freeze_module(model.mask_decoder)


def apply_hf_freeze(model, freeze_config: FreezeConfig) -> None:
    patterns = []
    if freeze_config.image_encoder:
        patterns.extend(["vision_encoder.", "image_encoder."])
    if freeze_config.prompt_encoder:
        patterns.append("prompt_encoder.")
    if freeze_config.mask_decoder:
        patterns.append("mask_decoder.")
    if not patterns:
        return
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in patterns):
            param.requires_grad = False


def save_checkpoint(model, run_dir: Path, tag: object) -> None:
    if isinstance(tag, int):
        name = f"sam_lora_epoch{tag:03d}.pt"
    else:
        name = f"sam_lora_{tag}.pt"
    path = run_dir / name
    torch.save(model.state_dict(), path)
    print(f"[INFO] Saved {path}")


def evaluate_meta_backend(
    sam,
    loader: DataLoader,
    device: torch.device,
    mask_threshold: float,
) -> Optional[EvalMetrics]:
    was_training = sam.training
    sam.eval()
    criterion = nn.BCEWithLogitsLoss()
    transform = _require_resize_longest_side()(sam.image_encoder.img_size)

    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total = 0

    with torch.no_grad():
        for batch in loader:
            sample = batch[0]
            data = prepare_sam_sample(
                sam,
                image=sample["image"],
                mask_tensor=sample["mask"],
                points=sample["input_point"],
                labels=sample["input_label"],
                device=device,
                transform=transform,
                allow_image_grad=False,
                allow_prompt_grad=False,
            )
            if data is None:
                continue

            logits = predict_meta_logits(sam, data)
            loss = compute_mask_loss(logits, data.target_mask, criterion)
            iou, dice = compute_iou_dice(logits, data.target_mask, mask_threshold)

            batch_size = int(data.target_mask.shape[0])
            total_loss += loss.item() * batch_size
            total_iou += iou * batch_size
            total_dice += dice * batch_size
            total += batch_size

    if was_training:
        sam.train()
    if total == 0:
        return None
    return EvalMetrics(
        loss=total_loss / total,
        iou=total_iou / total,
        dice=total_dice / total,
        samples=total,
    )


def prepare_hf_batch(batch, device: torch.device, input_size: int = 1024):
    images = []
    masks = []
    points = []
    labels = []
    for sample in batch:
        img_np = as_numpy_image(sample["image"])
        img = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        orig_h, orig_w = int(img.shape[1]), int(img.shape[2])

        mask = sample["mask"]
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        mask = mask.float()

        pt = sample["input_point"]
        if not torch.is_tensor(pt):
            pt = torch.as_tensor(pt)
        pt = pt.float()
        if pt.dim() == 2:
            pt = pt.unsqueeze(0)
        if pt.dim() != 3 or pt.shape[-1] != 2:
            raise ValueError(f"Unexpected input_point shape {tuple(pt.shape)}")

        lbl = sample["input_label"]
        if not torch.is_tensor(lbl):
            lbl = torch.as_tensor(lbl)
        lbl = lbl.long()
        if lbl.dim() == 1:
            lbl = lbl.unsqueeze(0)

        if input_size > 0:
            target_h = int(input_size)
            target_w = int(input_size)
            if orig_h != target_h or orig_w != target_w:
                img = F.interpolate(
                    img.unsqueeze(0),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                mask = F.interpolate(
                    mask.unsqueeze(0),
                    size=(target_h, target_w),
                    mode="nearest",
                ).squeeze(0)
                scale_x = float(target_w) / float(orig_w)
                scale_y = float(target_h) / float(orig_h)
                pt = pt.clone()
                pt[..., 0] = torch.clamp(pt[..., 0] * scale_x, min=0.0, max=float(target_w - 1))
                pt[..., 1] = torch.clamp(pt[..., 1] * scale_y, min=0.0, max=float(target_h - 1))

        images.append(img)
        masks.append(mask)
        points.append(pt)
        labels.append(lbl)

    pixel_values = torch.stack(images).to(device)
    target_masks = torch.stack(masks).to(device)
    input_points = torch.stack(points).to(device)
    input_labels = torch.stack(labels).to(device)
    return pixel_values, target_masks, input_points, input_labels


def evaluate_hf_backend(
    model,
    loader: DataLoader,
    device: torch.device,
    mask_threshold: float,
    input_size: int = 1024,
) -> Optional[EvalMetrics]:
    was_training = model.training
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total = 0

    with torch.no_grad():
        for batch in loader:
            pixel_values, target_masks, input_points, input_labels = prepare_hf_batch(
                batch,
                device,
                input_size=input_size,
            )
            outputs = model(
                pixel_values=pixel_values,
                input_points=input_points,
                input_labels=input_labels,
                multimask_output=False,
            )
            pred_low = outputs.masks if hasattr(outputs, "masks") else outputs.pred_masks
            pred_low = to_b1hw(pred_low)
            pred = F.interpolate(
                pred_low,
                size=target_masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            loss = compute_mask_loss(pred, target_masks, criterion)
            iou, dice = compute_iou_dice(pred, target_masks, mask_threshold)

            batch_size = int(target_masks.shape[0])
            total_loss += loss.item() * batch_size
            total_iou += iou * batch_size
            total_dice += dice * batch_size
            total += batch_size

    if was_training:
        model.train()
    if total == 0:
        return None
    return EvalMetrics(
        loss=total_loss / total,
        iou=total_iou / total,
        dice=total_dice / total,
        samples=total,
    )


def log_metrics(writer: SummaryWriter, prefix: str, epoch: int, metrics: EvalMetrics) -> None:
    writer.add_scalar(f"{prefix}/loss", metrics.loss, epoch)
    writer.add_scalar(f"{prefix}/iou", metrics.iou, epoch)
    writer.add_scalar(f"{prefix}/dice", metrics.dice, epoch)


def compute_grad_norm(params: List[torch.nn.Parameter]) -> float:
    total_norm_sq = 0.0
    for param in params:
        if param.grad is None:
            continue
        param_norm = param.grad.detach().norm(2)
        total_norm_sq += float(param_norm) ** 2
    return total_norm_sq**0.5


def log_train_step(
    writer: SummaryWriter,
    global_step: int,
    loss_value: float,
    optimizer: torch.optim.Optimizer,
    grad_norm: float,
    device: torch.device,
) -> None:
    writer.add_scalar("train/loss_iter", loss_value, global_step)
    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
    writer.add_scalar("train/grad_norm", grad_norm, global_step)
    if device.type == "cuda":
        writer.add_scalar(
            "train/gpu_mem_allocated_mb",
            torch.cuda.memory_allocated(device) / (1024**2),
            global_step,
        )
        writer.add_scalar(
            "train/gpu_mem_reserved_mb",
            torch.cuda.memory_reserved(device) / (1024**2),
            global_step,
        )


def train_meta_backend(
    args: argparse.Namespace,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    freeze_config: FreezeConfig,
    run_dir: Path,
    log_dir: Path,
    mask_threshold: float,
) -> torch.nn.Module:
    if args.sam_checkpoint is None:
        raise ValueError("--sam-checkpoint is required when backend=meta")

    sam, replaced = load_meta_sam_with_lora(
        checkpoint=args.sam_checkpoint,
        model_type=args.sam_model_type,
        device=device,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_targets,
    )
    if not replaced:
        print("[WARN] No LoRA layers were replaced. Check --lora-targets for Meta SAM.")
    else:
        print(f"[INFO] Applied LoRA to {len(replaced)} linear layers on Meta SAM backend")

    sam.train()
    apply_meta_freeze(sam, freeze_config)

    lora_params = [p for p in collect_lora_parameters(sam) if p.requires_grad]
    if not lora_params:
        raise RuntimeError(
            "No trainable LoRA parameters after applying freeze config. "
            "Check --lora-targets (e.g., include qkv/proj for vision encoder) "
            "or adjust --freeze-config."
        )

    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    transform = _require_resize_longest_side()(sam.image_encoder.img_size)
    allow_image_grad = not freeze_config.image_encoder
    allow_prompt_grad = not freeze_config.prompt_encoder

    best_val_loss: Optional[float] = None

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_start = time.time()
        running_loss = 0.0
        valid_samples = 0
        for batch in train_loader:
            sample = batch[0]
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                data = prepare_sam_sample(
                    sam,
                    image=sample["image"],
                    mask_tensor=sample["mask"],
                    points=sample["input_point"],
                    labels=sample["input_label"],
                    device=device,
                    transform=transform,
                    allow_image_grad=allow_image_grad,
                    allow_prompt_grad=allow_prompt_grad,
                )
                if data is None:
                    continue
                logits = predict_meta_logits(sam, data)
                loss = compute_mask_loss(logits, data.target_mask, criterion)
            loss.backward()
            if args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
            else:
                grad_norm = compute_grad_norm(lora_params)
            optimizer.step()

            running_loss += loss.item()
            valid_samples += 1
            log_train_step(
                writer,
                global_step,
                loss.item(),
                optimizer,
                float(grad_norm),
                device,
            )
            global_step += 1

        if valid_samples == 0:
            print("[WARN] No valid samples processed this epoch")
            continue

        avg_loss = running_loss / valid_samples
        writer.add_scalar("train/loss_epoch", avg_loss, epoch)
        epoch_time = time.time() - epoch_start
        writer.add_scalar("train/epoch_time_sec", epoch_time, epoch)
        if device.type == "cuda":
            writer.add_scalar(
                "train/gpu_mem_peak_mb",
                torch.cuda.max_memory_allocated(device) / (1024**2),
                epoch,
            )
        print(f"[INFO] Epoch {epoch}/{args.epochs} - loss {avg_loss:.4f}")

        if args.save_every and epoch % args.save_every == 0:
            save_checkpoint(sam, run_dir, epoch)

        if val_loader and args.eval_every > 0 and epoch % args.eval_every == 0:
            val_start = time.time()
            metrics = evaluate_meta_backend(sam, val_loader, device, mask_threshold)
            val_time = time.time() - val_start
            if metrics:
                log_metrics(writer, "val", epoch, metrics)
                append_metrics(run_dir, "val", epoch, metrics, extra={"epoch_time_sec": epoch_time, "val_time_sec": val_time})
                if args.save_best and (best_val_loss is None or metrics.loss < best_val_loss):
                    best_val_loss = metrics.loss
                    save_checkpoint(sam, run_dir, "best")
                    write_named_metrics(run_dir, "best_metrics", epoch, metrics)
            else:
                append_metrics(run_dir, "val", epoch, EvalMetrics(0.0, 0.0, 0.0, 0), extra={"epoch_time_sec": epoch_time, "val_time_sec": val_time})
        else:
            append_metrics(run_dir, "train", epoch, EvalMetrics(avg_loss, 0.0, 0.0, valid_samples), extra={"epoch_time_sec": epoch_time})

    save_checkpoint(sam, run_dir, args.epochs)
    writer.close()
    return sam


def train_hf_backend(
    args: argparse.Namespace,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    freeze_config: FreezeConfig,
    run_dir: Path,
    log_dir: Path,
    mask_threshold: float,
) -> torch.nn.Module:
    model = load_hf_lora_model(
        model_name=args.hf_model_id,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=args.lora_targets,
    )
    model.to(device)
    model.train()

    apply_hf_freeze(model, freeze_config)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError(
            "No trainable LoRA parameters after applying freeze config. "
            "Check --lora-targets (e.g., include qkv/proj for vision encoder) "
            "or adjust --freeze-config."
        )

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    best_val_loss: Optional[float] = None

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_start = time.time()
        running_loss = 0.0
        valid_samples = 0
        for batch in train_loader:
            pixel_values, target_masks, input_points, input_labels = prepare_hf_batch(
                batch,
                device,
                input_size=int(args.hf_input_size),
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(
                    pixel_values=pixel_values,
                    input_points=input_points,
                    input_labels=input_labels,
                    multimask_output=False,
                )
                pred_low = outputs.masks if hasattr(outputs, "masks") else outputs.pred_masks
                pred_low = to_b1hw(pred_low)
                pred = F.interpolate(
                    pred_low,
                    size=target_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                loss = compute_mask_loss(pred, target_masks, criterion)
            loss.backward()
            if args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            else:
                grad_norm = compute_grad_norm(trainable_params)
            optimizer.step()

            running_loss += loss.item()
            valid_samples += 1
            log_train_step(
                writer,
                global_step,
                loss.item(),
                optimizer,
                float(grad_norm),
                device,
            )
            global_step += 1

        if valid_samples == 0:
            print("[WARN] No valid samples processed this epoch")
            continue

        avg_loss = running_loss / valid_samples
        writer.add_scalar("train/loss_epoch", avg_loss, epoch)
        epoch_time = time.time() - epoch_start
        writer.add_scalar("train/epoch_time_sec", epoch_time, epoch)
        if device.type == "cuda":
            writer.add_scalar(
                "train/gpu_mem_peak_mb",
                torch.cuda.max_memory_allocated(device) / (1024**2),
                epoch,
            )
        print(f"[INFO] Epoch {epoch}/{args.epochs} - loss {avg_loss:.4f}")

        if args.save_every and epoch % args.save_every == 0:
            save_checkpoint(model, run_dir, epoch)

        if val_loader and args.eval_every > 0 and epoch % args.eval_every == 0:
            val_start = time.time()
            metrics = evaluate_hf_backend(
                model,
                val_loader,
                device,
                mask_threshold,
                input_size=int(args.hf_input_size),
            )
            val_time = time.time() - val_start
            if metrics:
                log_metrics(writer, "val", epoch, metrics)
                append_metrics(run_dir, "val", epoch, metrics, extra={"epoch_time_sec": epoch_time, "val_time_sec": val_time})
                if args.save_best and (best_val_loss is None or metrics.loss < best_val_loss):
                    best_val_loss = metrics.loss
                    save_checkpoint(model, run_dir, "best")
                    write_named_metrics(run_dir, "best_metrics", epoch, metrics)
            else:
                append_metrics(run_dir, "val", epoch, EvalMetrics(0.0, 0.0, 0.0, 0), extra={"epoch_time_sec": epoch_time, "val_time_sec": val_time})
        else:
            append_metrics(run_dir, "train", epoch, EvalMetrics(avg_loss, 0.0, 0.0, valid_samples), extra={"epoch_time_sec": epoch_time})

    save_checkpoint(model, run_dir, args.epochs)
    writer.close()
    return model


def run_single_split(
    args: argparse.Namespace,
    image_dir: Path,
    mask_dir: Path,
    image_files: List[str],
    mask_files: List[str],
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
    run_dir: Path,
    log_dir: Path,
    freeze_config: FreezeConfig,
    device: torch.device,
    run_tag: Optional[str],
) -> Optional[EvalMetrics]:
    if not train_idx:
        raise ValueError("Training split is empty")

    write_run_metadata(
        run_dir,
        args=args,
        freeze_config=asdict(freeze_config),
        freeze_label=freeze_config.describe(),
        device=str(device),
        log_dir=log_dir,
        run_tag=run_tag,
    )
    write_split_metadata(run_dir, image_files, mask_files, train_idx, val_idx, test_idx)

    print(f"[INFO] Run dir: {run_dir}")
    print(f"[INFO] Freeze config: {freeze_config.describe()}")

    train_loader = create_dataloader(
        args,
        image_dir,
        mask_dir,
        image_files,
        mask_files,
        train_idx,
        shuffle=True,
    )
    val_loader = None
    if val_idx:
        val_loader = create_dataloader(
            args,
            image_dir,
            mask_dir,
            image_files,
            mask_files,
            val_idx,
            shuffle=False,
        )
    test_loader = None
    if test_idx:
        test_loader = create_dataloader(
            args,
            image_dir,
            mask_dir,
            image_files,
            mask_files,
            test_idx,
            shuffle=False,
        )

    run_start = time.time()
    if args.backend == "meta":
        model = train_meta_backend(
            args,
            train_loader,
            val_loader,
            device,
            freeze_config,
            run_dir,
            log_dir,
            args.mask_threshold,
        )
        test_metrics = None
        if test_loader:
            metrics = evaluate_meta_backend(model, test_loader, device, args.mask_threshold)
            if metrics:
                append_metrics(run_dir, "test", args.epochs, metrics)
                write_named_metrics(run_dir, "test_metrics", args.epochs, metrics)
                test_metrics = metrics
        total_time = time.time() - run_start
        append_metrics(run_dir, "run", args.epochs, EvalMetrics(0.0, 0.0, 0.0, 0), extra={"total_time_sec": total_time})
        return test_metrics

    model = train_hf_backend(
        args,
        train_loader,
        val_loader,
        device,
        freeze_config,
        run_dir,
        log_dir,
        args.mask_threshold,
    )
    test_metrics = None
    if test_loader:
        metrics = evaluate_hf_backend(
            model,
            test_loader,
            device,
            args.mask_threshold,
            input_size=int(args.hf_input_size),
        )
        if metrics:
            append_metrics(run_dir, "test", args.epochs, metrics)
            write_named_metrics(run_dir, "test_metrics", args.epochs, metrics)
            test_metrics = metrics
    total_time = time.time() - run_start
    append_metrics(run_dir, "run", args.epochs, EvalMetrics(0.0, 0.0, 0.0, 0), extra={"total_time_sec": total_time})
    return test_metrics


def train_model(args: argparse.Namespace) -> None:
    if args.mask_threshold < 0 or args.mask_threshold > 1:
        raise ValueError("mask_threshold must be in [0, 1]")
    if args.hf_mask_threshold is not None:
        args.mask_threshold = args.hf_mask_threshold
    if int(args.hf_input_size) < 0:
        raise ValueError("hf_input_size must be >= 0")

    if args.k_fold < 1:
        raise ValueError("k_fold must be >= 1")
    if args.k_fold == 1 and args.fold_index is not None:
        raise ValueError("fold_index requires k_fold >= 2")

    set_global_seed(args.seed)
    split_seed = args.split_seed if args.split_seed is not None else args.seed
    if split_seed is None:
        split_seed = 42

    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    image_files, mask_files = list_image_mask_files(image_dir, mask_dir)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    freeze_config = resolve_freeze_config(args)
    base_tag = build_run_tag(args, freeze_config)

    if args.k_fold > 1:
        if args.train_split != 0.0 or args.test_split != 0.0:
            print("[INFO] k-fold enabled: --train-split/--test-split are ignored.")
        folds = make_kfold_splits(len(image_files), args.k_fold, int(split_seed))
        fold_indices = select_fold_indices(args.k_fold, args.fold_index)
        cv_tag = f"{base_tag}_kf{args.k_fold}"
        parent_run_dir = create_run_dir(Path(args.output_dir), cv_tag)
        print(f"[INFO] CV run dir: {parent_run_dir}")

        fold_results: List[Dict[str, Any]] = []
        for fold_idx in fold_indices:
            fold_name = f"fold{fold_idx + 1:02d}"
            run_dir = parent_run_dir / fold_name
            run_dir.mkdir(parents=True, exist_ok=False)
            log_dir = resolve_log_dir(args, run_dir, fold_name)

            remaining = [idx for i, fold in enumerate(folds) if i != fold_idx for idx in fold]
            train_idx, val_idx = split_train_val_indices(remaining, args.val_split, int(split_seed) + fold_idx)
            test_idx = folds[fold_idx]

            fold_args = argparse.Namespace(**vars(args))
            fold_args.fold_index = fold_idx + 1
            if args.seed is not None:
                set_global_seed(args.seed)

            metrics = run_single_split(
                fold_args,
                image_dir,
                mask_dir,
                image_files,
                mask_files,
                train_idx,
                val_idx,
                test_idx,
                run_dir,
                log_dir,
                freeze_config,
                device,
                run_tag=f"{base_tag}_{fold_name}",
            )
            fold_results.append({"fold": fold_idx + 1, "test": metrics})
            if device.type == "cuda":
                torch.cuda.empty_cache()

        summary = compute_cv_summary(fold_results)
        summary.update(
            {
                "k_fold": args.k_fold,
                "fold_indices": [idx + 1 for idx in fold_indices],
                "run_tag": base_tag,
            }
        )
        write_cv_summary(parent_run_dir, summary)
        print(f"[INFO] CV summary: {parent_run_dir / 'cv_summary.json'}")
        return

    train_idx, val_idx, test_idx = split_indices(
        len(image_files),
        args.train_split,
        args.val_split,
        args.test_split,
        int(split_seed),
    )
    run_dir = create_run_dir(Path(args.output_dir), base_tag)
    log_dir = resolve_log_dir(args, run_dir)
    run_single_split(
        args,
        image_dir,
        mask_dir,
        image_files,
        mask_files,
        train_idx,
        val_idx,
        test_idx,
        run_dir,
        log_dir,
        freeze_config,
        device,
        run_tag=base_tag,
    )
