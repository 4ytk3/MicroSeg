from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from microseg.sam_lora_utils import load_lora_sam_model, to_b1hw


@dataclass
class ImageInfo:
    original_size: Tuple[int, int]  # (H, W)
    scale_x: float
    scale_y: float
    pil_image: Image.Image


class HFSamPredictor:
    def __init__(self, model, device: torch.device, threshold: float = 0.5) -> None:
        self.model = model
        self.device = device
        self.threshold = float(threshold)
        self.pixel_values: torch.Tensor | None = None
        self.image_info: ImageInfo | None = None
        self.model.to(self.device)
        self.model.eval()

    def set_image(self, image_bgr: np.ndarray) -> None:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        orig_w, orig_h = pil_img.size
        resized = pil_img.resize((1024, 1024), Image.BILINEAR)
        tensor = transforms.ToTensor()(resized).unsqueeze(0).to(self.device)
        self.pixel_values = tensor
        self.image_info = ImageInfo(
            original_size=(orig_h, orig_w),
            scale_x=1024.0 / float(orig_w),
            scale_y=1024.0 / float(orig_h),
            pil_image=pil_img,
        )

    def _prepare_points(
        self,
        points: Sequence[Tuple[float, float]],
        labels: Sequence[int],
    ) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.pixel_values is None or self.image_info is None:
            raise RuntimeError("set_image must be called before predict.")
        if len(points) == 0:
            return None, None
        scaled = [[[(p[0] * self.image_info.scale_x), (p[1] * self.image_info.scale_y)] for p in points]]
        pts = torch.tensor([scaled], dtype=torch.float32, device=self.device)
        lbls = torch.tensor([[[label for label in labels]]], dtype=torch.int64, device=self.device)
        return pts, lbls

    def _prepare_box(self, box: Tuple[float, float, float, float]) -> torch.Tensor:
        if self.pixel_values is None or self.image_info is None:
            raise RuntimeError("set_image must be called before predict.")
        x0, y0, x1, y1 = box
        scaled = [[
            float(x0) * self.image_info.scale_x,
            float(y0) * self.image_info.scale_y,
            float(x1) * self.image_info.scale_x,
            float(y1) * self.image_info.scale_y,
        ]]
        return torch.tensor([scaled], dtype=torch.float32, device=self.device)

    def predict(
        self,
        points: Sequence[Tuple[float, float]],
        labels: Sequence[int] | None = None,
        box: Tuple[float, float, float, float] | None = None,
        multimask_output: bool = False,
        use_adapter: bool = True,
    ) -> Tuple[List[np.ndarray], List[float]]:
        if labels is None and len(points) > 0:
            labels = [1] * len(points)
        if labels is None:
            labels = []
        if len(labels) != len(points):
            raise ValueError("labels length must match points length")

        input_points, input_labels = self._prepare_points(points, labels)
        input_boxes = self._prepare_box(box) if box is not None else None
        if input_points is None and input_boxes is None:
            raise ValueError("At least one prompt is required (point or box).")
        if self.pixel_values is None or self.image_info is None:
            raise RuntimeError("Pixel values not prepared.")

        autocast_device = "cuda" if self.device.type == "cuda" else "cpu"
        autocast_enabled = self.device.type == "cuda"
        disable_ctx = nullcontext()
        if not use_adapter and hasattr(self.model, "disable_adapter"):
            try:
                disable_ctx = self.model.disable_adapter()
            except Exception:
                disable_ctx = nullcontext()

        with disable_ctx:
            with torch.no_grad(), torch.autocast(device_type=autocast_device, enabled=autocast_enabled):
                kwargs = {
                    "pixel_values": self.pixel_values,
                    "multimask_output": multimask_output,
                }
                if input_points is not None:
                    kwargs["input_points"] = input_points
                if input_labels is not None:
                    kwargs["input_labels"] = input_labels
                if input_boxes is not None:
                    kwargs["input_boxes"] = input_boxes
                outputs = self.model(**kwargs)

        masks = outputs.masks if hasattr(outputs, "masks") else outputs.pred_masks
        masks = to_b1hw(masks)
        masks_upsampled = F.interpolate(masks, size=(1024, 1024), mode="bilinear", align_corners=False)
        orig_h, orig_w = self.image_info.original_size
        masks_orig = F.interpolate(masks_upsampled, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        probs = masks_orig.sigmoid()
        mask_arrays = (probs > self.threshold).to(torch.uint8).cpu().numpy()

        iou_scores = outputs.iou_scores if hasattr(outputs, "iou_scores") else None
        if iou_scores is None:
            score_values = torch.zeros((masks_orig.shape[0], masks_orig.shape[1]), device=self.device)
        else:
            score_values = torch.sigmoid(iou_scores)
        score_values = score_values.squeeze(0).flatten().cpu().numpy()

        result_masks: List[np.ndarray] = []
        result_scores: List[float] = []
        for idx in range(mask_arrays.shape[1]):
            mask = mask_arrays[0, idx]
            result_masks.append(mask)
            score_val = score_values[idx] if idx < len(score_values) else score_values[-1]
            result_scores.append(float(score_val))
        return result_masks, result_scores


def load_hf_sam_model(
    model_id: str,
    device: torch.device,
    lora_checkpoint: Optional[Path] = None,
    *,
    rank: int = 4,
    alpha: float = 16.0,
    dropout: float = 0.1,
    target_modules: Sequence[str] = ("q_proj", "k_proj", "v_proj", "out_proj", "qkv", "proj"),
):
    sam = load_lora_sam_model(
        model_name=model_id,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules,
    )
    if lora_checkpoint:
        state = torch.load(str(lora_checkpoint), map_location="cpu")
        state_dict = state.get("state_dict", state)
        missing, unexpected = sam.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Loaded LoRA weights from {lora_checkpoint} (missing={len(missing)}, unexpected={len(unexpected)})")
    sam.to(device)
    sam.eval()
    return sam
