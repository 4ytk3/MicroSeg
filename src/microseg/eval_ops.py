from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _slugify(value: str) -> str:
    keep = []
    for ch in value.strip().lower().replace(" ", "_"):
        if ch.isalnum() or ch in ("_", "-"):
            keep.append(ch)
    return "".join(keep) or "run"


def _require_roifile():
    try:
        from roifile import ImagejRoi
    except ImportError as exc:
        raise ImportError("roifile is required for ROI GT. Install with `pip install roifile`.") from exc
    return ImagejRoi


def load_roi_instances(
    roi_path: Path,
    shape: Tuple[int, int],
    min_area: int,
    tmp_root: Path,
) -> List[np.ndarray]:
    ImagejRoi = _require_roifile()
    from skimage.draw import polygon2mask

    tmp_dir: Optional[Path] = None
    if roi_path.is_dir():
        roi_files = sorted(roi_path.rglob("*.roi"))
    elif roi_path.suffix.lower() == ".zip":
        tmp_dir = tmp_root / f"roi_tmp_{_slugify(roi_path.stem)}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(roi_path, "r") as zf:
            zf.extractall(tmp_dir)
        roi_files = sorted(tmp_dir.rglob("*.roi"))
    else:
        roi_files = [roi_path]

    instances: List[np.ndarray] = []
    if not roi_files:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return instances

    for path in roi_files:
        try:
            roi = ImagejRoi.fromfile(str(path))
        except Exception:
            continue

        mask = None
        if hasattr(roi, "to_mask"):
            try:
                mask = roi.to_mask(shape)
            except Exception:
                mask = None

        if mask is None:
            try:
                coords = np.array(roi.coordinates())
            except Exception:
                continue
            if coords.size == 0:
                continue
            mask = polygon2mask(shape, np.flip(coords, axis=1))

        mask = mask.astype(bool)
        if int(mask.sum()) < int(min_area):
            continue
        instances.append(mask)

    if tmp_dir is not None:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return instances


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def compute_dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    total = mask_a.sum() + mask_b.sum()
    if total == 0:
        return 0.0
    return float((2 * inter) / total)


def _to_tensor(mask: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(mask.astype(np.float32))[None, None]


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    tensor = _to_tensor(mask)
    pooled = F.max_pool2d(tensor, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return (pooled > 0.5).squeeze(0).squeeze(0).cpu().numpy().astype(bool)


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    tensor = _to_tensor(mask)
    pooled = -F.max_pool2d(-tensor, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return (pooled > 0.5).squeeze(0).squeeze(0).cpu().numpy().astype(bool)


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return mask.astype(bool)
    eroded = _erode(mask.astype(bool), radius=1)
    return np.logical_and(mask.astype(bool), np.logical_not(eroded))


def boundary_fscore(
    pred: np.ndarray,
    target: np.ndarray,
    tolerance: int,
) -> Tuple[float, float, float]:
    boundary_pred = mask_boundary(pred)
    boundary_gt = mask_boundary(target)
    pred_count = int(boundary_pred.sum())
    gt_count = int(boundary_gt.sum())
    if pred_count == 0 and gt_count == 0:
        return 1.0, 1.0, 1.0
    if pred_count == 0 or gt_count == 0:
        return 0.0, 0.0, 0.0

    dilated_gt = _dilate(boundary_gt, tolerance)
    dilated_pred = _dilate(boundary_pred, tolerance)
    pred_match = int(np.logical_and(boundary_pred, dilated_gt).sum())
    gt_match = int(np.logical_and(boundary_gt, dilated_pred).sum())

    precision = float(pred_match / (pred_count + 1e-6))
    recall = float(gt_match / (gt_count + 1e-6))
    f1 = float((2 * precision * recall) / (precision + recall + 1e-6))
    return f1, precision, recall


def resolve_tolerance(shape: Tuple[int, int], tol_px: Optional[int], tol_ratio: float) -> int:
    if tol_px is not None:
        return max(0, int(tol_px))
    diag = float(np.hypot(shape[0], shape[1]))
    return max(1, int(round(tol_ratio * diag)))


def match_instances(
    gt_masks: Sequence[np.ndarray],
    pred_masks: Sequence[np.ndarray],
    min_iou: float,
) -> Dict[int, int]:
    if not gt_masks or not pred_masks:
        return {}
    pairs: List[Tuple[float, int, int]] = []
    for gi, g in enumerate(gt_masks):
        for pi, p in enumerate(pred_masks):
            iou = compute_iou(g, p)
            if iou > min_iou:
                pairs.append((iou, gi, pi))
    pairs.sort(key=lambda item: item[0], reverse=True)
    gt_used = [False] * len(gt_masks)
    pred_used = [False] * len(pred_masks)
    matches: Dict[int, int] = {}
    for iou, gi, pi in pairs:
        if not gt_used[gi] and not pred_used[pi]:
            gt_used[gi] = True
            pred_used[pi] = True
            matches[gi] = pi
    return matches


def boundary_from_instances(
    instances: Sequence[np.ndarray],
    shape: Tuple[int, int],
    dilate: int = 0,
) -> np.ndarray:
    line = np.zeros(shape, dtype=bool)
    for inst in instances:
        line |= mask_boundary(inst)
    if dilate > 0:
        line = _dilate(line, dilate)
    return line
