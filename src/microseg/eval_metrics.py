from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from microseg.eval_ops import (
    boundary_fscore,
    compute_dice,
    compute_iou,
    match_instances,
    resolve_tolerance,
)


def evaluate_image_metrics(
    image_name: str,
    image_shape: Sequence[int],
    pred_masks: Sequence[np.ndarray],
    gt_instances: Sequence[np.ndarray],
    match_iou: float,
    boundary_ratio: float,
    image_idx: int,
) -> Dict[str, Any]:
    shape_hw = (int(image_shape[0]), int(image_shape[1]))
    tolerance = resolve_tolerance(shape_hw, None, boundary_ratio)
    matches = match_instances(gt_instances, pred_masks, match_iou)

    iou_vals: List[float] = []
    dice_vals: List[float] = []
    bf1_vals: List[float] = []
    for gt_idx, gt_mask in enumerate(gt_instances):
        pred_idx = matches.get(gt_idx)
        pred_mask = pred_masks[pred_idx] if pred_idx is not None else np.zeros_like(gt_mask, dtype=bool)
        iou_vals.append(float(compute_iou(gt_mask, pred_mask)))
        dice_vals.append(float(compute_dice(gt_mask, pred_mask)))
        bf1, _bp, _br = boundary_fscore(pred_mask, gt_mask, tolerance)
        bf1_vals.append(float(bf1))

    return {
        "image_idx": int(image_idx),
        "image_name": str(image_name),
        "gt_count": len(gt_instances),
        "pred_count": len(pred_masks),
        "matched_count": len(matches),
        "iou": iou_vals,
        "dice": dice_vals,
        "bf1": bf1_vals,
    }


def run_eval_scope(
    indices: Sequence[int],
    image_sessions: Sequence[Any],
    pred_min_area: int,
    current_only: bool,
    allow_shared_for_all: bool,
    get_gt_instances: Callable[[int, bool, bool], Optional[Sequence[np.ndarray]]],
    match_iou: float,
    boundary_ratio: float,
) -> Dict[str, Any]:
    iou_all: List[float] = []
    dice_all: List[float] = []
    bf1_all: List[float] = []
    gt_total = 0
    pred_total = 0
    matched_total = 0
    images_eval = 0
    skipped_no_mask = 0
    skipped_no_gt = 0

    for idx in indices:
        if idx < 0 or idx >= len(image_sessions):
            continue
        state = image_sessions[idx]
        pred_masks: List[np.ndarray] = []
        for rec in state.set_masks:
            mask_bool = rec.mask.astype(bool)
            if int(mask_bool.sum()) < int(pred_min_area):
                continue
            pred_masks.append(mask_bool)
        if not pred_masks:
            skipped_no_mask += 1
            continue

        gt_instances = get_gt_instances(
            idx,
            True if current_only else allow_shared_for_all,
            not current_only,
        )
        if not gt_instances:
            skipped_no_gt += 1
            continue

        res = evaluate_image_metrics(
            image_name=state.image_path.name,
            image_shape=state.image_bgr.shape[:2],
            pred_masks=pred_masks,
            gt_instances=gt_instances,
            match_iou=match_iou,
            boundary_ratio=boundary_ratio,
            image_idx=idx,
        )
        images_eval += 1
        gt_total += int(res["gt_count"])
        pred_total += int(res["pred_count"])
        matched_total += int(res["matched_count"])
        iou_all.extend(res["iou"])
        dice_all.extend(res["dice"])
        bf1_all.extend(res["bf1"])

    return {
        "current_only": bool(current_only),
        "images_eval": images_eval,
        "images_total": len(indices),
        "skipped_no_mask": skipped_no_mask,
        "skipped_no_gt": skipped_no_gt,
        "gt_total": gt_total,
        "pred_total": pred_total,
        "matched_total": matched_total,
        "iou": iou_all,
        "dice": dice_all,
        "bf1": bf1_all,
    }
