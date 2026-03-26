from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from microseg.eval_ops import load_roi_instances


ROI_FILE_EXTS = (".roi", ".zip")
ID_MAP_FILE_EXTS = (".png", ".tif", ".tiff", ".bmp")


def is_eval_roi_file(path: Path) -> bool:
    return path.suffix.lower() in ROI_FILE_EXTS


def is_eval_id_map_file(path: Path) -> bool:
    return path.suffix.lower() in ID_MAP_FILE_EXTS


def is_eval_gt_supported_file(path: Path) -> bool:
    return is_eval_roi_file(path) or is_eval_id_map_file(path)


def find_eval_gt_file_in_dir(root: Path, image_stem: str, allow_shared_dir: bool) -> Optional[Path]:
    if not root.exists() or not root.is_dir():
        return None
    supported_files = sorted([p for p in root.rglob("*") if p.is_file() and is_eval_gt_supported_file(p)])
    if not supported_files:
        return None
    lower_stem = image_stem.lower()
    roi_matches = [p for p in supported_files if is_eval_roi_file(p) and p.stem.lower() == lower_stem]
    if roi_matches:
        return roi_matches[0]
    id_matches = [p for p in supported_files if is_eval_id_map_file(p) and p.stem.lower() == lower_stem]
    if id_matches:
        return id_matches[0]
    # Dedicated per-image directory may contain many ROI files with arbitrary names.
    if root.name.lower() == lower_stem:
        if any(is_eval_roi_file(p) for p in supported_files):
            return root
        id_files = [p for p in supported_files if is_eval_id_map_file(p)]
        if id_files:
            return id_files[0]
    if allow_shared_dir and len(supported_files) == 1:
        return supported_files[0]
    return None


def resolve_eval_gt_source(root: Path, image_stem: str, allow_shared_dir: bool) -> Optional[Path]:
    if not root.exists():
        return None
    if root.is_file():
        return root if is_eval_gt_supported_file(root) else None
    if not root.is_dir():
        return None

    direct_candidates = [root / f"{image_stem}.roi", root / f"{image_stem}.zip"]
    for ext in ID_MAP_FILE_EXTS:
        direct_candidates.append(root / f"{image_stem}{ext}")
    for cand in direct_candidates:
        if cand.exists() and cand.is_file():
            return cand

    lower_stem = image_stem.lower()
    for cand in root.iterdir():
        if not cand.is_file() or not is_eval_gt_supported_file(cand):
            continue
        if cand.stem.lower() == lower_stem:
            return cand

    per_image_dir = root / image_stem
    if per_image_dir.exists() and per_image_dir.is_dir():
        mapped = find_eval_gt_file_in_dir(per_image_dir, image_stem, allow_shared_dir=True)
        if mapped is not None:
            return mapped

    mapped_root = find_eval_gt_file_in_dir(root, image_stem, allow_shared_dir=allow_shared_dir)
    if mapped_root is not None:
        return mapped_root

    if allow_shared_dir and any(p.suffix.lower() == ".roi" for p in root.rglob("*.roi")):
        return root
    return None


def load_eval_instances_from_id_map(
    id_map_path: Path,
    shape: Tuple[int, int],
    min_area: int,
) -> List[np.ndarray]:
    id_img = cv2.imread(str(id_map_path), cv2.IMREAD_UNCHANGED)
    if id_img is None:
        raise FileNotFoundError(f"Could not read GT ID map: {id_map_path}")

    if id_img.ndim == 3:
        if id_img.shape[2] == 4:
            id_img = id_img[:, :, :3]
        if id_img.shape[2] >= 3:
            c0 = id_img[:, :, 0]
            c1 = id_img[:, :, 1]
            c2 = id_img[:, :, 2]
            if np.array_equal(c0, c1) and np.array_equal(c0, c2):
                id_map = c0.astype(np.int64, copy=False)
            else:
                # Support color-coded ID maps by packing BGR into one integer ID.
                id_map = (
                    c0.astype(np.uint32)
                    + (c1.astype(np.uint32) << 8)
                    + (c2.astype(np.uint32) << 16)
                ).astype(np.int64)
        else:
            id_map = id_img[:, :, 0].astype(np.int64, copy=False)
    else:
        id_map = id_img.astype(np.int64, copy=False)

    target_h, target_w = int(shape[0]), int(shape[1])
    if id_map.shape != (target_h, target_w):
        id_map = cv2.resize(
            id_map.astype(np.float32),
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int64)

    instances: List[np.ndarray] = []
    instance_ids = sorted(int(v) for v in np.unique(id_map) if int(v) > 0)
    for inst_id in instance_ids:
        mask = id_map == inst_id
        if int(mask.sum()) < int(min_area):
            continue
        instances.append(mask.astype(bool))

    # Fallback: binary GT map (single non-zero value) -> split connected components.
    if not instances:
        fg = (id_map > 0).astype(np.uint8)
        if int(fg.sum()) > 0:
            n_labels, labels = cv2.connectedComponents(fg, connectivity=8)
            for label_idx in range(1, int(n_labels)):
                comp = labels == label_idx
                if int(comp.sum()) < int(min_area):
                    continue
                instances.append(comp.astype(bool))
    return instances


def load_eval_gt_instances(
    source: Path,
    image_stem: str,
    shape: Tuple[int, int],
    min_area: int,
    tmp_root: Path,
    allow_shared_dir: bool,
) -> Sequence[np.ndarray]:
    source_to_load = source
    if source_to_load.is_dir():
        mapped = find_eval_gt_file_in_dir(source_to_load, image_stem, allow_shared_dir=allow_shared_dir)
        if mapped is not None and mapped != source_to_load:
            source_to_load = mapped

    if source_to_load.is_file() and is_eval_id_map_file(source_to_load):
        return load_eval_instances_from_id_map(source_to_load, shape, min_area)

    if source_to_load.is_dir():
        has_roi = any(p.suffix.lower() == ".roi" for p in source_to_load.rglob("*.roi"))
        if has_roi:
            return load_roi_instances(source_to_load, shape, min_area, tmp_root=tmp_root)
        mapped = find_eval_gt_file_in_dir(source_to_load, image_stem, allow_shared_dir=True)
        if mapped is not None and mapped.is_file() and is_eval_id_map_file(mapped):
            return load_eval_instances_from_id_map(mapped, shape, min_area)
        return []

    return load_roi_instances(source_to_load, shape, min_area, tmp_root=tmp_root)
