from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_image_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def load_mask_dir(mask_dir: Path, image_shape: Tuple[int, int]) -> List[np.ndarray]:
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
    mask_files = sorted([p for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() in (".png", ".tif", ".tiff")])
    masks: List[np.ndarray] = []
    for path in mask_files:
        m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape != image_shape:
            m = cv2.resize(m, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
        masks.append((m > 0).astype(np.uint8))
    if not masks:
        raise ValueError(f"No masks found in {mask_dir}")
    return masks
