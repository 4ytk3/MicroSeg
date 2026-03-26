from __future__ import annotations

import numpy as np
from skimage import measure


def mask_contours(mask: np.ndarray) -> np.ndarray:
    contours = measure.find_contours(mask, 0.5)
    if not contours:
        return np.empty((0, 2), dtype=np.float32)
    return np.vstack(contours).astype(np.float32)
