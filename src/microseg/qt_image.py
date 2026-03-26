from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from PySide6 import QtGui


def rgb_to_qpixmap(rgb: np.ndarray) -> QtGui.QPixmap:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape HxWx3, got {rgb.shape}")
    img = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, _ = img.shape
    qimg = QtGui.QImage(img.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimg.copy())


def bgr_to_qpixmap(bgr: np.ndarray) -> QtGui.QPixmap:
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"Expected BGR image with shape HxWx3, got {bgr.shape}")
    rgb = cv2.cvtColor(np.ascontiguousarray(bgr, dtype=np.uint8), cv2.COLOR_BGR2RGB)
    return rgb_to_qpixmap(rgb)


def pixmap_to_rgb(pix: Optional[QtGui.QPixmap]) -> Optional[np.ndarray]:
    if pix is None:
        return None
    qimg = pix.toImage().convertToFormat(QtGui.QImage.Format_RGB888)
    h, w = qimg.height(), qimg.width()
    buf = qimg.bits().tobytes()
    if not buf:
        return None
    bpl = qimg.bytesPerLine()
    arr = np.frombuffer(buf, np.uint8)
    if arr.size < h * bpl:
        return None
    arr = arr.reshape((h, bpl))
    arr = arr[:, : w * 3].reshape(h, w, 3)
    return arr.copy()


def fit_rgb_to_cell(
    image: Optional[np.ndarray],
    target_w: int,
    target_h: int,
    bg_color: Tuple[int, int, int] = (248, 250, 252),
) -> np.ndarray:
    canvas = np.full((max(1, target_h), max(1, target_w), 3), bg_color, dtype=np.uint8)
    if image is None or image.size == 0:
        return canvas
    ih, iw = image.shape[:2]
    if ih <= 0 or iw <= 0:
        return canvas
    scale = min(float(target_w) / float(iw), float(target_h) / float(ih))
    new_w = max(1, int(round(iw * scale)))
    new_h = max(1, int(round(ih * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
    y0 = max(0, (target_h - new_h) // 2)
    x0 = max(0, (target_w - new_w) // 2)
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas
