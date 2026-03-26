from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.spatial.distance import cdist

from microseg.geometry import mask_contours


# -----------------------------------------------------------------------------#
# Data structures                                                              #
# -----------------------------------------------------------------------------#
@dataclass
class MaskEntry:
    mask: np.ndarray
    raw: np.ndarray
    score: Optional[float]
    prompt_data: Optional[Dict[str, Any]] = None


# -----------------------------------------------------------------------------#
# Geometry / cleaning helpers                                                  #
# -----------------------------------------------------------------------------#
def choose_best_mask(masks: Sequence[np.ndarray], scores: Sequence[float]) -> Tuple[np.ndarray, float]:
    if not masks:
        raise ValueError("No masks returned")
    best_idx = int(np.argmax(scores))
    return masks[best_idx], float(scores[best_idx])


def clean_mask(mask: np.ndarray, min_area: int = 20, largest_only: bool = False) -> np.ndarray:
    if mask.sum() == 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(eroded, connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1
    if largest_only:
        keep = [largest_label]
    else:
        keep = [i + 1 for i, a in enumerate(areas) if a >= min_area]
        if largest_label not in keep:
            keep.append(largest_label)
    eroded_keep = np.isin(labels, keep).astype(np.uint8)
    dilated = cv2.dilate(eroded_keep, kernel, iterations=1)
    cleaned = np.logical_and(dilated > 0, mask > 0).astype(np.uint8)
    return cleaned


def mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
    y, x = np.nonzero(mask)
    if x.size == 0:
        return 0.0, 0.0
    return float(x.mean()), float(y.mean())


def _mask_feret_geometry(mask_u8: np.ndarray) -> Optional[Dict[str, Any]]:
    if int(mask_u8.sum()) <= 0:
        return None
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if contour is None or len(contour) < 2:
        return None
    hull = cv2.convexHull(contour, returnPoints=True)
    if hull is None:
        return None
    pts = hull.reshape(-1, 2).astype(np.float64)
    if pts.shape[0] < 2:
        return None

    # Major Feret diameter: maximum pair distance on convex hull vertices.
    best_d2 = -1.0
    best_i = 0
    best_j = 1
    n = int(pts.shape[0])
    for i in range(n - 1):
        pi = pts[i]
        diffs = pts[i + 1 :] - pi
        if diffs.size == 0:
            continue
        d2 = np.einsum("ij,ij->i", diffs, diffs)
        j_rel = int(np.argmax(d2))
        v = float(d2[j_rel])
        if v > best_d2:
            best_d2 = v
            best_i = i
            best_j = i + 1 + j_rel
    if not np.isfinite(best_d2) or best_d2 <= 0.0:
        return None
    major_len = float(np.sqrt(best_d2))
    major_p0 = pts[best_i]
    major_p1 = pts[best_j]

    # Minor Feret diameter: minimum caliper width over hull edge normals.
    best_width = None
    best_min_proj = None
    best_max_proj = None
    best_normal = None
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i + 1) % n]
        edge = p1 - p0
        edge_len = float(np.hypot(float(edge[0]), float(edge[1])))
        if edge_len <= 1e-9:
            continue
        normal = np.array([-edge[1], edge[0]], dtype=np.float64) / edge_len
        proj = pts @ normal
        pmin = float(np.min(proj))
        pmax = float(np.max(proj))
        width = float(pmax - pmin)
        if not np.isfinite(width):
            continue
        if best_width is None or width < best_width:
            best_width = width
            best_min_proj = pmin
            best_max_proj = pmax
            best_normal = normal
    if best_width is None or best_normal is None or best_min_proj is None or best_max_proj is None or best_width <= 0.0:
        return None

    center = pts.mean(axis=0)
    center_proj = float(center @ best_normal)
    minor_p0 = center + best_normal * (best_min_proj - center_proj)
    minor_p1 = center + best_normal * (best_max_proj - center_proj)

    return {
        "major_px": major_len,
        "minor_px": float(best_width),
        "major_seg": ((float(major_p0[0]), float(major_p0[1])), (float(major_p1[0]), float(major_p1[1]))),
        "minor_seg": ((float(minor_p0[0]), float(minor_p0[1])), (float(minor_p1[0]), float(minor_p1[1]))),
    }


def mask_feret_major_minor_segments_px(
    mask: np.ndarray,
) -> Optional[
    Tuple[
        Tuple[float, float],
        Tuple[float, float],
        Tuple[float, float],
        Tuple[float, float],
    ]
]:
    mask_u8 = (mask > 0).astype(np.uint8)
    geom = _mask_feret_geometry(mask_u8)
    if geom is None:
        return None
    ma0, ma1 = geom["major_seg"]
    mi0, mi1 = geom["minor_seg"]
    return ma0, ma1, mi0, mi1


def _mask_ellipse_geometry(mask_u8: np.ndarray) -> Optional[Dict[str, Any]]:
    if int(mask_u8.sum()) <= 0:
        return None
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if contour is None or len(contour) < 5:
        return None

    (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
    major_len = float(max(axis_a, axis_b))
    minor_len = float(min(axis_a, axis_b))
    if not np.isfinite(major_len) or not np.isfinite(minor_len):
        return None
    if major_len <= 0.0 or minor_len <= 0.0:
        return None

    theta = np.deg2rad(float(angle))
    if axis_b > axis_a:
        theta += np.pi * 0.5
    angle_major_deg = float(np.rad2deg(theta))
    center = np.array([float(cx), float(cy)], dtype=np.float64)
    major_vec = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    minor_vec = np.array([-major_vec[1], major_vec[0]], dtype=np.float64)
    major_half = major_len * 0.5
    minor_half = minor_len * 0.5
    major_p0 = center - major_vec * major_half
    major_p1 = center + major_vec * major_half
    minor_p0 = center - minor_vec * minor_half
    minor_p1 = center + minor_vec * minor_half

    return {
        "center": (float(center[0]), float(center[1])),
        "major_px": major_len,
        "minor_px": minor_len,
        "angle_major_deg": angle_major_deg,
        "major_seg": ((float(major_p0[0]), float(major_p0[1])), (float(major_p1[0]), float(major_p1[1]))),
        "minor_seg": ((float(minor_p0[0]), float(minor_p0[1])), (float(minor_p1[0]), float(minor_p1[1]))),
    }


def mask_ellipse_fit_params_px(
    mask: np.ndarray,
) -> Optional[Tuple[Tuple[float, float], float, float, float]]:
    mask_u8 = (mask > 0).astype(np.uint8)
    geom = _mask_ellipse_geometry(mask_u8)
    if geom is None:
        return None
    return (
        (float(geom["center"][0]), float(geom["center"][1])),
        float(geom["major_px"]),
        float(geom["minor_px"]),
        float(geom["angle_major_deg"]),
    )


def _feret_parallelogram_area_px(geom: Dict[str, Any]) -> Optional[float]:
    """Area spanned by Feret major/minor vectors (major*minor*sin(theta))."""
    major_seg = geom.get("major_seg")
    minor_seg = geom.get("minor_seg")
    if (
        not isinstance(major_seg, tuple)
        or not isinstance(minor_seg, tuple)
        or len(major_seg) != 2
        or len(minor_seg) != 2
    ):
        return None
    try:
        ma0 = np.array(major_seg[0], dtype=np.float64)
        ma1 = np.array(major_seg[1], dtype=np.float64)
        mi0 = np.array(minor_seg[0], dtype=np.float64)
        mi1 = np.array(minor_seg[1], dtype=np.float64)
    except Exception:
        return None
    if ma0.shape != (2,) or ma1.shape != (2,) or mi0.shape != (2,) or mi1.shape != (2,):
        return None
    v_major = ma1 - ma0
    v_minor = mi1 - mi0
    len_major = float(np.hypot(float(v_major[0]), float(v_major[1])))
    len_minor = float(np.hypot(float(v_minor[0]), float(v_minor[1])))
    if len_major <= 1e-9 or len_minor <= 1e-9:
        return None
    area = abs(float(v_major[0] * v_minor[1] - v_major[1] * v_minor[0]))
    if not np.isfinite(area):
        return None
    return area


def compute_mask_shape_metrics(mask: np.ndarray, scale_nm_per_px: float) -> Dict:
    mask_u8 = (mask > 0).astype(np.uint8)
    coords = np.argwhere(mask_u8 > 0)
    if coords.size == 0:
        return {
            "centroid_px": (None, None),
            "bbox_xywh_px": (None, None, None, None),
            "bbox_area_px": 0.0,
            "bbox_area_nm2": 0.0,
            "feret_rect_area_px": None,
            "feret_rect_area_nm2": None,
            "area_px": 0.0,
            "area_nm2": 0.0,
            "area_vesd_nm2": None,
            "ecd_nm": 0.0,
            "vesd_nm": None,
            "major_axis_px": None,
            "minor_axis_px": None,
            "major_axis_nm": None,
            "minor_axis_nm": None,
            "ellipse_major_axis_px": None,
            "ellipse_minor_axis_px": None,
            "ellipse_major_axis_nm": None,
            "ellipse_minor_axis_nm": None,
            "aspect_ratio": None,
            "shape_ratio": None,
        }

    ys = coords[:, 0]
    xs = coords[:, 1]
    centroid = (float(xs.mean()), float(ys.mean()))
    x0 = int(xs.min())
    y0 = int(ys.min())
    bw = int(xs.max() - xs.min() + 1)
    bh = int(ys.max() - ys.min() + 1)
    bbox_area_px = float(bw * bh)
    area_px = float(mask_u8.sum())
    scale = float(scale_nm_per_px)
    bbox_area_nm2 = bbox_area_px * (scale**2)
    area_nm2 = area_px * (scale**2)
    ecd_nm = 2.0 * np.sqrt(area_nm2 / np.pi) if area_nm2 > 0 else 0.0

    feret_geom = _mask_feret_geometry(mask_u8)
    if feret_geom is None:
        major_px = None
        minor_px = None
    else:
        major_px = float(feret_geom.get("major_px", 0.0) or 0.0)
        minor_px = float(feret_geom.get("minor_px", 0.0) or 0.0)
        if major_px <= 0.0 or minor_px <= 0.0:
            major_px = None
            minor_px = None
    ellipse_geom = _mask_ellipse_geometry(mask_u8)
    ellipse_major_px = float(ellipse_geom["major_px"]) if ellipse_geom is not None else None
    ellipse_minor_px = float(ellipse_geom["minor_px"]) if ellipse_geom is not None else None
    if major_px is None or minor_px is None:
        major_nm = None
        minor_nm = None
        feret_rect_area_px = None
        feret_rect_area_nm2 = None
        vesd_nm = None
        area_vesd_nm2 = None
        aspect_ratio = None
        shape_ratio = None
    else:
        major_nm = float(major_px * scale)
        minor_nm = float(minor_px * scale)
        # Use perpendicular component between major/minor (parallelogram), not rectangle area.
        para_area_px = _feret_parallelogram_area_px(feret_geom) if feret_geom is not None else None
        if para_area_px is None:
            para_area_px = float(major_px * minor_px)
        feret_rect_area_px = float(para_area_px)
        feret_rect_area_nm2 = float(feret_rect_area_px * (scale**2))
        # Equivalent sphere diameter from an axisymmetric ellipsoid (major x minor x minor).
        vesd_nm = float((major_nm * minor_nm * minor_nm) ** (1.0 / 3.0)) if major_nm > 0 and minor_nm > 0 else None
        area_vesd_nm2 = float(np.pi * (vesd_nm * 0.5) ** 2) if vesd_nm is not None and vesd_nm > 0 else None
        aspect_ratio = float(major_nm / minor_nm) if minor_nm > 0 else None
        shape_ratio = float(minor_nm / major_nm) if major_nm > 0 else None
    ellipse_major_nm = float(ellipse_major_px * scale) if ellipse_major_px is not None else None
    ellipse_minor_nm = float(ellipse_minor_px * scale) if ellipse_minor_px is not None else None

    return {
        "centroid_px": centroid,
        "bbox_xywh_px": (x0, y0, bw, bh),
        "bbox_area_px": bbox_area_px,
        "bbox_area_nm2": bbox_area_nm2,
        "feret_rect_area_px": feret_rect_area_px,
        "feret_rect_area_nm2": feret_rect_area_nm2,
        "area_px": area_px,
        "area_nm2": area_nm2,
        "area_vesd_nm2": area_vesd_nm2,
        "ecd_nm": ecd_nm,
        "vesd_nm": vesd_nm,
        "major_axis_px": major_px,
        "minor_axis_px": minor_px,
        "major_axis_nm": major_nm,
        "minor_axis_nm": minor_nm,
        "ellipse_major_axis_px": ellipse_major_px,
        "ellipse_minor_axis_px": ellipse_minor_px,
        "ellipse_major_axis_nm": ellipse_major_nm,
        "ellipse_minor_axis_nm": ellipse_minor_nm,
        "aspect_ratio": aspect_ratio,
        "shape_ratio": shape_ratio,
    }


# -----------------------------------------------------------------------------#
# Box-counting / fractal                                                       #
# -----------------------------------------------------------------------------#
def boxcount_counts(mask: np.ndarray, slides: int = 1) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    m = mask.astype(bool)
    h, w = m.shape
    min_side = min(h, w)
    size_min = 5
    if min_side < size_min:
        return None
    size_max = max(size_min, int(min_side * 0.05))
    size_max = min(size_max, min_side)
    sizes = list(range(size_min, size_max + 1))
    if not sizes:
        return None
    counts = []
    for s in sizes:
        total = 0.0
        used = 0
        for k in range(slides):
            oy = (k * 37) % s
            ox = (k * 53) % s
            h_c = (h - oy) // s
            w_c = (w - ox) // s
            if h_c <= 0 or w_c <= 0:
                continue
            cropped = m[oy : oy + h_c * s, ox : ox + w_c * s]
            blocks = cropped.reshape(h_c, s, w_c, s)
            block_any = blocks.any(axis=(1, 3))
            total += float(block_any.sum())
            used += 1
        counts.append(total / used if used > 0 else 0.0)
    return np.array(counts, dtype=np.float32), np.array(sizes, dtype=np.float32)


def boxcount_fractal_dimension(mask: np.ndarray, slides: int = 1) -> Optional[float]:
    res = boxcount_counts(mask, slides=slides)
    if res is None:
        return None
    counts, sizes = res
    log_eps = np.log(sizes)
    valid = counts > 0
    if valid.sum() < 2:
        return None
    slope, _ = np.polyfit(log_eps[valid], np.log(counts[valid]), 1)
    return float(slope)


# -----------------------------------------------------------------------------#
# Distance computation                                                         #
# -----------------------------------------------------------------------------#
def compute_two_nearest(
    masks: Sequence[np.ndarray],
    scale_nm_per_px: float,
    max_distance_nm: Optional[float] = None,
    include_zero: bool = False,
) -> List[Dict]:
    contours = [mask_contours(m) for m in masks]
    centroids = [mask_centroid(m) for m in masks]
    scale = float(scale_nm_per_px)
    max_px = None if max_distance_nm is None else max_distance_nm / scale
    results: List[Dict] = []
    for i, ci in enumerate(contours):
        if ci.size == 0:
            results.append({"nearest": [], "nearest_centroid": [], "nearest_all": [], "nearest_centroid_all": []})
            continue
        candidates = []
        cand_centroids = []
        for j, cj in enumerate(contours):
            if i == j or cj.size == 0:
                continue
            dist_mat = cdist(ci, cj)
            min_idx = np.unravel_index(np.argmin(dist_mat), dist_mat.shape)
            dist_val = float(dist_mat[min_idx])
            if dist_val < 0 or (dist_val == 0 and not include_zero):
                continue
            if max_px is not None and dist_val > max_px:
                continue
            pa = tuple(map(float, ci[min_idx[0]]))
            pb = tuple(map(float, cj[min_idx[1]]))
            candidates.append(
                {
                    "index": j,
                    "distance_px": dist_val,
                    "distance_nm": dist_val * scale,
                    "contour_point_a": pa,
                    "contour_point_b": pb,
                }
            )
            cx0, cy0 = centroids[i]
            cx1, cy1 = centroids[j]
            cdist_px = float(np.hypot(cx0 - cx1, cy0 - cy1))
            if cdist_px < 0 or (cdist_px == 0 and not include_zero):
                pass
            elif max_px is not None and cdist_px > max_px:
                pass
            else:
                cand_centroids.append(
                    {
                        "index": j,
                        "distance_px": cdist_px,
                        "distance_nm": cdist_px * scale,
                        "centroid_a": (cx0, cy0),
                        "centroid_b": (cx1, cy1),
                    }
                )
        candidates.sort(key=lambda x: x["distance_px"])
        cand_centroids.sort(key=lambda x: x["distance_px"])
        results.append(
            {
                "nearest": candidates[:2],
                "nearest_centroid": cand_centroids[:2],
                "nearest_all": candidates,
                "nearest_centroid_all": cand_centroids,
            }
        )
    return results


# -----------------------------------------------------------------------------#
# Summaries                                                                    #
# -----------------------------------------------------------------------------#
def summarize(pair_results: List[Dict]) -> Dict:
    first = []
    second = []
    c_first = []
    c_second = []
    for res in pair_results:
        near = res.get("nearest", [])
        cnear = res.get("nearest_centroid", [])
        if len(near) >= 1:
            first.append(near[0]["distance_nm"])
        if len(near) >= 2:
            second.append(near[1]["distance_nm"])
        if len(cnear) >= 1:
            c_first.append(cnear[0]["distance_nm"])
        if len(cnear) >= 2:
            c_second.append(cnear[1]["distance_nm"])

    def stats(arr):
        if not arr:
            return {
                "count": 0,
                "mean": None,
                "median": None,
                "std": None,
                "cv_pct": None,
                "min": None,
                "max": None,
            }
        arr_np = np.array(arr, dtype=np.float32)
        mean = float(arr_np.mean())
        median = float(np.median(arr_np))
        std = float(arr_np.std())
        cv_pct = float((std / mean) * 100.0) if abs(mean) > 1e-12 else None
        return {
            "count": int(len(arr)),
            "mean": mean,
            "median": median,
            "std": std,
            "cv_pct": cv_pct,
            "min": float(arr_np.min()),
            "max": float(arr_np.max()),
        }

    return {
        "nearest1": stats(first),
        "nearest2": stats(second),
        "hist_first": first,
        "hist_second": second,
        "centroid1": stats(c_first),
        "centroid2": stats(c_second),
        "hist_centroid1": c_first,
        "hist_centroid2": c_second,
    }


def summarize_sizes(masks: Sequence[np.ndarray], scale_nm_per_px: float, fractal_slides: int = 0) -> Dict:
    ecd = []
    vesd = []
    area = []
    area_vesd = []
    bbox_area = []
    feret_rect_area = []
    major_axis = []
    minor_axis = []
    ellipse_major_axis = []
    ellipse_minor_axis = []
    aspect_ratio = []
    shape_ratio = []
    frac = []
    frac_counts_list: List[np.ndarray] = []
    frac_sizes: Optional[np.ndarray] = None
    union_mask = None
    scale = float(scale_nm_per_px)
    for m in masks:
        metrics = compute_mask_shape_metrics(m, scale)
        area_px = float(metrics["area_px"])
        if area_px <= 0:
            continue
        area.append(float(metrics["area_nm2"]))
        if metrics.get("bbox_area_nm2") is not None:
            bbox_area.append(float(metrics["bbox_area_nm2"]))
        if metrics.get("feret_rect_area_nm2") is not None:
            feret_rect_area.append(float(metrics["feret_rect_area_nm2"]))
        if metrics.get("area_vesd_nm2") is not None:
            area_vesd.append(float(metrics["area_vesd_nm2"]))
        ecd.append(float(metrics["ecd_nm"]))
        if metrics.get("vesd_nm") is not None:
            vesd.append(float(metrics["vesd_nm"]))
        if metrics["major_axis_nm"] is not None:
            major_axis.append(float(metrics["major_axis_nm"]))
        if metrics["minor_axis_nm"] is not None:
            minor_axis.append(float(metrics["minor_axis_nm"]))
        if metrics.get("ellipse_major_axis_nm") is not None:
            ellipse_major_axis.append(float(metrics["ellipse_major_axis_nm"]))
        if metrics.get("ellipse_minor_axis_nm") is not None:
            ellipse_minor_axis.append(float(metrics["ellipse_minor_axis_nm"]))
        if metrics["aspect_ratio"] is not None:
            aspect_ratio.append(float(metrics["aspect_ratio"]))
        if metrics["shape_ratio"] is not None:
            shape_ratio.append(float(metrics["shape_ratio"]))
        union_mask = m if union_mask is None else np.logical_or(union_mask > 0, m > 0).astype(np.uint8)
        bc = boxcount_counts(m, slides=fractal_slides)
        if bc is not None:
            counts, sizes = bc
            if frac_sizes is None:
                frac_sizes = sizes
            if frac_sizes.shape == sizes.shape and np.allclose(frac_sizes, sizes):
                frac_counts_list.append(counts)
                fd = boxcount_fractal_dimension(m, slides=fractal_slides)
                if fd is not None:
                    frac.append(fd)

    def stats(arr):
        if not arr:
            return {
                "count": 0,
                "mean": None,
                "median": None,
                "std": None,
                "cv_pct": None,
                "min": None,
                "max": None,
            }
        arr_np = np.array(arr, dtype=np.float32)
        mean = float(arr_np.mean())
        median = float(np.median(arr_np))
        std = float(arr_np.std())
        cv_pct = float((std / mean) * 100.0) if abs(mean) > 1e-12 else None
        return {
            "count": int(len(arr)),
            "mean": mean,
            "median": median,
            "std": std,
            "cv_pct": cv_pct,
            "min": float(arr_np.min()),
            "max": float(arr_np.max()),
        }

    fractal_curve = {"log_eps": [], "log_counts": [], "slope": None, "sizes_px": [], "counts_mean": []}
    if frac_sizes is not None and frac_counts_list:
        counts_stack = np.vstack(frac_counts_list)
        counts_mean = counts_stack.mean(axis=0)
        fractal_curve["sizes_px"] = list(map(float, frac_sizes))
        fractal_curve["counts_mean"] = list(map(float, counts_mean))
        with np.errstate(divide="ignore"):
            log_eps_all = np.log(frac_sizes)
            log_counts = np.log(counts_stack)
        valid_mask = ~np.isinf(log_counts)
        mean_log_counts = []
        mean_log_eps = []
        for i in range(log_counts.shape[1]):
            vals = log_counts[:, i][valid_mask[:, i]]
            if vals.size > 0:
                mean_log_counts.append(float(vals.mean()))
                mean_log_eps.append(float(log_eps_all[i]))
        if len(mean_log_counts) >= 2:
            slope, _ = np.polyfit(np.array(mean_log_eps), np.array(mean_log_counts), 1)
            fractal_curve["slope"] = float(slope)
        fractal_curve["log_eps"] = mean_log_eps
        fractal_curve["log_counts"] = mean_log_counts

    fractal_global = {"value": None, "log_eps": [], "log_counts": [], "slope": None, "sizes_px": [], "counts": []}
    if union_mask is not None:
        bc_global = boxcount_counts(union_mask, slides=fractal_slides)
        if bc_global is not None:
            counts_g, sizes_g = bc_global
            fractal_global["sizes_px"] = list(map(float, sizes_g))
            fractal_global["counts"] = list(map(float, counts_g))
            valid_g = counts_g > 0
            if valid_g.sum() >= 2:
                log_eps_g = np.log(sizes_g[valid_g])
                log_counts_g = np.log(counts_g[valid_g])
                slope_g, _ = np.polyfit(log_eps_g, log_counts_g, 1)
                fractal_global["slope"] = float(slope_g)
                fractal_global["value"] = -float(slope_g)
                fractal_global["log_eps"] = list(map(float, log_eps_g))
                fractal_global["log_counts"] = list(map(float, log_counts_g))

    volume_mean_diameter = None
    if vesd:
        arr = np.asarray(vesd, dtype=np.float64)
        denom = float(np.sum(arr**3))
        if denom > 0:
            volume_mean_diameter = float(np.sum(arr**4) / denom)

    return {
        "ecd": stats(ecd),
        "hist_ecd": ecd,
        "vesd": stats(vesd),
        "hist_vesd": vesd,
        "volume_mean_diameter": volume_mean_diameter,
        "area": stats(area),
        "hist_area": area,
        "bbox_area": stats(bbox_area),
        "hist_bbox_area": bbox_area,
        "feret_rect_area": stats(feret_rect_area),
        "hist_feret_rect_area": feret_rect_area,
        "area_vesd": stats(area_vesd),
        "hist_area_vesd": area_vesd,
        "major_axis": stats(major_axis),
        "hist_major_axis": major_axis,
        "minor_axis": stats(minor_axis),
        "hist_minor_axis": minor_axis,
        "ellipse_major_axis": stats(ellipse_major_axis),
        "hist_ellipse_major_axis": ellipse_major_axis,
        "ellipse_minor_axis": stats(ellipse_minor_axis),
        "hist_ellipse_minor_axis": ellipse_minor_axis,
        "aspect_ratio": stats(aspect_ratio),
        "hist_aspect_ratio": aspect_ratio,
        "shape_ratio": stats(shape_ratio),
        "hist_shape_ratio": shape_ratio,
        "fractal": stats(frac),
        "hist_fractal": frac,
        "fractal_curve": fractal_curve,
        "fractal_global": fractal_global,
    }


# -----------------------------------------------------------------------------#
# Persistence                                                                  #
# -----------------------------------------------------------------------------#
def save_payload(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
