from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional


def _normalize_length_unit(length_unit: str) -> str:
    u = (length_unit or "nm").strip().lower().replace("µ", "u").replace("μ", "u")
    if u not in {"nm", "um", "mm"}:
        return "nm"
    return u


def _length_from_nm(value_nm: Optional[float], length_unit: str) -> Optional[float]:
    if value_nm is None:
        return None
    unit = _normalize_length_unit(length_unit)
    scale = 1.0
    if unit == "um":
        scale = 1000.0
    elif unit == "mm":
        scale = 1000.0 * 1000.0
    return float(value_nm) / scale


def _area_from_nm2(value_nm2: Optional[float], length_unit: str) -> Optional[float]:
    if value_nm2 is None:
        return None
    unit = _normalize_length_unit(length_unit)
    scale = 1.0
    if unit == "um":
        scale = 1000.0
    elif unit == "mm":
        scale = 1000.0 * 1000.0
    return float(value_nm2) / (scale * scale)


def _convert_stats_value(stats: Dict, key: str, kind: str, length_unit: str) -> Optional[float]:
    if not stats:
        return None
    raw = stats.get(key)
    if raw is None:
        return None
    if key in {"count", "cv_pct"}:
        return raw
    if kind == "length":
        return _length_from_nm(raw, length_unit)
    if kind == "area":
        return _area_from_nm2(raw, length_unit)
    return raw


def write_csv_summary(
    out_dir: Path,
    summary: Dict,
    size_summary: Dict,
    cluster_stats: Dict,
    length_unit: str = "nm",
    filename: str = "summary.csv",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / str(filename or "summary.csv")
    unit = _normalize_length_unit(length_unit)
    headers = [
        "metric",
        "length_unit",
        "area_unit",
        "N1",
        "N2",
        "Cent1",
        "Cent2",
        "ECD",
        "VESD",
        "Area(px)",
        "BBox Area",
        "Area(VESD)",
        "FeretMaj",
        "FeretMin",
        "Area(FeretRect)",
        "EllipseMaj",
        "EllipseMin",
        "Aspect",
        "Shape",
        "VolMeanDia",
        "Cluster",
    ]
    rows = []
    for key, label in [("mean", "mean"), ("median", "median"), ("std", "std"), ("cv_pct", "cv(%)"), ("min", "min"), ("max", "max")]:
        row = {"metric": label}
        row["length_unit"] = unit
        row["area_unit"] = f"{unit}2"
        row["N1"] = _convert_stats_value(summary.get("nearest1", {}), key, "length", unit) if summary else None
        row["N2"] = _convert_stats_value(summary.get("nearest2", {}), key, "length", unit) if summary else None
        row["Cent1"] = _convert_stats_value(summary.get("centroid1", {}), key, "length", unit) if summary else None
        row["Cent2"] = _convert_stats_value(summary.get("centroid2", {}), key, "length", unit) if summary else None
        row["ECD"] = _convert_stats_value(size_summary.get("ecd", {}), key, "length", unit) if size_summary else None
        row["VESD"] = _convert_stats_value(size_summary.get("vesd", {}), key, "length", unit) if size_summary else None
        row["Area(px)"] = _convert_stats_value(size_summary.get("area", {}), key, "area", unit) if size_summary else None
        row["BBox Area"] = _convert_stats_value(size_summary.get("bbox_area", {}), key, "area", unit) if size_summary else None
        row["Area(VESD)"] = _convert_stats_value(size_summary.get("area_vesd", {}), key, "area", unit) if size_summary else None
        row["FeretMaj"] = _convert_stats_value(size_summary.get("major_axis", {}), key, "length", unit) if size_summary else None
        row["FeretMin"] = _convert_stats_value(size_summary.get("minor_axis", {}), key, "length", unit) if size_summary else None
        row["Area(FeretRect)"] = _convert_stats_value(size_summary.get("feret_rect_area", {}), key, "area", unit) if size_summary else None
        row["EllipseMaj"] = _convert_stats_value(size_summary.get("ellipse_major_axis", {}), key, "length", unit) if size_summary else None
        row["EllipseMin"] = _convert_stats_value(size_summary.get("ellipse_minor_axis", {}), key, "length", unit) if size_summary else None
        row["Aspect"] = size_summary.get("aspect_ratio", {}).get(key) if size_summary else None
        row["Shape"] = size_summary.get("shape_ratio", {}).get(key) if size_summary else None
        row["VolMeanDia"] = _length_from_nm(size_summary.get("volume_mean_diameter"), unit) if (size_summary and key == "mean") else None
        row["Cluster"] = cluster_stats.get(key) if cluster_stats else None
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_mask_csv(out_dir: Path, masks: list[dict], length_unit: str = "nm") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "masks.csv"
    unit = _normalize_length_unit(length_unit)
    ecd_col = f"ecd_{unit}"
    area_col = f"area_{unit}2"
    bbox_area_col = f"bbox_area_{unit}2"
    area_vesd_col = f"area_vesd_{unit}2"
    major_col = f"major_axis_{unit}"
    minor_col = f"minor_axis_{unit}"
    feret_rect_area_col = f"feret_rect_area_{unit}2"
    ellipse_major_col = f"ellipse_major_axis_{unit}"
    ellipse_minor_col = f"ellipse_minor_axis_{unit}"
    vesd_col = f"vesd_{unit}"
    headers = [
        "index",
        area_col,
        bbox_area_col,
        area_vesd_col,
        ecd_col,
        vesd_col,
        major_col,
        minor_col,
        feret_rect_area_col,
        ellipse_major_col,
        ellipse_minor_col,
        "aspect_ratio",
        "score",
        "centroid_x_px",
        "centroid_y_px",
        "bbox_x_px",
        "bbox_y_px",
        "bbox_cx_px",
        "bbox_cy_px",
        "length_unit",
        "area_unit",
        "score_text",
        "bbox_w_px",
        "bbox_h_px",
        "bbox_text",
        "centroid_text",
        "area_px",
        "bbox_area_px",
        "major_axis_px",
        "minor_axis_px",
        "feret_rect_area_px",
        "ellipse_major_axis_px",
        "ellipse_minor_axis_px",
        "shape_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for m in masks:
            cx, cy = m.get("centroid_px", (None, None))
            bx, by, bw, bh = m.get("bbox_xywh_px", (None, None, None, None))
            bbox_cx = None
            bbox_cy = None
            if None not in (bx, by, bw, bh):
                bbox_cx = float(bx) + (float(bw) - 1.0) * 0.5
                bbox_cy = float(by) + (float(bh) - 1.0) * 0.5
            score_raw = m.get("score")
            try:
                score_text = "N/A" if score_raw is None else f"{float(score_raw):.3f}"
            except Exception:
                score_text = "N/A"
            bbox_text = "-"
            if bw is not None and bh is not None and bx is not None and by is not None:
                bbox_text = f"{int(bw)}x{int(bh)}@{int(bx)}, {int(by)}"
            centroid_text = "-"
            if cx is not None and cy is not None:
                centroid_text = f"{int(round(float(cx)))}, {int(round(float(cy)))}"
            writer.writerow(
                {
                    "index": m.get("index"),
                    area_col: _area_from_nm2(m.get("area_nm2"), unit),
                    bbox_area_col: _area_from_nm2(m.get("bbox_area_nm2"), unit),
                    area_vesd_col: _area_from_nm2(m.get("area_vesd_nm2"), unit),
                    ecd_col: _length_from_nm(m.get("ecd_nm"), unit),
                    vesd_col: _length_from_nm(m.get("vesd_nm"), unit),
                    major_col: _length_from_nm(m.get("major_axis_nm"), unit),
                    minor_col: _length_from_nm(m.get("minor_axis_nm"), unit),
                    feret_rect_area_col: _area_from_nm2(m.get("feret_rect_area_nm2"), unit),
                    ellipse_major_col: _length_from_nm(m.get("ellipse_major_axis_nm"), unit),
                    ellipse_minor_col: _length_from_nm(m.get("ellipse_minor_axis_nm"), unit),
                    "aspect_ratio": m.get("aspect_ratio"),
                    "score": m.get("score"),
                    "centroid_x_px": cx,
                    "centroid_y_px": cy,
                    "bbox_x_px": bx,
                    "bbox_y_px": by,
                    "bbox_cx_px": bbox_cx,
                    "bbox_cy_px": bbox_cy,
                    "length_unit": unit,
                    "area_unit": f"{unit}2",
                    "score_text": score_text,
                    "bbox_w_px": bw,
                    "bbox_h_px": bh,
                    "bbox_text": bbox_text,
                    "centroid_text": centroid_text,
                    "area_px": m.get("area_px"),
                    "bbox_area_px": m.get("bbox_area_px"),
                    "major_axis_px": m.get("major_axis_px"),
                    "minor_axis_px": m.get("minor_axis_px"),
                    "feret_rect_area_px": m.get("feret_rect_area_px"),
                    "ellipse_major_axis_px": m.get("ellipse_major_axis_px"),
                    "ellipse_minor_axis_px": m.get("ellipse_minor_axis_px"),
                    "shape_ratio": m.get("shape_ratio"),
                }
            )


def _write_columns(path: Path, headers: list[str], columns: list[list]) -> None:
    max_len = max((len(col) for col in columns), default=0)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i in range(max_len):
            row = []
            for col in columns:
                row.append(col[i] if i < len(col) else "")
            writer.writerow(row)


def write_hist_nearest_csv(
    out_dir: Path,
    hist_first: list,
    hist_second: list,
    hist_centroid1: list,
    hist_centroid2: list,
    length_unit: str = "nm",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "hist_nearest.csv"
    unit = _normalize_length_unit(length_unit)
    headers = [
        f"n1_surface_{unit}",
        f"n2_surface_{unit}",
        f"n1_centroid_{unit}",
        f"n2_centroid_{unit}",
    ]
    _write_columns(
        path,
        headers,
        [
            [_length_from_nm(v, unit) for v in hist_first],
            [_length_from_nm(v, unit) for v in hist_second],
            [_length_from_nm(v, unit) for v in hist_centroid1],
            [_length_from_nm(v, unit) for v in hist_centroid2],
        ],
    )


def write_hist_size_csv(
    out_dir: Path,
    hist_ecd: list,
    hist_area: list,
    hist_fractal: list | None = None,
    hist_major_axis: list | None = None,
    hist_minor_axis: list | None = None,
    hist_feret_rect_area: list | None = None,
    hist_aspect_ratio: list | None = None,
    hist_shape_ratio: list | None = None,
    length_unit: str = "nm",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "hist_size.csv"
    unit = _normalize_length_unit(length_unit)
    headers = [f"ecd_{unit}", f"area_{unit}2"]
    columns = [
        [_length_from_nm(v, unit) for v in hist_ecd],
        [_area_from_nm2(v, unit) for v in hist_area],
    ]
    if hist_major_axis is not None:
        headers.append(f"major_axis_{unit}")
        columns.append([_length_from_nm(v, unit) for v in hist_major_axis])
    if hist_minor_axis is not None:
        headers.append(f"minor_axis_{unit}")
        columns.append([_length_from_nm(v, unit) for v in hist_minor_axis])
    if hist_feret_rect_area is not None:
        headers.append(f"feret_rect_area_{unit}2")
        columns.append([_area_from_nm2(v, unit) for v in hist_feret_rect_area])
    if hist_aspect_ratio is not None:
        headers.append("aspect_ratio")
        columns.append(hist_aspect_ratio)
    if hist_shape_ratio is not None:
        headers.append("shape_ratio")
        columns.append(hist_shape_ratio)
    if hist_fractal is not None:
        headers.append("fractal_dim")
        columns.append(hist_fractal)
    _write_columns(path, headers, columns)


def write_fractal_curve_csv(out_dir: Path, log_eps: list, log_counts: list, slope: Optional[float]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fractal_curve.csv"
    headers = ["log_eps", "log_counts", "slope"]
    slope_col = [slope] * len(log_eps) if log_eps else []
    _write_columns(path, headers, [log_eps, log_counts, slope_col])


def write_fractal_global_csv(out_dir: Path, log_eps: list, log_counts: list, slope: Optional[float], value: Optional[float]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fractal_global.csv"
    headers = ["log_eps", "log_counts", "slope", "value"]
    slope_col = [slope] * len(log_eps) if log_eps else []
    value_col = [value] * len(log_eps) if log_eps else []
    _write_columns(path, headers, [log_eps, log_counts, slope_col, value_col])


def write_boxcount_curve_csv(out_dir: Path, sizes_px: list, counts_mean: list) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "boxcount_curve.csv"
    headers = ["size_px", "count_mean"]
    _write_columns(path, headers, [sizes_px, counts_mean])


def write_boxcount_global_csv(out_dir: Path, sizes_px: list, counts: list) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "boxcount_global.csv"
    headers = ["size_px", "count"]
    _write_columns(path, headers, [sizes_px, counts])


def write_all_images_summary_csv(out_dir: Path, rows: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "all_images_summary.csv"
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_index", "image_name", "image_path", "output_dir", "mask_count"])
        return

    preferred = [
        "image_index",
        "image_name",
        "image_path",
        "output_dir",
        "mask_count",
        "length_unit",
        "area_unit",
        "n1_mean",
        "n1_std",
        "n1_cv_pct",
        "n2_mean",
        "n2_std",
        "n2_cv_pct",
        "cent1_mean",
        "cent1_std",
        "cent1_cv_pct",
        "cent2_mean",
        "cent2_std",
        "cent2_cv_pct",
        "ecd_mean",
        "ecd_std",
        "ecd_cv_pct",
        "vesd_mean",
        "vesd_std",
        "vesd_cv_pct",
        "area_mean",
        "area_std",
        "area_cv_pct",
        "bbox_area_mean",
        "bbox_area_std",
        "bbox_area_cv_pct",
        "area_vesd_mean",
        "area_vesd_std",
        "area_vesd_cv_pct",
        "major_axis_mean",
        "major_axis_std",
        "major_axis_cv_pct",
        "minor_axis_mean",
        "minor_axis_std",
        "minor_axis_cv_pct",
        "feret_rect_area_mean",
        "feret_rect_area_std",
        "feret_rect_area_cv_pct",
        "ellipse_major_axis_mean",
        "ellipse_major_axis_std",
        "ellipse_major_axis_cv_pct",
        "ellipse_minor_axis_mean",
        "ellipse_minor_axis_std",
        "ellipse_minor_axis_cv_pct",
        "aspect_ratio_mean",
        "aspect_ratio_std",
        "aspect_ratio_cv_pct",
        "shape_ratio_mean",
        "shape_ratio_std",
        "shape_ratio_cv_pct",
        "volume_mean_diameter",
        "cluster_count",
        "cluster_mean_size",
        "cluster_std_size",
        "cluster_cv_pct",
    ]
    discovered = set()
    for row in rows:
        discovered.update(row.keys())
    headers = [h for h in preferred if h in discovered]
    for key in sorted(discovered):
        if key not in headers:
            headers.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_all_images_index_csv(out_dir: Path, rows: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "all_images_index.csv"
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_index", "image_name", "image_path", "output_dir", "mask_count"])
        return

    preferred = [
        "image_index",
        "image_name",
        "image_path",
        "output_dir",
        "mask_count",
        "length_unit",
        "area_unit",
        "n1_mean",
        "n1_std",
        "n1_cv_pct",
        "n2_mean",
        "n2_std",
        "n2_cv_pct",
        "cent1_mean",
        "cent1_std",
        "cent1_cv_pct",
        "cent2_mean",
        "cent2_std",
        "cent2_cv_pct",
        "ecd_mean",
        "ecd_std",
        "ecd_cv_pct",
        "vesd_mean",
        "vesd_std",
        "vesd_cv_pct",
        "area_mean",
        "area_std",
        "area_cv_pct",
        "bbox_area_mean",
        "bbox_area_std",
        "bbox_area_cv_pct",
        "area_vesd_mean",
        "area_vesd_std",
        "area_vesd_cv_pct",
        "major_axis_mean",
        "major_axis_std",
        "major_axis_cv_pct",
        "minor_axis_mean",
        "minor_axis_std",
        "minor_axis_cv_pct",
        "feret_rect_area_mean",
        "feret_rect_area_std",
        "feret_rect_area_cv_pct",
        "aspect_ratio_mean",
        "aspect_ratio_std",
        "aspect_ratio_cv_pct",
        "shape_ratio_mean",
        "shape_ratio_std",
        "shape_ratio_cv_pct",
        "volume_mean_diameter",
        "cluster_count",
        "cluster_mean_size",
        "cluster_std_size",
        "cluster_cv_pct",
    ]
    discovered = set()
    for row in rows:
        discovered.update(row.keys())
    headers = [h for h in preferred if h in discovered]
    for key in sorted(discovered):
        if key not in headers:
            headers.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
