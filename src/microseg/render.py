from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

FIRST_COLOR = (255, 255, 0)
SECOND_COLOR = (0, 170, 255)

FIG_BG = "#f4f7fc"
AX_BG = "#fbfdff"
GRID_COLOR = "#d6e0ee"
AXIS_COLOR = "#aab8cc"
TEXT_COLOR = "#1f344f"


def _style_axes(ax: plt.Axes, title: Optional[str] = None, xlabel: Optional[str] = None, ylabel: Optional[str] = None) -> None:
    ax.set_facecolor(AX_BG)
    if title:
        ax.set_title(title, loc="left", fontsize=9.2, fontweight="semibold", color=TEXT_COLOR)
    if xlabel:
        ax.set_xlabel(xlabel, color="#2f4159")
    if ylabel:
        ax.set_ylabel(ylabel, color="#2f4159")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.9, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS_COLOR)
    ax.spines["bottom"].set_color(AXIS_COLOR)
    ax.tick_params(colors="#334155", labelsize=8.5)


def _fig_to_rgb(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    img = rgba[:, :, :3].copy()
    plt.close(fig)
    return img


def _value_to_interval_index(value: float, edges: Sequence[float]) -> Optional[int]:
    if len(edges) < 2:
        return None
    if not np.isfinite(float(value)):
        return None
    arr = np.asarray(edges, dtype=np.float32)
    if arr.size < 2:
        return None
    if float(value) < float(arr[0]) or float(value) > float(arr[-1]):
        return None
    idx = int(np.searchsorted(arr, float(value), side="right") - 1)
    if idx == int(arr.size - 1) and np.isclose(float(value), float(arr[-1])):
        idx -= 1
    if idx < 0 or idx >= int(arr.size - 1):
        return None
    return idx


def _apply_interval_colors(
    patches: Sequence,
    hist_edges: Sequence[float],
    interval_edges: Sequence[float] | None,
    interval_colors: Sequence[str] | None,
) -> None:
    if interval_edges is None or interval_colors is None:
        return
    if len(interval_edges) < 2 or len(interval_colors) < 1:
        return
    for i, patch in enumerate(patches):
        if i + 1 >= len(hist_edges):
            break
        left = float(hist_edges[i])
        right = float(hist_edges[i + 1])
        center = 0.5 * (left + right)
        bin_idx = _value_to_interval_index(center, interval_edges)
        if bin_idx is None or bin_idx < 0 or bin_idx >= len(interval_colors):
            continue
        patch.set_facecolor(str(interval_colors[bin_idx]))
        patch.set_edgecolor("#ffffff")
        patch.set_alpha(0.9)


def _draw_hist(
    ax: plt.Axes,
    values: Sequence[float],
    bins: int,
    fill_color: str,
    mean_color: str,
    label: str,
    title: str,
    overlay_values: Sequence[float] | None = None,
    overlay_color: str = "#1d4ed8",
    overlay_label: str = "overlay",
    selected_value: Optional[float] = None,
    selected_bin_color: str = "#f59e0b",
    interval_edges: Sequence[float] | None = None,
    interval_colors: Sequence[str] | None = None,
    show_legend: bool = True,
) -> None:
    _, bin_edges, patches = ax.hist(
        values,
        bins=bins,
        color=fill_color,
        edgecolor="#ffffff",
        linewidth=0.9,
        alpha=0.88,
        label=label,
    )
    _apply_interval_colors(patches, bin_edges, interval_edges, interval_colors)
    if len(values) > 0:
        mean_v = float(np.mean(values))
        ax.axvline(mean_v, color=mean_color, linestyle="--", linewidth=1.5, alpha=0.9)
    if overlay_values is not None and len(overlay_values) > 0:
        ax.hist(overlay_values, bins=bins, histtype="step", color=overlay_color, linewidth=1.8, label=overlay_label)
    if selected_value is not None and np.isfinite(float(selected_value)):
        sv = float(selected_value)
        if len(bin_edges) >= 2 and len(patches) > 0:
            idx = int(np.searchsorted(bin_edges, sv, side="right") - 1)
            if idx == len(bin_edges) - 1 and np.isclose(sv, float(bin_edges[-1])):
                idx -= 1
            if 0 <= idx < len(patches):
                patch = patches[idx]
                patch.set_facecolor(selected_bin_color)
                patch.set_edgecolor("#111827")
                patch.set_linewidth(1.3)
                patch.set_alpha(0.95)
    _style_axes(ax, title=title)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    handles, labels = ax.get_legend_handles_labels()
    if show_legend and labels:
        ax.legend(loc="upper right", frameon=False, fontsize=8)


def draw_gradient_line(
    img: np.ndarray,
    pa: Tuple[int, int],
    pb: Tuple[int, int],
    color_start: Tuple[int, int, int],
    color_end: Tuple[int, int, int],
    width: int = 2,
) -> None:
    p0 = np.array([pa[0], pa[1]], dtype=np.float32)
    p1 = np.array([pb[0], pb[1]], dtype=np.float32)
    delta = p1 - p0
    steps = int(max(abs(delta[0]), abs(delta[1]))) + 1
    steps = max(2, min(steps, 512))
    cs = np.array(color_start, dtype=np.float32)
    ce = np.array(color_end, dtype=np.float32)
    for i in range(steps - 1):
        t0 = i / (steps - 1)
        t1 = (i + 1) / (steps - 1)
        p_start = np.round(p0 + delta * t0).astype(int)
        p_end = np.round(p0 + delta * t1).astype(int)
        c = cs + (ce - cs) * t0
        cv2.line(img, tuple(p_start), tuple(p_end), tuple(int(x) for x in c), width, cv2.LINE_AA)
    cv2.circle(img, pa, 3, color_start, -1, cv2.LINE_AA)
    cv2.circle(img, pb, 3, color_end, -1, cv2.LINE_AA)


def render_hist_image_rgb(
    hist_first: Sequence[float],
    hist_second: Sequence[float],
    hist_centroid1: Sequence[float] | None = None,
    hist_centroid2: Sequence[float] | None = None,
    selected_first: Optional[float] = None,
    selected_second: Optional[float] = None,
    width: int = 640,
    height: int = 320,
) -> np.ndarray:
    fig, axes = plt.subplots(1, 2, figsize=(width / 100, height / 100), facecolor=FIG_BG)
    _draw_hist(
        axes[0],
        values=hist_first,
        bins=22,
        fill_color="#56b4e9",
        mean_color="#1d4ed8",
        label="surface",
        title="Nearest1",
        overlay_values=hist_centroid1,
        overlay_color="#2563eb",
        overlay_label="centroid",
        selected_value=selected_first,
        selected_bin_color="#1d4ed8",
    )
    _draw_hist(
        axes[1],
        values=hist_second,
        bins=22,
        fill_color="#f4a261",
        mean_color="#b45309",
        label="surface",
        title="Nearest2",
        overlay_values=hist_centroid2,
        overlay_color="#c2410c",
        overlay_label="centroid",
        selected_value=selected_second,
        selected_bin_color="#c2410c",
    )
    fig.tight_layout(pad=1.3)
    return _fig_to_rgb(fig)


def render_nearest_hist_rgb(
    hist_first: Sequence[float],
    hist_second: Sequence[float],
    hist_centroid1: Sequence[float] | None = None,
    hist_centroid2: Sequence[float] | None = None,
    display_mode: str = "nearest1",
    selected_first: Optional[float] = None,
    selected_second: Optional[float] = None,
    width: int = 640,
    height: int = 320,
) -> np.ndarray:
    mode = str(display_mode or "nearest1").strip().lower()
    use_second = mode in {"nearest2", "n2", "second", "2"}
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), facecolor=FIG_BG)
    if use_second:
        _draw_hist(
            ax,
            values=hist_second,
            bins=22,
            fill_color="#f4a261",
            mean_color="#b45309",
            label="surface",
            title="Nearest2",
            overlay_values=hist_centroid2,
            overlay_color="#c2410c",
            overlay_label="centroid",
            selected_value=selected_second,
            selected_bin_color="#c2410c",
        )
    else:
        _draw_hist(
            ax,
            values=hist_first,
            bins=22,
            fill_color="#56b4e9",
            mean_color="#1d4ed8",
            label="surface",
            title="Nearest1",
            overlay_values=hist_centroid1,
            overlay_color="#2563eb",
            overlay_label="centroid",
            selected_value=selected_first,
            selected_bin_color="#1d4ed8",
        )
    fig.tight_layout(pad=1.2)
    return _fig_to_rgb(fig)


def render_size_area_combo_rgb(hist_size: Sequence[float], hist_area: Sequence[float], width: int = 640, height: int = 220) -> np.ndarray:
    fig, axes = plt.subplots(1, 2, figsize=(width / 100, height / 100), facecolor=FIG_BG)
    _draw_hist(
        axes[0],
        values=hist_size,
        bins=22,
        fill_color="#68d391",
        mean_color="#047857",
        label="ECD",
        title="ECD",
    )
    _draw_hist(
        axes[1],
        values=hist_area,
        bins=22,
        fill_color="#c4b5fd",
        mean_color="#7c3aed",
        label="Area",
        title="Area",
    )
    fig.tight_layout(pad=1.2)
    return _fig_to_rgb(fig)


def _render_single_hist_rgb(
    values: Sequence[float],
    title: str,
    label: str,
    fill_color: str,
    mean_color: str,
    selected_value: Optional[float] = None,
    selected_color: Optional[str] = None,
    interval_edges: Sequence[float] | None = None,
    interval_colors: Sequence[str] | None = None,
    show_legend: bool = True,
    width: int = 320,
    height: int = 220,
) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), facecolor=FIG_BG)
    _draw_hist(
        ax,
        values=values,
        bins=22,
        fill_color=fill_color,
        mean_color=mean_color,
        label=label,
        title=title,
        selected_value=selected_value,
        selected_bin_color=selected_color or mean_color,
        interval_edges=interval_edges,
        interval_colors=interval_colors,
        show_legend=show_legend,
    )
    fig.tight_layout(pad=1.2)
    return _fig_to_rgb(fig)


def render_ecd_hist_rgb(
    hist_size: Sequence[float],
    hist_vesd: Sequence[float] | None = None,
    selected_ecd: Optional[float] = None,
    selected_vesd: Optional[float] = None,
    display_mode: str = "ecd",
    ecd_interval_edges: Sequence[float] | None = None,
    ecd_interval_colors: Sequence[str] | None = None,
    vesd_interval_edges: Sequence[float] | None = None,
    vesd_interval_colors: Sequence[str] | None = None,
    width: int = 320,
    height: int = 220,
) -> np.ndarray:
    mode = str(display_mode or "ecd").strip().lower()
    if mode not in {"ecd", "vesd", "both"}:
        mode = "ecd"
    show_ecd = mode in {"ecd", "both"}
    show_vesd = mode in {"vesd", "both"}
    ecd_vals = [float(v) for v in hist_size]
    vesd_vals = [float(v) for v in (hist_vesd or [])]
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), facecolor=FIG_BG)

    combined: list[float] = []
    if show_ecd:
        combined.extend(ecd_vals)
    if show_vesd:
        combined.extend(vesd_vals)
    if combined:
        try:
            bins = np.histogram_bin_edges(np.asarray(combined, dtype=np.float32), bins=22)
        except Exception:
            bins = 22
    else:
        bins = 22

    if show_ecd:
        _, ecd_edges, ecd_patches = ax.hist(
            ecd_vals,
            bins=bins,
            color="#68d391",
            edgecolor="#ffffff",
            linewidth=0.9,
            alpha=0.72,
            label="ECD",
        )
        _apply_interval_colors(ecd_patches, ecd_edges, ecd_interval_edges, ecd_interval_colors)
        if ecd_vals:
            ecd_mean = float(np.mean(ecd_vals))
            ax.axvline(ecd_mean, color="#047857", linestyle="--", linewidth=1.5, alpha=0.9)
        if selected_ecd is not None and np.isfinite(float(selected_ecd)):
            ecd_sel = float(selected_ecd)
            if len(ecd_edges) >= 2 and len(ecd_patches) > 0:
                idx = int(np.searchsorted(ecd_edges, ecd_sel, side="right") - 1)
                if idx == len(ecd_edges) - 1 and np.isclose(ecd_sel, float(ecd_edges[-1])):
                    idx -= 1
                if 0 <= idx < len(ecd_patches):
                    patch = ecd_patches[idx]
                    patch.set_facecolor("#047857")
                    patch.set_edgecolor("#111827")
                    patch.set_linewidth(1.3)
                    patch.set_alpha(0.95)

    if show_vesd:
        _, vesd_edges, vesd_patches = ax.hist(
            vesd_vals,
            bins=bins,
            color="#93c5fd",
            edgecolor="#1d4ed8",
            linewidth=0.9,
            alpha=0.72 if mode == "vesd" else 0.46,
            label="VESD",
        )
        _apply_interval_colors(vesd_patches, vesd_edges, vesd_interval_edges, vesd_interval_colors)
        if vesd_vals:
            vesd_mean = float(np.mean(vesd_vals))
            ax.axvline(vesd_mean, color="#1d4ed8", linestyle=":", linewidth=1.5, alpha=0.9)
        if selected_vesd is not None and np.isfinite(float(selected_vesd)):
            vesd_sel = float(selected_vesd)
            if len(vesd_edges) >= 2 and len(vesd_patches) > 0:
                idx = int(np.searchsorted(vesd_edges, vesd_sel, side="right") - 1)
                if idx == len(vesd_edges) - 1 and np.isclose(vesd_sel, float(vesd_edges[-1])):
                    idx -= 1
                if 0 <= idx < len(vesd_patches):
                    patch = vesd_patches[idx]
                    patch.set_facecolor("#1d4ed8")
                    patch.set_edgecolor("#111827")
                    patch.set_linewidth(1.3)
                    patch.set_alpha(0.95)

    if mode == "vesd":
        title = "Diameter: VESD"
    elif mode == "ecd":
        title = "Diameter: ECD"
    else:
        title = "Diameter: ECD / VESD"
    _style_axes(ax, title=title)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout(pad=1.2)
    return _fig_to_rgb(fig)


def render_area_hist_rgb(
    hist_area: Sequence[float],
    selected_area: Optional[float] = None,
    title: str = "Area",
    label: str = "Area",
    interval_edges: Sequence[float] | None = None,
    interval_colors: Sequence[str] | None = None,
    width: int = 320,
    height: int = 220,
) -> np.ndarray:
    return _render_single_hist_rgb(
        values=hist_area,
        title=title,
        label=label,
        fill_color="#c4b5fd",
        mean_color="#7c3aed",
        selected_value=selected_area,
        selected_color="#7c3aed",
        interval_edges=interval_edges,
        interval_colors=interval_colors,
        show_legend=False,
        width=width,
        height=height,
    )


def render_aspect_hist_rgb(
    hist_aspect: Sequence[float],
    selected_aspect: Optional[float] = None,
    title: str = "Aspect Ratio",
    interval_edges: Sequence[float] | None = None,
    interval_colors: Sequence[str] | None = None,
    width: int = 320,
    height: int = 220,
) -> np.ndarray:
    return _render_single_hist_rgb(
        values=hist_aspect,
        title=title,
        label="Aspect",
        fill_color="#93c5fd",
        mean_color="#1d4ed8",
        selected_value=selected_aspect,
        selected_color="#1d4ed8",
        interval_edges=interval_edges,
        interval_colors=interval_colors,
        width=width,
        height=height,
    )


def render_fractal_loglog_curve(log_eps: Sequence[float], log_counts: Sequence[float], slope: Optional[float], width: int = 640, height: int = 200) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), facecolor=FIG_BG)
    ax.plot(log_eps, log_counts, "o-", color="#0ea5e9", linewidth=1.8, markersize=4.2, label="mean log-log")
    if slope is not None and len(log_eps) >= 2:
        x = np.array(log_eps, dtype=np.float32)
        y_fit = slope * x + (np.array(log_counts, dtype=np.float32).mean() - slope * x.mean())
        ax.plot(log_eps, y_fit, "--", color="#1d4ed8", linewidth=1.6, label=f"slope={slope:.3f}, D≈{-slope:.3f}")
    _style_axes(ax, title="Fractal Dimension", xlabel="log(eps)", ylabel="log N(eps)")
    ax.legend(loc="best", frameon=False, fontsize=8.2)
    fig.tight_layout(pad=1.1)
    return _fig_to_rgb(fig)


def render_fractal_loglog_multi(series: Sequence[dict], width: int = 640, height: int = 200) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), facecolor=FIG_BG)
    cmap = plt.get_cmap("tab20")
    max_legend = 12
    drawn = 0
    legend_handles = []
    legend_labels = []

    for idx, item in enumerate(series):
        if not isinstance(item, dict):
            continue
        eps_raw = item.get("log_eps")
        cnt_raw = item.get("log_counts")
        if not isinstance(eps_raw, list) or not isinstance(cnt_raw, list) or len(eps_raw) != len(cnt_raw):
            continue
        eps: list[float] = []
        cnt: list[float] = []
        for e, c in zip(eps_raw, cnt_raw):
            try:
                ev = float(e)
                cv = float(c)
            except Exception:
                continue
            if np.isfinite(ev) and np.isfinite(cv):
                eps.append(ev)
                cnt.append(cv)
        if len(eps) < 2:
            continue

        color = cmap(idx % cmap.N)
        line, = ax.plot(eps, cnt, "o-", color=color, linewidth=1.4, markersize=2.8, alpha=0.92)
        drawn += 1
        if drawn <= max_legend:
            name = str(item.get("image_name") or f"image {idx + 1}")
            slope_raw = item.get("slope")
            try:
                slope_v = float(slope_raw)
            except Exception:
                slope_v = None
            if slope_v is not None and np.isfinite(slope_v):
                label = f"{name} (s={slope_v:.3f})"
            else:
                label = name
            legend_handles.append(line)
            legend_labels.append(label)

    _style_axes(ax, title="Fractal Dimension", xlabel="log(eps)", ylabel="log N(eps)")
    if legend_handles:
        legend_cols = 2 if len(legend_handles) > 8 else 1
        ax.legend(legend_handles, legend_labels, loc="best", frameon=False, fontsize=7.8, ncol=legend_cols)
    if drawn > max_legend:
        ax.text(
            0.99,
            0.01,
            f"+{drawn - max_legend} more",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.2,
            color="#64748b",
        )
    fig.tight_layout(pad=1.1)
    return _fig_to_rgb(fig)


def render_fractal_loglog_both(
    curve_eps: Sequence[float],
    curve_counts: Sequence[float],
    curve_slope: Optional[float],
    global_eps: Sequence[float],
    global_counts: Sequence[float],
    global_slope: Optional[float],
    width: int = 640,
    height: int = 200,
) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), facecolor=FIG_BG)
    ax.plot(curve_eps, curve_counts, "o-", color="#06b6d4", linewidth=1.8, markersize=4.0, label="curve")
    if curve_slope is not None and len(curve_eps) >= 2:
        x = np.array(curve_eps, dtype=np.float32)
        y_fit = curve_slope * x + (np.array(curve_counts, dtype=np.float32).mean() - curve_slope * x.mean())
        ax.plot(curve_eps, y_fit, "--", color="#0f3f8c", linewidth=1.5, label=f"curve slope={curve_slope:.3f}, D≈{-curve_slope:.3f}")
    ax.plot(global_eps, global_counts, "o-", color="#f59e0b", linewidth=1.8, markersize=4.0, label="global")
    if global_slope is not None and len(global_eps) >= 2:
        xg = np.array(global_eps, dtype=np.float32)
        yg_fit = global_slope * xg + (np.array(global_counts, dtype=np.float32).mean() - global_slope * xg.mean())
        ax.plot(global_eps, yg_fit, "--", color="#b45309", linewidth=1.5, label=f"global slope={global_slope:.3f}, D≈{-global_slope:.3f}")
    _style_axes(ax, title="Fractal Dimension", xlabel="log(eps)", ylabel="log N(eps)")
    ax.legend(loc="best", frameon=False, fontsize=8.2)
    fig.tight_layout(pad=1.1)
    return _fig_to_rgb(fig)


def render_eval_metrics_panel(
    iou_values: Sequence[float],
    dice_values: Sequence[float],
    bf1_values: Sequence[float],
    width: int = 900,
    height: int = 340,
) -> np.ndarray:
    fig, axes = plt.subplots(2, 2, figsize=(width / 100, height / 100), facecolor=FIG_BG)
    ax_iou, ax_dice = axes[0]
    ax_bf1, ax_box = axes[1]

    _draw_hist(
        ax_iou,
        values=iou_values,
        bins=22,
        fill_color="#7cc6ff",
        mean_color="#1d4ed8",
        label="IoU",
        title="Instance IoU",
    )
    _draw_hist(
        ax_dice,
        values=dice_values,
        bins=22,
        fill_color="#9ee9b8",
        mean_color="#047857",
        label="Dice",
        title="Instance Dice",
    )
    _draw_hist(
        ax_bf1,
        values=bf1_values,
        bins=22,
        fill_color="#f8c98d",
        mean_color="#c2410c",
        label="BF1",
        title="Boundary F1",
    )

    series = []
    labels = []
    for arr, label in ((iou_values, "IoU"), (dice_values, "Dice"), (bf1_values, "BF1")):
        if len(arr) > 0:
            series.append(arr)
            labels.append(label)
    if series:
        bp = ax_box.boxplot(series, labels=labels, patch_artist=True)
        colors = ["#7cc6ff", "#9ee9b8", "#f8c98d"]
        for patch, color in zip(bp["boxes"], colors[: len(bp["boxes"])]):
            patch.set_facecolor(color)
            patch.set_edgecolor("#7b8ca6")
        for med in bp["medians"]:
            med.set_color("#1f344f")
            med.set_linewidth(1.6)
    _style_axes(ax_box, title="Distribution (boxplot)")
    ax_box.grid(axis="y", color=GRID_COLOR, linewidth=0.9, alpha=0.85)
    ax_box.grid(axis="x", visible=False)

    fig.tight_layout(pad=1.0)
    return _fig_to_rgb(fig)


def render_train_metrics_panel(
    train_epochs: Sequence[int],
    train_loss: Sequence[float],
    train_iou: Sequence[float],
    train_dice: Sequence[float],
    val_epochs: Sequence[int],
    val_loss: Sequence[float],
    val_iou: Sequence[float],
    val_dice: Sequence[float],
    test_epochs: Sequence[int],
    test_loss: Sequence[float],
    test_iou: Sequence[float],
    test_dice: Sequence[float],
    width: int = 900,
    height: int = 300,
) -> np.ndarray:
    fig, axes = plt.subplots(1, 3, figsize=(width / 100, height / 100), facecolor=FIG_BG)
    metric_specs = [
        ("Loss", train_loss, val_loss, test_loss),
        ("IoU", train_iou, val_iou, test_iou),
        ("Dice", train_dice, val_dice, test_dice),
    ]
    colors = {
        "train": "#2563eb",
        "val": "#16a34a",
        "test": "#b45309",
    }

    for ax, (title, train_vals, val_vals, test_vals) in zip(axes, metric_specs):
        has_any = False
        if train_epochs and train_vals and len(train_epochs) == len(train_vals):
            ax.plot(train_epochs, train_vals, color=colors["train"], linewidth=1.8, marker="o", markersize=3.2, label="train")
            has_any = True
        if val_epochs and val_vals and len(val_epochs) == len(val_vals):
            ax.plot(val_epochs, val_vals, color=colors["val"], linewidth=1.8, marker="o", markersize=3.2, label="val")
            has_any = True
        if test_epochs and test_vals and len(test_epochs) == len(test_vals):
            ax.scatter(test_epochs, test_vals, color=colors["test"], s=20, marker="D", label="test")
            has_any = True

        _style_axes(ax, title=title, xlabel="Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if title in {"IoU", "Dice"}:
            ax.set_ylim(-0.02, 1.02)
        if has_any:
            ax.legend(loc="best", frameon=False, fontsize=8)
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", color="#64748b", fontsize=9)

    fig.tight_layout(pad=1.0)
    return _fig_to_rgb(fig)


def save_histograms(
    out_dir: Path,
    hist_first: Sequence[float],
    hist_second: Sequence[float],
    hist_centroid1: Sequence[float] | None = None,
    hist_centroid2: Sequence[float] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), facecolor=FIG_BG)
    _draw_hist(
        axes[0],
        values=hist_first,
        bins=22,
        fill_color="#56b4e9",
        mean_color="#1d4ed8",
        label="surface",
        title="Nearest1",
        overlay_values=hist_centroid1,
        overlay_color="#2563eb",
        overlay_label="centroid",
    )
    _draw_hist(
        axes[1],
        values=hist_second,
        bins=22,
        fill_color="#f4a261",
        mean_color="#b45309",
        label="surface",
        title="Nearest2",
        overlay_values=hist_centroid2,
        overlay_color="#c2410c",
        overlay_label="centroid",
    )
    fig.tight_layout(pad=1.3)
    fig.savefig(str(out_dir / "histogram_nearest.png"), dpi=200)
    plt.close(fig)


def save_size_histogram(out_dir: Path, hist_size: Sequence[float]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3), facecolor=FIG_BG)
    _draw_hist(ax, values=hist_size, bins=22, fill_color="#68d391", mean_color="#047857", label="ECD", title="ECD")
    fig.tight_layout(pad=1.2)
    fig.savefig(str(out_dir / "histogram_ecd.png"), dpi=200)
    plt.close(fig)


def save_area_histogram(out_dir: Path, hist_area: Sequence[float]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3), facecolor=FIG_BG)
    _draw_hist(ax, values=hist_area, bins=22, fill_color="#c4b5fd", mean_color="#7c3aed", label="Area", title="Area")
    fig.tight_layout(pad=1.2)
    fig.savefig(str(out_dir / "histogram_area.png"), dpi=200)
    plt.close(fig)


def save_fractal_loglog(out_dir: Path, log_eps: Sequence[float], log_counts: Sequence[float], slope: Optional[float], name: str = "fractal_loglog") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3), facecolor=FIG_BG)
    ax.plot(log_eps, log_counts, "o-", color="#0ea5e9", linewidth=1.8, markersize=4.2, label="mean log-log")
    if slope is not None and len(log_eps) >= 2:
        x = np.array(log_eps, dtype=np.float32)
        y_fit = slope * x + (np.array(log_counts, dtype=np.float32).mean() - slope * x.mean())
        ax.plot(log_eps, y_fit, "--", color="#1d4ed8", linewidth=1.6, label=f"slope={slope:.3f}, D≈{-slope:.3f}")
    _style_axes(ax, title="Fractal Dimension", xlabel="log(eps)", ylabel="log N(eps)")
    ax.legend(loc="best", frameon=False, fontsize=8.2)
    fig.tight_layout(pad=1.1)
    fig.savefig(str(out_dir / f"{name}.png"), dpi=200)
    plt.close(fig)


def save_size_area_combo(out_dir: Path, hist_size: Sequence[float], hist_area: Sequence[float]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), facecolor=FIG_BG)
    _draw_hist(axes[0], values=hist_size, bins=22, fill_color="#68d391", mean_color="#047857", label="ECD", title="ECD")
    _draw_hist(axes[1], values=hist_area, bins=22, fill_color="#c4b5fd", mean_color="#7c3aed", label="Area", title="Area")
    fig.tight_layout(pad=1.2)
    fig.savefig(str(out_dir / "histogram_size_area.png"), dpi=200)
    plt.close(fig)
