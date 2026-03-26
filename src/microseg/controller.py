from __future__ import annotations

import csv
import json
from collections import Counter
import colorsys
import copy
from datetime import datetime
from pathlib import Path
import re
import shlex
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PySide6 import QtCore, QtGui, QtWidgets
from scipy.ndimage import binary_fill_holes

from microseg.hf_sam_predictor import HFSamPredictor, load_hf_sam_model
from microseg.io_utils import ensure_outdir, load_image_bgr, load_mask_dir
from microseg.config import DEFAULT_SCALE, MAGNIFICATION_PRESETS, SCALE_FACTOR_NM
from microseg.compute import (
    MaskEntry,
    clean_mask,
    compute_mask_shape_metrics,
    compute_two_nearest,
    mask_ellipse_fit_params_px,
    mask_centroid,
    mask_feret_major_minor_segments_px,
    summarize,
    summarize_sizes,
)
from microseg.render import (
    FIRST_COLOR,
    SECOND_COLOR,
    draw_gradient_line,
    render_area_hist_rgb,
    render_aspect_hist_rgb,
    render_ecd_hist_rgb,
    render_eval_metrics_panel,
    render_fractal_loglog_curve,
    render_fractal_loglog_multi,
    render_nearest_hist_rgb,
    render_train_metrics_panel,
    save_fractal_loglog,
    save_histograms,
    save_size_area_combo,
)
from microseg.csv_export import (
    write_csv_summary,
    write_hist_nearest_csv,
    write_hist_size_csv,
    write_fractal_global_csv,
    write_boxcount_global_csv,
    write_mask_csv,
    write_all_images_index_csv,
)
from microseg.eval_gt import (
    ID_MAP_FILE_EXTS,
    load_eval_gt_instances,
    resolve_eval_gt_source,
)
from microseg.eval_metrics import run_eval_scope
from microseg.session import ImageSessionState
from PySide6.QtWidgets import QFileDialog, QMessageBox
from microseg.view import MaskDistanceView
from microseg.eval_ops import boundary_from_instances
from microseg.qt_image import bgr_to_qpixmap, fit_rgb_to_cell, pixmap_to_rgb, rgb_to_qpixmap
from microseg.train_monitor import TrainMonitorState


class MaskDistanceController(QtCore.QObject):
    WORKSPACE_ANALYZE = 0
    WORKSPACE_FILTERS = 1
    WORKSPACE_TRAIN = 2
    WORKSPACE_TRACK = 3

    def __init__(
        self,
        image_paths: List[Path],
        output_dir: Path,
        hf_model_id: str,
        lora_checkpoint: Optional[Path],
        lora_rank: int,
        lora_alpha: float,
        lora_dropout: float,
        mask_threshold: float,
        scale_nm_per_px: float = DEFAULT_SCALE,
        max_distance_nm: Optional[float] = None,
        fractal_slides: int = 0,
        init_mask_id_path: Optional[Path] = None,
        init_mask_dir: Optional[Path] = None,
    ):
        super().__init__()
        if init_mask_id_path is not None and init_mask_dir is not None:
            raise ValueError("Specify only one of init_mask_id_path and init_mask_dir")
        if not image_paths:
            raise ValueError("At least one image path is required")
        self.image_paths = [Path(p) for p in image_paths]
        if len(self.image_paths) > 1 and (init_mask_id_path is not None or init_mask_dir is not None):
            raise ValueError("--init-mask-id/--init-mask-dir is supported only for single-image mode")

        first_path = self.image_paths[0]
        first_image = load_image_bgr(first_path)
        self.base_output_dir = output_dir
        self.output_dir = output_dir
        self.base_image_path = first_path
        self.image_bgr = first_image
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hf_model_id = str(hf_model_id)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.mask_threshold = float(mask_threshold)
        self.lora_checkpoint_path = Path(lora_checkpoint).expanduser() if lora_checkpoint is not None else None
        self.predictor = self._build_predictor(self.lora_checkpoint_path)
        self.lora_mode_available = self.lora_checkpoint_path is not None

        self.scale_nm_per_px = scale_nm_per_px
        self.max_distance_nm = max_distance_nm
        self.fractal_slides_setting = max(0, fractal_slides)
        self.overlay_use_centroid = False
        self.cluster_threshold_nm: float = 1.0
        self.cluster_labels: List[int] = []
        self.cluster_stats: Dict = {}
        self.realtime_calc_enabled = False
        self.include_zero_distance = False
        self.show_nearest_links = True
        self.show_set_masks = True
        self.show_current_mask = True
        self.show_prompts = True
        self.show_gt_overlay = False
        self.show_bbox_overlay = False
        self.show_axes_overlay = False
        self.show_feret_parallelogram_overlay = False
        self.show_ellipse_overlay = False
        self.cluster_k = 3
        self.nearest_hist_metric = "nearest1"  # nearest1 | nearest2
        self.size_hist_metric = "ecd"  # ecd | vesd
        self.area_hist_metric = "area"  # area | bbox | vesd
        self.aspect_hist_metric = "feret"  # feret | ellipse
        self.main_graph_metric = "ecd"  # nearest1 | nearest2 | ecd | vesd | area | bbox | area_vesd | aspect_feret | aspect_ellipse | fractal
        self.distribution_metric = "none"  # none | vesd | ecd | aspect
        self.distribution_edges: List[float] = []
        self._distribution_slider_min_internal: Optional[float] = None
        self._distribution_slider_max_internal: Optional[float] = None
        self.current_is_raw_display = False
        self.use_original_mode = False
        self.mode = "sam"  # sam | lora | polygon
        self.polygon_mode = False
        self.scale_calibration_mode = False
        self.scale_bar_points: List[Tuple[float, float]] = []
        self.scale_bar_px_length: Optional[float] = None
        self._scale_px_before_measure: str = ""
        self.display_length_unit = "nm"
        self._scale_preset_labels: List[str] = [str(name) for name in MAGNIFICATION_PRESETS]
        self._scale_preset_magnifications: List[Optional[float]] = [
            self._parse_magnification_label_to_value(name) for name in MAGNIFICATION_PRESETS
        ]
        self._scale_preset_factor_nm = float(SCALE_FACTOR_NM) if np.isfinite(float(SCALE_FACTOR_NM)) and float(SCALE_FACTOR_NM) > 0 else 18800.0
        self._scale_slider_steps = 1000
        self._scale_slider_snap_norm = 0.02
        self._scale_nonpreset_mag_step = 10000.0
        self._scale_mag_min = 0.0
        self._scale_mag_max = 0.0
        self._scale_mag_log_span = 0.0
        self._scale_preset_norm_positions = self._build_scale_preset_norm_positions()
        self.analysis_scope = "current"  # current | all
        self.zoom_min = 1.0
        self.zoom_max = 16.0
        self.workspace_tab = self.WORKSPACE_ANALYZE
        self.filter_input_source = "filtered"  # filtered | original
        self.filter_brightness: int = 0
        self.filter_contrast: float = 1.0
        self.filter_gamma: float = 1.0
        self.spatial_filter_chain: List[Dict[str, Any]] = self._default_filter_chain()
        self.frequency_filter_chain: List[Dict[str, Any]] = []
        self.filter_chain: List[Dict[str, Any]] = []
        self.filtered_image_bgr: np.ndarray = self.image_bgr.copy()
        self.spatial_filter_selected_row: int = -1
        self.frequency_filter_selected_row: int = -1
        self.filter_selected_domain: str = "spatial"
        self.filter_fft_mode: bool = False
        self._sym_notch_drag_active: bool = False
        self._sym_notch_drag_sign: int = 1
        self._sym_notch_drag_step_idx: int = -1

        # Dirty/save state and recovery.
        self.unsaved_images: set[int] = set()
        self._last_save_note = self._now_text()
        self.recovery_dir = self.base_output_dir / ".recovery"
        self.recovery_manifest = self.recovery_dir / "manifest.json"
        self.autosave_interval_ms = 45000

        self._disp_size = (0, 0)
        self._disp_origin = (0, 0)
        self._scale_x = 1.0
        self._scale_y = 1.0
        h, w = first_image.shape[:2]
        self._view_x0 = 0.0
        self._view_y0 = 0.0
        self._view_w = float(w)
        self._view_h = float(h)
        self._view_center_x = float(w) / 2.0
        self._view_center_y = float(h) / 2.0
        self.zoom_factor = 1.0

        self.set_masks: List[MaskEntry] = []
        self.current: Optional[MaskEntry] = None
        self.current_is_raw_display = False
        self.pair_results: List[Dict] = []
        self.summary: Dict = {}
        self.size_summary: Dict = {}
        self.summary_stale = False
        self.selected_idx: Optional[int] = None
        self.selected_indices: set[int] = set()
        self.prompt_points: List[Tuple[float, float, int]] = []
        self.prompt_box: Optional[Tuple[float, float, float, float]] = None
        self.drag_prompt_box: Optional[Tuple[float, float, float, float]] = None
        self.polygon_points: List[Tuple[float, float]] = []
        self.cluster_labels = []
        self.cluster_stats = {}
        self._undo_stack: List[Dict[str, Any]] = []
        self._undo_limit = 30
        self._last_undoable_action = ""

        # Cache keys for faster redraw.
        self._overlay_revision = 0
        self._analysis_revision = 0
        self._overlay_cache_key: Optional[Tuple[int, int, int]] = None
        self._overlay_cache: Optional[np.ndarray] = None
        self._graphs_cache_key: Optional[Tuple[Any, ...]] = None
        self._graphs_cache_pixmap: Optional[QtGui.QPixmap] = None
        self._graph_panels_cache_key: Optional[Tuple[Any, ...]] = None
        self._graph_panels_cache: Optional[Dict[str, Optional[QtGui.QPixmap]]] = None
        self._stats_cache_revision = -1
        self._review_cache_key: Optional[Tuple[int, int, int]] = None
        self._review_cache_rows: List[List[str]] = []
        self._review_cache_summary: str = ""
        self._review_row_to_mask_idx: List[int] = []
        self.review_sort_key = "index"
        self.review_sort_desc = False

        self._project_root = Path(__file__).resolve().parents[2]
        self._train_process: Optional[QtCore.QProcess] = None
        self._train_monitor = TrainMonitorState(expected_epochs=0)
        self._eval_gt_min_area = 30
        self._eval_pred_min_area = 50
        self._eval_match_iou = 0.0
        self._eval_boundary_ratio = 0.005
        self._eval_gt_source_cache: Dict[Tuple[int, str, bool], Optional[Path]] = {}
        self._eval_gt_instances_cache: Dict[Tuple[int, str, bool], Optional[List[np.ndarray]]] = {}
        self._eval_current_result: Dict[str, Any] = {}
        self._eval_all_result: Dict[str, Any] = {}
        self._resize_refresh_pending = False
        self._train_preview_key: Optional[Tuple[str, str]] = None
        self._train_preview_overlay: Optional[np.ndarray] = None
        self._train_preview_note: str = "Train preview: set image/mask dirs."
        self._track_preview_key: Optional[str] = None
        self._track_preview_image: Optional[np.ndarray] = None
        self._track_preview_note: str = "Track preview: import image."

        ts = QtCore.QDateTime.currentDateTime().toString("yyyyMMddHHmmss")
        self.image_sessions: List[ImageSessionState] = []
        for idx, path in enumerate(self.image_paths):
            img = first_image if idx == 0 else load_image_bgr(path)
            ih, iw = img.shape[:2]
            out_name = (
                f"{path.stem}_{ts}"
                if len(self.image_paths) == 1
                else f"{idx + 1:04d}_{path.stem}_{ts}"
            )
            initial_masks = self._load_initial_masks(init_mask_id_path, init_mask_dir) if idx == 0 else []
            self.image_sessions.append(
                ImageSessionState(
                    image_path=path,
                    image_bgr=img,
                    output_dir=self.base_output_dir / out_name,
                    scale_nm_per_px=float(scale_nm_per_px),
                    set_masks=initial_masks,
                    view_center_x=float(iw) / 2.0,
                    view_center_y=float(ih) / 2.0,
                    zoom_factor=1.0,
                    summary_stale=bool(initial_masks),
                    dirty=bool(initial_masks),
                    filter_brightness=0,
                    filter_contrast=1.0,
                    filter_gamma=1.0,
                    spatial_filter_chain=copy.deepcopy(self.spatial_filter_chain),
                    frequency_filter_chain=copy.deepcopy(self.frequency_filter_chain),
                    filter_chain=copy.deepcopy(self._combined_filter_chain()),
                    filtered_image_bgr=None,
                    spatial_filter_selected_row=-1,
                    frequency_filter_selected_row=-1,
                    filter_selected_domain="spatial",
                    filter_fft_mode=False,
                )
            )
            if initial_masks:
                self.unsaved_images.add(idx)
        for state in self.image_sessions:
            self._recompute_filtered_image_for_state(state)
        self.current_image_idx = 0
        first_state = self.image_sessions[0]
        self.spatial_filter_chain = copy.deepcopy(first_state.spatial_filter_chain)
        self.frequency_filter_chain = copy.deepcopy(first_state.frequency_filter_chain)
        self.filter_chain = copy.deepcopy(first_state.filter_chain)
        self.filtered_image_bgr = (
            self.image_sessions[0].filtered_image_bgr
            if isinstance(self.image_sessions[0].filtered_image_bgr, np.ndarray)
            else self.image_sessions[0].image_bgr
        )
        self.spatial_filter_selected_row = int(getattr(first_state, "spatial_filter_selected_row", -1))
        self.frequency_filter_selected_row = int(getattr(first_state, "frequency_filter_selected_row", -1))
        self.filter_selected_domain = str(getattr(first_state, "filter_selected_domain", "spatial") or "spatial")
        self.filter_brightness = int(self.image_sessions[0].filter_brightness)
        self.filter_contrast = float(self.image_sessions[0].filter_contrast)
        self.filter_gamma = float(self.image_sessions[0].filter_gamma)
        self.filter_fft_mode = bool(self.image_sessions[0].filter_fft_mode)

        self.view = MaskDistanceView(self)
        self.view.set_scale_text(f"{self.scale_nm_per_px:.3f}")
        self._sync_scale_preset_ui_from_scale()
        self.view.set_display_unit(self.display_length_unit)
        self.view.set_scale_calibration_mode(self.scale_calibration_mode)
        self.view.set_scale_calibration_pixels(self.scale_bar_px_length)
        self.view.set_cluster_text(f"{self.cluster_threshold_nm:.0f}")
        self.view.set_fractal_checked(self.fractal_slides_setting >= 20)
        self.view.set_realtime_calc_checked(self.realtime_calc_enabled)
        self.view.set_overlay_centroid_checked(self.overlay_use_centroid)
        self.view.set_include_zero_checked(self.include_zero_distance)
        self.view.set_show_nearest_checked(self.show_nearest_links)
        self.view.set_show_set_checked(self.show_set_masks)
        self.view.set_show_current_checked(self.show_current_mask)
        self.view.set_show_prompts_checked(self.show_prompts)
        self.view.set_show_gt_checked(self.show_gt_overlay)
        self.view.set_show_bbox_checked(self.show_bbox_overlay)
        self.view.set_show_axes_checked(self.show_axes_overlay)
        self.view.set_show_feret_parallelogram_checked(self.show_feret_parallelogram_overlay)
        self.view.set_show_ellipse_checked(self.show_ellipse_overlay)
        self.view.set_nearest_hist_metric(self.nearest_hist_metric)
        self.view.set_size_hist_metric(self.size_hist_metric)
        self.view.set_area_hist_metric(self.area_hist_metric)
        self.view.set_aspect_hist_metric(self.aspect_hist_metric)
        self.view.set_main_graph_metric(self.main_graph_metric)
        self.view.set_distribution_metric(self.distribution_metric)
        self.view.set_distribution_bins_count(3)
        self.view.set_distribution_bins_enabled(False)
        self.view.set_distribution_edges_text("")
        self.view.set_distribution_status("")
        self.view.set_lora_mode_enabled(self.lora_mode_available)
        self.view.set_lora_checkpoint_text(str(self.lora_checkpoint_path) if self.lora_checkpoint_path is not None else "")
        self.view.set_lora_runtime_status(
            self.lora_checkpoint_path.name if self.lora_checkpoint_path is not None else "",
            True if self.lora_checkpoint_path is not None else None,
        )
        self.view.set_filter_input_source(self.filter_input_source)
        self.view.set_filter_adjustments(self.filter_brightness, self.filter_contrast, self.filter_gamma)
        self.view.set_filter_fft_mode(self.filter_fft_mode)
        self._refresh_filter_ui()
        self.view.set_mode(self.mode)
        self.view.set_action_mode(self.mode)
        self.view.set_save_state("Unsaved" if self.unsaved_images else "Saved", bool(self.unsaved_images))
        if not self.view.train_output_dir_edit.text().strip():
            self.view.train_output_dir_edit.setText(str((self.base_output_dir / "train_runs").resolve()))
        self.view.set_train_running(False)
        self._reset_train_monitor_ui(clear_graph=True)
        self.view.set_eval_running(False)
        self.view.set_eval_scope("current")
        self.view.set_eval_scope_plot("current", None, "Run Compare to view current-image metrics")
        self.view.set_eval_scope_plot("all", None, "Run Compare to view all-images metrics")
        self.view.set_eval_status_text("Idle")
        self._sync_eval_default_paths()
        self._autosave_timer = QtCore.QTimer(self.view)
        self._autosave_timer.setInterval(self.autosave_interval_ms)
        self._autosave_timer.timeout.connect(self._on_autosave_timer)
        self._autosave_timer.start()
        self._try_restore_recovery()
        self._apply_image_session(self.current_image_idx)
        self._sync_workspace_ui()
        self._refresh_predictor_image()
        self._auto_calc_if_ready()
        self._update_image_status_ui()

    def _build_predictor(self, lora_checkpoint: Optional[Path]) -> HFSamPredictor:
        sam = load_hf_sam_model(
            model_id=self.hf_model_id,
            device=self._device,
            lora_checkpoint=lora_checkpoint,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout,
        )
        predictor = HFSamPredictor(sam, device=self._device, threshold=self.mask_threshold)
        predictor.set_image(self.image_bgr)
        return predictor

    @staticmethod
    def _normalize_filter_kind(kind: Any) -> str:
        token = str(kind or "").strip().lower()
        if token in {"gaussian", "gauss", "blur"}:
            return "gaussian"
        if token in {"median"}:
            return "median"
        if token in {"clahe"}:
            return "clahe"
        if token in {"unsharp", "sharpen"}:
            return "unsharp"
        if token in {"lowpass", "low-pass", "lp", "low"}:
            return "lowpass"
        if token in {"highpass", "high-pass", "hp", "high"}:
            return "highpass"
        if token in {"bandpass", "band-pass", "bp", "band"}:
            return "bandpass"
        if token in {"sym_notch", "sym-notch", "symmetric_notch", "notch", "paired_notch"}:
            return "sym_notch"
        return "gaussian"

    def _default_filter_chain(self) -> List[Dict[str, Any]]:
        return []

    def _normalize_filter_step(self, step: Any) -> Dict[str, Any]:
        if not isinstance(step, dict):
            step = {}
        kind = self._normalize_filter_kind(step.get("kind"))
        params_raw = step.get("params")
        params = dict(params_raw) if isinstance(params_raw, dict) else {}
        if kind == "gaussian":
            sigma = float(params.get("sigma", 1.2) or 1.2)
            sigma = min(max(sigma, 0.0), 10.0)
            params_out = {"sigma": sigma}
        elif kind == "median":
            ksize = int(params.get("ksize", 3) or 3)
            ksize = min(max(ksize, 1), 31)
            if ksize % 2 == 0:
                ksize += 1 if ksize < 31 else -1
            params_out = {"ksize": int(ksize)}
        elif kind == "clahe":
            clip = float(params.get("clip", 2.0) or 2.0)
            clip = min(max(clip, 0.1), 10.0)
            grid = int(params.get("grid", 8) or 8)
            grid = min(max(grid, 2), 32)
            params_out = {"clip": clip, "grid": int(grid)}
        elif kind == "unsharp":
            amount = float(params.get("amount", 1.0) or 1.0)
            amount = min(max(amount, 0.0), 5.0)
            sigma = float(params.get("sigma", 1.0) or 1.0)
            sigma = min(max(sigma, 0.1), 10.0)
            params_out = {"amount": amount, "sigma": sigma}
        elif kind == "lowpass":
            cutoff = float(params.get("cutoff", 0.2) or 0.2)
            cutoff = min(max(cutoff, 0.01), 1.0)
            params_out = {"cutoff": cutoff}
        elif kind == "highpass":
            cutoff = float(params.get("cutoff", 0.1) or 0.1)
            cutoff = min(max(cutoff, 0.01), 1.0)
            params_out = {"cutoff": cutoff}
        elif kind == "bandpass":
            inner = float(params.get("inner", 0.08) or 0.08)
            outer = float(params.get("outer", 0.28) or 0.28)
            inner = min(max(inner, 0.0), 0.98)
            outer = min(max(outer, inner + 0.01), 1.0)
            params_out = {"inner": inner, "outer": outer}
        else:  # sym_notch
            radius = float(params.get("radius", 0.35) or 0.35)
            radius = min(max(radius, 0.01), 1.0)
            width = float(params.get("width", 0.06) or 0.06)
            width = min(max(width, 0.005), 0.5)
            angle_deg = float(params.get("angle_deg", 0.0) or 0.0)
            angle_deg = ((angle_deg + 180.0) % 360.0) - 180.0
            params_out = {"radius": radius, "width": width, "angle_deg": angle_deg}
        return {"kind": kind, "params": params_out}

    def _normalize_filter_chain(self, chain: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if isinstance(chain, list):
            for raw in chain:
                if isinstance(raw, dict) and ("enabled" in raw) and (not bool(raw.get("enabled", True))):
                    continue
                out.append(self._normalize_filter_step(raw))
        return out

    def _normalize_spatial_filter_chain(self, chain: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for step in self._normalize_filter_chain(chain):
            if self._normalize_filter_kind(step.get("kind")) in {"lowpass", "highpass", "bandpass", "sym_notch"}:
                continue
            out.append(step)
        return out

    def _normalize_frequency_filter_chain(self, chain: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for step in self._normalize_filter_chain(chain):
            if self._normalize_filter_kind(step.get("kind")) in {"lowpass", "highpass", "bandpass", "sym_notch"}:
                out.append(step)
        return out

    def _split_filter_chain_by_domain(self, chain: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        normalized = self._normalize_filter_chain(chain)
        return self._normalize_spatial_filter_chain(normalized), self._normalize_frequency_filter_chain(normalized)

    def _combined_filter_chain(self) -> List[Dict[str, Any]]:
        return [*self._normalize_spatial_filter_chain(self.spatial_filter_chain), *self._normalize_frequency_filter_chain(self.frequency_filter_chain)]

    @staticmethod
    def _fmt_filter_num(v: float) -> str:
        if abs(float(v) - round(float(v))) < 1e-6:
            return str(int(round(float(v))))
        return f"{float(v):.1f}"

    def _normalize_filter_adjustments(self, payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            payload = {}
        try:
            brightness = int(round(float(payload.get("brightness", 0) or 0)))
        except Exception:
            brightness = 0
        try:
            contrast = float(payload.get("contrast", 1.0) or 1.0)
        except Exception:
            contrast = 1.0
        try:
            gamma = float(payload.get("gamma", 1.0) or 1.0)
        except Exception:
            gamma = 1.0
        brightness = int(np.clip(brightness, -100, 100))
        contrast = float(np.clip(contrast, 0.1, 3.0))
        gamma = float(np.clip(gamma, 0.2, 3.0))
        return {
            "brightness": int(brightness),
            "contrast": float(contrast),
            "gamma": float(gamma),
        }

    @staticmethod
    def _brightness_ui_to_internal(ui_value: Any) -> int:
        try:
            ui = float(ui_value)
        except Exception:
            ui = 50.0
        ui = float(np.clip(ui, 0.0, 100.0))
        return int(round(ui * 2.0 - 100.0))

    @staticmethod
    def _contrast_ui_to_internal(ui_value: Any) -> float:
        try:
            ui = float(ui_value)
        except Exception:
            ui = 50.0
        ui = float(np.clip(ui, 0.0, 100.0))
        if ui >= 50.0:
            val = 1.0 + ((ui - 50.0) * (2.0 / 50.0))
        else:
            val = 0.1 + (ui * (0.9 / 50.0))
        return float(np.clip(val, 0.1, 3.0))

    @staticmethod
    def _gamma_ui_to_internal(ui_value: Any) -> float:
        try:
            ui = float(ui_value)
        except Exception:
            ui = 50.0
        ui = float(np.clip(ui, 0.0, 100.0))
        if ui >= 50.0:
            val = 1.0 + ((ui - 50.0) * (2.0 / 50.0))
        else:
            val = 0.2 + (ui * (0.8 / 50.0))
        return float(np.clip(val, 0.2, 3.0))

    def _current_filter_adjustments(self) -> Dict[str, Any]:
        return self._normalize_filter_adjustments(
            {
                "brightness": self.filter_brightness,
                "contrast": self.filter_contrast,
                "gamma": self.filter_gamma,
            }
        )

    def _filter_adjust_slug(self, adjustments: Dict[str, Any]) -> str:
        adj = self._normalize_filter_adjustments(adjustments)
        b = int(adj["brightness"])
        c = self._fmt_filter_num(float(adj["contrast"])).replace(".", "p")
        g = self._fmt_filter_num(float(adj["gamma"])).replace(".", "p")
        return f"br{b:+d}_ct{c}_gm{g}".replace("+", "p").replace("-", "m")

    def _apply_image_adjustments(self, image_bgr: np.ndarray, adjustments: Dict[str, Any]) -> np.ndarray:
        adj = self._normalize_filter_adjustments(adjustments)
        brightness = float(adj["brightness"])
        contrast = float(adj["contrast"])
        gamma = float(adj["gamma"])

        out = image_bgr.astype(np.float32)
        out = (out - 127.5) * contrast + 127.5 + brightness
        out = np.clip(out, 0.0, 255.0)

        if abs(gamma - 1.0) > 1e-6:
            inv_gamma = 1.0 / max(gamma, 1e-6)
            out = np.power(out / 255.0, inv_gamma) * 255.0
            out = np.clip(out, 0.0, 255.0)
        return out.astype(np.uint8)

    def _filter_step_text(self, step: Dict[str, Any], index: int) -> str:
        kind = self._normalize_filter_kind(step.get("kind"))
        params = step.get("params", {}) if isinstance(step.get("params"), dict) else {}
        if kind == "gaussian":
            detail = f"sigma={self._fmt_filter_num(float(params.get('sigma', 1.2)))}"
            title = "Gaussian"
        elif kind == "median":
            detail = f"ksize={int(params.get('ksize', 3))}"
            title = "Median"
        elif kind == "clahe":
            detail = (
                f"clip={self._fmt_filter_num(float(params.get('clip', 2.0)))}, "
                f"grid={int(params.get('grid', 8))}"
            )
            title = "CLAHE"
        elif kind == "unsharp":
            detail = (
                f"amount={self._fmt_filter_num(float(params.get('amount', 1.0)))}, "
                f"sigma={self._fmt_filter_num(float(params.get('sigma', 1.0)))}"
            )
            title = "Unsharp"
        elif kind == "lowpass":
            detail = f"cutoff={self._fmt_filter_num(float(params.get('cutoff', 0.2)) * 100.0)}%"
            title = "Low-pass"
        elif kind == "highpass":
            detail = f"cutoff={self._fmt_filter_num(float(params.get('cutoff', 0.1)) * 100.0)}%"
            title = "High-pass"
        elif kind == "bandpass":
            detail = (
                f"inner={self._fmt_filter_num(float(params.get('inner', 0.08)) * 100.0)}%, "
                f"outer={self._fmt_filter_num(float(params.get('outer', 0.28)) * 100.0)}%"
            )
            title = "Band-pass"
        else:
            detail = (
                f"radius={self._fmt_filter_num(float(params.get('radius', 0.35)) * 100.0)}%, "
                f"width={self._fmt_filter_num(float(params.get('width', 0.06)) * 100.0)}%, "
                f"angle={self._fmt_filter_num(float(params.get('angle_deg', 0.0)))}deg"
            )
            title = "Sym Notch"
        return f"{index + 1}. {title} ({detail})"

    def _filter_chain_slug(self, chain: Sequence[Dict[str, Any]]) -> str:
        tokens: List[str] = []
        for step in chain:
            kind = self._normalize_filter_kind(step.get("kind"))
            params = step.get("params", {}) if isinstance(step.get("params"), dict) else {}
            if kind == "gaussian":
                tokens.append(f"g{self._fmt_filter_num(float(params.get('sigma', 1.2))).replace('.', 'p')}")
            elif kind == "median":
                tokens.append(f"m{int(params.get('ksize', 3))}")
            elif kind == "clahe":
                clip = self._fmt_filter_num(float(params.get("clip", 2.0))).replace(".", "p")
                grid = int(params.get("grid", 8))
                tokens.append(f"c{clip}g{grid}")
            elif kind == "unsharp":
                amount = self._fmt_filter_num(float(params.get("amount", 1.0))).replace(".", "p")
                sigma = self._fmt_filter_num(float(params.get("sigma", 1.0))).replace(".", "p")
                tokens.append(f"u{amount}s{sigma}")
            elif kind == "lowpass":
                c = self._fmt_filter_num(float(params.get("cutoff", 0.2)) * 100.0).replace(".", "p")
                tokens.append(f"lp{c}")
            elif kind == "highpass":
                c = self._fmt_filter_num(float(params.get("cutoff", 0.1)) * 100.0).replace(".", "p")
                tokens.append(f"hp{c}")
            elif kind == "bandpass":
                i = self._fmt_filter_num(float(params.get("inner", 0.08)) * 100.0).replace(".", "p")
                o = self._fmt_filter_num(float(params.get("outer", 0.28)) * 100.0).replace(".", "p")
                tokens.append(f"bp{i}-{o}")
            else:
                r = self._fmt_filter_num(float(params.get("radius", 0.35)) * 100.0).replace(".", "p")
                w = self._fmt_filter_num(float(params.get("width", 0.06)) * 100.0).replace(".", "p")
                a = self._fmt_filter_num(float(params.get("angle_deg", 0.0))).replace(".", "p")
                a = a.replace("-", "m")
                tokens.append(f"sn{r}w{w}a{a}")
        return "_".join(tokens) if tokens else "nofilter"

    @staticmethod
    def _sym_notch_geometry(shape: Tuple[int, int], step: Dict[str, Any]) -> Dict[str, float]:
        h, w = int(shape[0]), int(shape[1])
        cy = (h - 1) / 2.0
        cx = (w - 1) / 2.0
        max_r = max(1.0, min(float(h), float(w)) * 0.5)
        params = step.get("params", {}) if isinstance(step.get("params"), dict) else {}
        radius_norm = float(params.get("radius", 0.35))
        width_norm = float(params.get("width", 0.06))
        angle_deg = float(params.get("angle_deg", 0.0))
        radius_px = min(max(radius_norm, 0.01), 1.0) * max_r
        width_px = min(max(width_norm, 0.005), 0.5) * max_r
        rad = np.deg2rad(angle_deg)
        dx = radius_px * float(np.cos(rad))
        dy = radius_px * float(np.sin(rad))
        c1x = cx + dx
        c1y = cy + dy
        c2x = cx - dx
        c2y = cy - dy
        return {
            "cx": float(cx),
            "cy": float(cy),
            "max_r": float(max_r),
            "radius_norm": float(min(max(radius_norm, 0.01), 1.0)),
            "width_norm": float(min(max(width_norm, 0.005), 0.5)),
            "angle_deg": float(((angle_deg + 180.0) % 360.0) - 180.0),
            "radius_px": float(radius_px),
            "width_px": float(width_px),
            "c1x": float(c1x),
            "c1y": float(c1y),
            "c2x": float(c2x),
            "c2y": float(c2y),
        }

    def _get_active_sym_notch_step(self) -> Optional[Tuple[int, Dict[str, Any]]]:
        chain = self._normalize_frequency_filter_chain(self.frequency_filter_chain)
        if not chain:
            return None
        selected_idx = int(self.frequency_filter_selected_row)
        if 0 <= selected_idx < len(chain):
            step = self._normalize_filter_step(chain[selected_idx])
            if self._normalize_filter_kind(step.get("kind")) == "sym_notch":
                return selected_idx, step
        for idx, raw_step in enumerate(chain):
            step = self._normalize_filter_step(raw_step)
            if self._normalize_filter_kind(step.get("kind")) == "sym_notch":
                return idx, step
        return None

    def _clear_sym_notch_drag_state(self) -> None:
        self._sym_notch_drag_active = False
        self._sym_notch_drag_sign = 1
        self._sym_notch_drag_step_idx = -1

    def _sync_active_sym_notch_from_image_point(self, xx: float, yy: float) -> bool:
        if self.workspace_tab != self.WORKSPACE_FILTERS or not self.filter_fft_mode:
            return False
        if not self.frequency_filter_chain:
            return False
        step_idx = int(self._sym_notch_drag_step_idx)
        if step_idx < 0 or step_idx >= len(self.frequency_filter_chain):
            return False
        step = self._normalize_filter_step(self.frequency_filter_chain[step_idx])
        if self._normalize_filter_kind(step.get("kind")) != "sym_notch":
            return False

        geom = self._sym_notch_geometry(self.image_bgr.shape[:2], step)
        cx = float(geom["cx"])
        cy = float(geom["cy"])
        vx = float(xx) - cx
        vy = float(yy) - cy
        if int(self._sym_notch_drag_sign) < 0:
            vx = -vx
            vy = -vy
        radius_norm = float(np.hypot(vx, vy) / max(1e-6, float(geom["max_r"])))
        radius_norm = min(max(radius_norm, 0.01), 1.0)
        angle_deg = float(np.degrees(np.arctan2(vy, vx)))
        angle_deg = ((angle_deg + 180.0) % 360.0) - 180.0

        params = dict(step.get("params") or {})
        prev_radius = float(params.get("radius", 0.35))
        prev_angle = float(params.get("angle_deg", 0.0))
        if abs(prev_radius - radius_norm) < 1e-6 and abs(prev_angle - angle_deg) < 1e-6:
            return False
        params["radius"] = radius_norm
        params["angle_deg"] = angle_deg
        step["params"] = params
        self.frequency_filter_chain[step_idx] = self._normalize_filter_step(step)
        self.frequency_filter_selected_row = int(step_idx)
        self.filter_selected_domain = "frequency"
        self._refresh_filter_ui()
        self._on_filter_chain_changed(refresh_preview=True)
        return True

    def _handle_sym_notch_dragging_event(self, x0: int, y0: int, x1: int, y1: int, finalize: bool = False) -> bool:
        if self.workspace_tab != self.WORKSPACE_FILTERS or not self.filter_fft_mode:
            if self._sym_notch_drag_active:
                self._clear_sym_notch_drag_state()
            return False
        if not self._sym_notch_drag_active:
            active = self._get_active_sym_notch_step()
            if active is None:
                return False
            step_idx, step = active
            p0 = self._display_to_image_xy(x0, y0)
            if p0 is None:
                return False
            geom = self._sym_notch_geometry(self.image_bgr.shape[:2], step)
            px, py = float(p0[0]), float(p0[1])
            d1 = float(np.hypot(px - geom["c1x"], py - geom["c1y"]))
            d2 = float(np.hypot(px - geom["c2x"], py - geom["c2y"]))
            hit_radius = max(10.0, float(geom["width_px"]) * 1.4)
            if d1 > hit_radius and d2 > hit_radius:
                return False
            self._sym_notch_drag_active = True
            self._sym_notch_drag_step_idx = int(step_idx)
            self._sym_notch_drag_sign = 1 if d1 <= d2 else -1

        p1 = self._display_to_image_xy(x1, y1)
        if p1 is not None:
            self._sync_active_sym_notch_from_image_point(float(p1[0]), float(p1[1]))
        if finalize:
            self._clear_sym_notch_drag_state()
        return True

    def _draw_active_sym_notch_overlay(self, overlay: np.ndarray) -> None:
        active = self._get_active_sym_notch_step()
        if active is None:
            return
        _idx, step = active
        geom = self._sym_notch_geometry(overlay.shape[:2], step)
        cx = int(round(float(geom["cx"])))
        cy = int(round(float(geom["cy"])))
        p1 = (int(round(float(geom["c1x"]))), int(round(float(geom["c1y"]))))
        p2 = (int(round(float(geom["c2x"]))), int(round(float(geom["c2y"]))))
        notch_radius = max(3, int(round(float(geom["width_px"]))))
        cv2.line(overlay, p1, p2, (120, 180, 255), 1, cv2.LINE_AA)
        cv2.circle(overlay, (cx, cy), 3, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(overlay, p1, notch_radius, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.circle(overlay, p2, notch_radius, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.circle(overlay, p1, 4, (0, 220, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay, p2, 4, (0, 220, 255), -1, cv2.LINE_AA)

    def _frequency_mask_for_step(self, shape: Tuple[int, int], step: Dict[str, Any]) -> np.ndarray:
        h, w = int(shape[0]), int(shape[1])
        cy = (h - 1) / 2.0
        cx = (w - 1) / 2.0
        yy, xx = np.indices((h, w), dtype=np.float32)
        rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        max_r = max(1.0, min(float(h), float(w)) * 0.5)
        kind = str(step.get("kind", "lowpass"))
        params = step.get("params", {}) if isinstance(step.get("params"), dict) else {}
        mask = np.ones((h, w), dtype=np.float32)
        if kind == "lowpass":
            cutoff = float(params.get("cutoff", 0.2))
            cutoff = min(max(cutoff, 0.01), 1.0) * max_r
            mask = (rr <= cutoff).astype(np.float32)
        elif kind == "highpass":
            cutoff = float(params.get("cutoff", 0.1))
            cutoff = min(max(cutoff, 0.01), 1.0) * max_r
            mask = (rr >= cutoff).astype(np.float32)
        elif kind == "bandpass":
            inner = float(params.get("inner", 0.08))
            outer = float(params.get("outer", 0.28))
            inner = min(max(inner, 0.0), 0.98)
            outer = min(max(outer, inner + 0.01), 1.0)
            rin = inner * max_r
            rout = outer * max_r
            mask = np.logical_and(rr >= rin, rr <= rout).astype(np.float32)
        else:  # sym_notch
            geom = self._sym_notch_geometry(shape, step)
            width = float(geom["width_px"])
            c1x = float(geom["c1x"])
            c1y = float(geom["c1y"])
            c2x = float(geom["c2x"])
            c2y = float(geom["c2y"])
            notch1 = ((xx - c1x) ** 2 + (yy - c1y) ** 2) <= (width ** 2)
            notch2 = ((xx - c2x) ** 2 + (yy - c2y) ** 2) <= (width ** 2)
            mask[np.logical_or(notch1, notch2)] = 0.0
            # Keep DC component stable unless explicitly masked by user params.
            if abs(float(geom["radius_px"])) > 1e-6:
                mask[int(round(cy)), int(round(cx))] = 1.0
        return mask

    def _apply_frequency_filter_step(self, image_bgr: np.ndarray, step: Dict[str, Any]) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        spec = np.fft.fftshift(np.fft.fft2(gray))
        mask = self._frequency_mask_for_step(gray.shape, step)
        spec_filtered = spec * mask
        recon = np.fft.ifft2(np.fft.ifftshift(spec_filtered))
        out_gray = np.real(recon)
        out_gray = np.clip(out_gray, 0.0, 255.0).astype(np.uint8)
        return cv2.cvtColor(out_gray, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _fft_visualize_bgr(image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        spec = np.fft.fftshift(np.fft.fft2(gray))
        mag = np.log1p(np.abs(spec))
        mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(mag_u8, cv2.COLOR_GRAY2BGR)

    def _apply_filter_chain(self, image_bgr: np.ndarray, chain: Sequence[Dict[str, Any]]) -> np.ndarray:
        out = image_bgr.copy()
        for raw_step in chain:
            step = self._normalize_filter_step(raw_step)
            kind = step["kind"]
            params = step["params"]
            if kind == "gaussian":
                sigma = float(params.get("sigma", 1.2))
                if sigma <= 0.0:
                    continue
                k = int(max(3, round(sigma * 6.0)))
                if k % 2 == 0:
                    k += 1
                out = cv2.GaussianBlur(out, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT101)
            elif kind == "median":
                ksize = int(params.get("ksize", 3))
                if ksize < 3:
                    continue
                if ksize % 2 == 0:
                    ksize += 1
                out = cv2.medianBlur(out, ksize)
            elif kind == "clahe":
                clip = float(params.get("clip", 2.0))
                grid = int(params.get("grid", 8))
                grid = max(2, grid)
                lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=max(0.1, clip), tileGridSize=(grid, grid))
                l2 = clahe.apply(l)
                out = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
            elif kind == "unsharp":
                amount = float(params.get("amount", 1.0))
                sigma = float(params.get("sigma", 1.0))
                if amount <= 0.0:
                    continue
                blur = cv2.GaussianBlur(
                    out,
                    ksize=(0, 0),
                    sigmaX=max(0.1, sigma),
                    sigmaY=max(0.1, sigma),
                    borderType=cv2.BORDER_REFLECT101,
                )
                out = cv2.addWeighted(out, 1.0 + amount, blur, -amount, 0)
            else:
                out = self._apply_frequency_filter_step(out, step)
        return out

    def _recompute_filtered_image_for_state(self, state: ImageSessionState) -> None:
        norm_adj = self._normalize_filter_adjustments(
            {
                "brightness": state.filter_brightness,
                "contrast": state.filter_contrast,
                "gamma": state.filter_gamma,
            }
        )
        state.filter_brightness = int(norm_adj["brightness"])
        state.filter_contrast = float(norm_adj["contrast"])
        state.filter_gamma = float(norm_adj["gamma"])
        spatial_chain = self._normalize_spatial_filter_chain(getattr(state, "spatial_filter_chain", []))
        frequency_chain = self._normalize_frequency_filter_chain(getattr(state, "frequency_filter_chain", []))
        if not spatial_chain and not frequency_chain and getattr(state, "filter_chain", None):
            spatial_chain, frequency_chain = self._split_filter_chain_by_domain(state.filter_chain)
        state.spatial_filter_chain = copy.deepcopy(spatial_chain)
        state.frequency_filter_chain = copy.deepcopy(frequency_chain)
        state.filter_chain = [*spatial_chain, *frequency_chain]
        adjusted = self._apply_image_adjustments(state.image_bgr, norm_adj)
        out = self._apply_filter_chain(adjusted, spatial_chain)
        out = self._apply_filter_chain(out, frequency_chain)
        state.filtered_image_bgr = out

    def _recompute_filtered_for_current(self) -> None:
        state = self.image_sessions[self.current_image_idx]
        norm_adj = self._normalize_filter_adjustments(
            {
                "brightness": self.filter_brightness,
                "contrast": self.filter_contrast,
                "gamma": self.filter_gamma,
            }
        )
        self.filter_brightness = int(norm_adj["brightness"])
        self.filter_contrast = float(norm_adj["contrast"])
        self.filter_gamma = float(norm_adj["gamma"])
        state.filter_brightness = int(norm_adj["brightness"])
        state.filter_contrast = float(norm_adj["contrast"])
        state.filter_gamma = float(norm_adj["gamma"])
        self.spatial_filter_chain = self._normalize_spatial_filter_chain(self.spatial_filter_chain)
        self.frequency_filter_chain = self._normalize_frequency_filter_chain(self.frequency_filter_chain)
        state.spatial_filter_chain = copy.deepcopy(self.spatial_filter_chain)
        state.frequency_filter_chain = copy.deepcopy(self.frequency_filter_chain)
        self.filter_chain = self._combined_filter_chain()
        state.filter_chain = copy.deepcopy(self.filter_chain)
        adjusted = self._apply_image_adjustments(state.image_bgr, norm_adj)
        out = self._apply_filter_chain(adjusted, self.spatial_filter_chain)
        out = self._apply_filter_chain(out, self.frequency_filter_chain)
        state.filtered_image_bgr = out
        self.filtered_image_bgr = state.filtered_image_bgr

    def _active_predictor_image(self) -> np.ndarray:
        if self.filter_input_source == "filtered" and isinstance(self.filtered_image_bgr, np.ndarray):
            return self.filtered_image_bgr
        return self.image_bgr

    def _refresh_predictor_image(self) -> None:
        self.predictor.set_image(self._active_predictor_image())

    def _is_segmentation_workspace_active(self) -> bool:
        return self.workspace_tab in {self.WORKSPACE_ANALYZE, self.WORKSPACE_FILTERS}

    def _load_initial_masks(
        self,
        init_mask_id_path: Optional[Path],
        init_mask_dir: Optional[Path],
    ) -> List[MaskEntry]:
        if init_mask_id_path is None and init_mask_dir is None:
            return []
        if init_mask_id_path is not None:
            raw_masks = self._load_masks_from_id_map(init_mask_id_path)
        else:
            if init_mask_dir is None:
                return []
            raw_masks = load_mask_dir(init_mask_dir, self.image_bgr.shape[:2])
        scores = self._load_initial_scores(init_mask_id_path, init_mask_dir, len(raw_masks))
        entries: List[MaskEntry] = []
        for idx, mask in enumerate(raw_masks):
            mask_bin = (mask > 0).astype(np.uint8)
            score = scores[idx] if idx < len(scores) else None
            entries.append(MaskEntry(mask=mask_bin, raw=mask_bin.copy(), score=score))
        return entries

    def _load_initial_scores(
        self,
        init_mask_id_path: Optional[Path],
        init_mask_dir: Optional[Path],
        expected_count: int,
        extra_dirs: Optional[List[Path]] = None,
    ) -> List[Optional[float]]:
        if init_mask_id_path is None and init_mask_dir is None and not extra_dirs:
            return [None] * max(0, int(expected_count))

        def _to_score(value: Any) -> Optional[float]:
            if value is None:
                return None
            try:
                v = float(value)
            except Exception:
                return None
            if not np.isfinite(v):
                return None
            return v

        def _parse_scores_payload(payload: Any) -> List[Optional[float]]:
            if isinstance(payload, list):
                out: List[Optional[float]] = []
                for v in payload:
                    out.append(_to_score(v))
                return out
            if not isinstance(payload, dict):
                return []
            scores_raw = payload.get("scores")
            ids_raw = payload.get("instance_ids")
            if isinstance(scores_raw, list):
                if isinstance(ids_raw, list) and len(ids_raw) == len(scores_raw):
                    tmp: List[Tuple[int, Optional[float]]] = []
                    for idx, (inst, score) in enumerate(zip(ids_raw, scores_raw), start=1):
                        try:
                            inst_id = int(inst)
                        except Exception:
                            inst_id = idx
                        score_val = _to_score(score)
                        tmp.append((inst_id, score_val))
                    tmp.sort(key=lambda x: x[0])
                    return [s for _, s in tmp]
                return _parse_scores_payload(scores_raw)
            masks = payload.get("masks")
            if isinstance(masks, list):
                tmp: List[Tuple[int, Optional[float]]] = []
                for idx, item in enumerate(masks):
                    if not isinstance(item, dict):
                        continue
                    inst_id = item.get("instance_id", idx + 1)
                    score = item.get("score", None)
                    try:
                        tmp.append((int(inst_id), _to_score(score)))
                    except Exception:
                        continue
                if tmp:
                    tmp.sort(key=lambda x: x[0])
                    return [s for _, s in tmp]
            return []

        def _parse_scores_csv(path: Path) -> List[Optional[float]]:
            try:
                with path.open("r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    if not reader.fieldnames:
                        return []
                    fields = {name.strip().lower(): name for name in reader.fieldnames if name}
                    score_key = fields.get("score")
                    if score_key is None:
                        return []
                    id_key = fields.get("instance_id") or fields.get("id") or fields.get("index")
                    rows: List[Tuple[int, Optional[float]]] = []
                    for row_idx, row in enumerate(reader, start=1):
                        raw_score = row.get(score_key)
                        score_val = _to_score(raw_score)
                        if id_key is None:
                            inst_id = row_idx
                        else:
                            try:
                                inst_id = int(float(str(row.get(id_key, row_idx)).strip()))
                            except Exception:
                                inst_id = row_idx
                        rows.append((inst_id, score_val))
            except Exception:
                return []
            if not rows:
                return []
            rows.sort(key=lambda x: x[0])
            return [s for _, s in rows]

        def _extend_dir_candidates(paths: List[Path], base_dir: Path) -> None:
            paths.extend(
                [
                    base_dir / "instance_scores.json",
                    base_dir / "results.json",
                    base_dir / "scores.json",
                    base_dir / "masks.csv",
                ]
            )

        candidates: List[Path] = []
        if init_mask_id_path is not None:
            base = init_mask_id_path.parent
            _extend_dir_candidates(candidates, base)
        if init_mask_dir is not None:
            d = init_mask_dir
            _extend_dir_candidates(candidates, d)
            _extend_dir_candidates(candidates, d.parent)
        for extra in extra_dirs or []:
            _extend_dir_candidates(candidates, Path(extra))
        seen: set[str] = set()
        fallback_scores: Optional[List[Optional[float]]] = None
        for path in candidates:
            key = str(path.resolve()) if path.exists() else str(path)
            if key in seen:
                continue
            seen.add(key)
            if not path.exists() or not path.is_file():
                continue
            if path.suffix.lower() == ".csv":
                scores = _parse_scores_csv(path)
            else:
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                scores = _parse_scores_payload(data)
            if scores:
                if all(s is None for s in scores):
                    if fallback_scores is None:
                        fallback_scores = list(scores)
                    continue
                if expected_count <= 0:
                    return scores
                if len(scores) < expected_count:
                    scores = scores + [None] * (expected_count - len(scores))
                return scores[:expected_count]
        if fallback_scores:
            if expected_count <= 0:
                return fallback_scores
            if len(fallback_scores) < expected_count:
                fallback_scores = fallback_scores + [None] * (expected_count - len(fallback_scores))
            return fallback_scores[:expected_count]
        return [None] * max(0, int(expected_count))

    def _load_masks_from_id_map(self, id_path: Path) -> List[np.ndarray]:
        id_map = cv2.imread(str(id_path), cv2.IMREAD_UNCHANGED)
        if id_map is None:
            raise FileNotFoundError(f"Could not read instance-id image: {id_path}")
        if id_map.ndim == 3:
            if id_map.shape[2] == 4:
                id_map = cv2.cvtColor(id_map, cv2.COLOR_BGRA2GRAY)
            else:
                id_map = cv2.cvtColor(id_map, cv2.COLOR_BGR2GRAY)
        target_h, target_w = self.image_bgr.shape[:2]
        if id_map.shape != (target_h, target_w):
            id_map = cv2.resize(id_map, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        instance_ids = [int(v) for v in np.unique(id_map) if int(v) > 0]
        masks: List[np.ndarray] = []
        for inst_id in sorted(instance_ids):
            mask = (id_map == inst_id).astype(np.uint8)
            if mask.any():
                masks.append(mask)
        if not masks:
            raise ValueError(f"No instance IDs (>0) found in {id_path}")
        return masks

    def _load_masks_from_id_map_for_shape(self, id_path: Path, shape_hw: Tuple[int, int]) -> List[np.ndarray]:
        id_map = cv2.imread(str(id_path), cv2.IMREAD_UNCHANGED)
        if id_map is None:
            return []
        if id_map.ndim == 3:
            if id_map.shape[2] == 4:
                id_map = cv2.cvtColor(id_map, cv2.COLOR_BGRA2GRAY)
            else:
                id_map = cv2.cvtColor(id_map, cv2.COLOR_BGR2GRAY)
        target_h, target_w = int(shape_hw[0]), int(shape_hw[1])
        if id_map.shape != (target_h, target_w):
            id_map = cv2.resize(id_map, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        masks: List[np.ndarray] = []
        for inst_id in sorted(int(v) for v in np.unique(id_map) if int(v) > 0):
            m = (id_map == inst_id).astype(np.uint8)
            if m.any():
                masks.append(m)
        return masks

    def _now_text(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    @staticmethod
    def _score_value(score: Any) -> Optional[float]:
        if score is None:
            return None
        try:
            v = float(score)
        except Exception:
            return None
        if not np.isfinite(v):
            return None
        return v

    @staticmethod
    def _normalize_prompt_payload(payload: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(payload, dict):
            return None
        try:
            normalized = json.loads(json.dumps(payload))
        except Exception:
            return None
        if not isinstance(normalized, dict):
            return None
        return normalized

    def _score_text(self, score: Any, digits: int = 3, na_text: str = "N/A") -> str:
        v = self._score_value(score)
        if v is None:
            return na_text
        return f"{v:.{digits}f}"

    def _build_point_box_prompt_payload(self) -> Dict[str, Any]:
        positive_count = 0
        negative_count = 0
        points: List[Dict[str, Any]] = []
        for px, py, lbl in self.prompt_points:
            label = int(lbl)
            if label > 0:
                positive_count += 1
            else:
                negative_count += 1
            points.append(
                {
                    "x": float(px),
                    "y": float(py),
                    "label": label,
                    "label_name": "positive" if label > 0 else "negative",
                }
            )
        box_xyxy = None
        if self.prompt_box is not None:
            box_xyxy = [float(v) for v in self.prompt_box]
        return {
            "mode": self.mode,
            "tool": "points_box",
            "points": points,
            "box_xyxy": box_xyxy,
            "positive_count": positive_count,
            "negative_count": negative_count,
        }

    def _build_polygon_prompt_payload(self) -> Dict[str, Any]:
        vertices = [[float(px), float(py)] for px, py in self.polygon_points]
        return {
            "mode": "polygon",
            "tool": "polygon",
            "vertices": vertices,
            "vertex_count": len(vertices),
        }

    def _update_save_state_ui(self) -> None:
        dirty_count = len(self.unsaved_images)
        if dirty_count == 0:
            self.view.set_save_state("Saved", False)
            return
        self.view.set_save_state(f"Unsaved ({dirty_count})", True)

    def _mark_dirty_current(self) -> None:
        self.unsaved_images.add(int(self.current_image_idx))
        self.image_sessions[self.current_image_idx].dirty = True
        self._update_save_state_ui()

    def _mark_dirty_all(self) -> None:
        for idx in range(len(self.image_sessions)):
            self.unsaved_images.add(idx)
            self.image_sessions[idx].dirty = True
        self._update_save_state_ui()

    def _mark_saved_current(self) -> None:
        idx = int(self.current_image_idx)
        self.unsaved_images.discard(idx)
        self.image_sessions[idx].dirty = False
        self._last_save_note = self._now_text()
        self._update_save_state_ui()

    def _mark_saved_all(self) -> None:
        self.unsaved_images.clear()
        for s in self.image_sessions:
            s.dirty = False
        self._last_save_note = self._now_text()
        self._update_save_state_ui()

    def _capture_scale_from_ui_if_valid(self) -> None:
        if not hasattr(self, "view"):
            return
        try:
            px_raw = self.view.get_scale_px_text().strip().replace(",", ".")
            length_raw = self.view.get_scale_bar_length_text().strip().replace(",", ".")
        except Exception:
            return
        if not px_raw or not length_raw:
            return
        try:
            px_len = float(px_raw)
            known_length = float(length_raw)
        except Exception:
            return
        if not np.isfinite(px_len) or px_len <= 0.0:
            return
        if not np.isfinite(known_length) or known_length <= 0.0:
            return
        try:
            known_nm = self._length_to_nm(known_length, self.view.get_display_unit())
        except Exception:
            known_nm = None
        if known_nm is None or not np.isfinite(float(known_nm)) or float(known_nm) <= 0.0:
            return
        restored_scale = max(float(known_nm) / float(px_len), 0.0001)
        self.scale_nm_per_px = float(restored_scale)
        if 0 <= int(self.current_image_idx) < len(self.image_sessions):
            self.image_sessions[int(self.current_image_idx)].scale_nm_per_px = float(restored_scale)
        self.view.set_scale_text(f"{self.scale_nm_per_px:.3f}")

    def _mark_overlay_dirty(self) -> None:
        self._overlay_revision += 1
        self._overlay_cache_key = None
        self._overlay_cache = None
        self._review_cache_key = None

    def _mark_analysis_dirty(self) -> None:
        self._analysis_revision += 1
        self._graphs_cache_key = None
        self._graphs_cache_pixmap = None
        self._graph_panels_cache_key = None
        self._graph_panels_cache = None
        self._stats_cache_revision = -1
        self._mark_overlay_dirty()

    def _clear_analysis_state(self) -> None:
        self.pair_results = []
        self.summary = {}
        self.size_summary = {}
        self.cluster_labels = []
        self.cluster_stats = {}
        self._mark_analysis_dirty()

    def _session_to_id_map(self, state: ImageSessionState) -> np.ndarray:
        h, w = state.image_bgr.shape[:2]
        id_map = np.zeros((h, w), dtype=np.uint16)
        for idx, rec in enumerate(state.set_masks):
            inst_id = idx + 1
            id_map[rec.mask.astype(bool)] = inst_id
        return id_map

    def _write_recovery_snapshot(self) -> None:
        self._capture_scale_from_ui_if_valid()
        self._persist_current_session_state()
        self.recovery_dir.mkdir(parents=True, exist_ok=True)
        images_payload: List[Dict] = []
        for idx, state in enumerate(self.image_sessions):
            id_name = f"image_{idx + 1:04d}_instance_ids.tiff"
            id_path = self.recovery_dir / id_name
            id_map = self._session_to_id_map(state)
            cv2.imwrite(str(id_path), id_map)
            images_payload.append(
                {
                    "index": idx,
                    "image_path": str(state.image_path.resolve()),
                    "output_dir": str(state.output_dir),
                    "id_map_path": id_name,
                    "scores": [self._score_value(rec.score) for rec in state.set_masks],
                    "prompts": [self._normalize_prompt_payload(rec.prompt_data) for rec in state.set_masks],
                    "scale_nm_per_px": float(state.scale_nm_per_px),
                    "dirty": bool(state.dirty),
                    "spatial_filter_chain": copy.deepcopy(getattr(state, "spatial_filter_chain", [])),
                    "frequency_filter_chain": copy.deepcopy(getattr(state, "frequency_filter_chain", [])),
                    "filter_chain": copy.deepcopy(state.filter_chain),
                    "filter_adjustments": {
                        "brightness": int(state.filter_brightness),
                        "contrast": float(state.filter_contrast),
                        "gamma": float(state.filter_gamma),
                    },
                    "filter_fft_mode": bool(state.filter_fft_mode),
                }
            )
        payload = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "current_index": int(self.current_image_idx),
            "scale_nm_per_px": float(self.scale_nm_per_px),
            "max_distance_nm": self.max_distance_nm,
            "realtime_calc_enabled": bool(self.realtime_calc_enabled),
            "include_zero_distance": bool(self.include_zero_distance),
            "show_bbox_overlay": bool(self.show_bbox_overlay),
            "show_axes_overlay": bool(self.show_axes_overlay),
            "show_feret_parallelogram_overlay": bool(self.show_feret_parallelogram_overlay),
            "show_ellipse_overlay": bool(self.show_ellipse_overlay),
            "cluster_threshold_nm": float(self.cluster_threshold_nm),
            "fractal_slides_setting": int(self.fractal_slides_setting),
            "nearest_hist_metric": str(self.nearest_hist_metric),
            "size_hist_metric": str(self.size_hist_metric),
            "area_hist_metric": str(self.area_hist_metric),
            "aspect_hist_metric": str(self.aspect_hist_metric),
            "main_graph_metric": str(self.main_graph_metric),
            "distribution_metric": str(self.distribution_metric),
            "distribution_edges": [float(v) for v in self.distribution_edges],
            "filter_input_source": str(self.filter_input_source),
            "images": images_payload,
        }
        self.recovery_manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _try_restore_recovery(self) -> None:
        if not self.recovery_manifest.exists():
            return
        try:
            payload = json.loads(self.recovery_manifest.read_text(encoding="utf-8"))
        except Exception:
            return
        images_raw = payload.get("images") or []
        if not isinstance(images_raw, list) or not images_raw:
            return
        images_by_path: Dict[str, Dict[str, Any]] = {}
        for item in images_raw:
            if not isinstance(item, dict):
                continue
            path_raw = str(item.get("image_path", "")).strip()
            if not path_raw:
                continue
            images_by_path[path_raw] = item
        if not images_by_path:
            return
        matched = 0
        for p in self.image_paths:
            try:
                key = str(Path(p).resolve())
            except Exception:
                key = str(p)
            if key in images_by_path:
                matched += 1
        if matched == 0:
            return
        answer = QMessageBox.question(
            self.view,
            "Recovery Found",
            "Autosaved recovery data was found. Restore masks and state?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer != QMessageBox.Yes:
            return
        legacy_scale = float(payload.get("scale_nm_per_px", self.scale_nm_per_px))
        restored_dirty: set[int] = set()
        for idx, state in enumerate(self.image_sessions):
            try:
                state_key = str(state.image_path.resolve())
            except Exception:
                state_key = str(state.image_path)
            item = images_by_path.get(state_key)
            if not isinstance(item, dict):
                continue
            id_rel = item.get("id_map_path")
            if not id_rel:
                continue
            id_path = self.recovery_dir / str(id_rel)
            masks = self._load_masks_from_id_map_for_shape(id_path, state.image_bgr.shape[:2])
            scores_raw = item.get("scores")
            scores: List[Optional[float]] = []
            if isinstance(scores_raw, list):
                for v in scores_raw:
                    scores.append(self._score_value(v))
            prompts_raw = item.get("prompts")
            prompts: List[Optional[Dict[str, Any]]] = []
            if isinstance(prompts_raw, list):
                for p in prompts_raw:
                    prompts.append(self._normalize_prompt_payload(p))
            if (not scores or all(s is None for s in scores)) and masks:
                output_dir_raw = item.get("output_dir")
                extra_dirs: List[Path] = []
                if output_dir_raw:
                    out_dir = Path(str(output_dir_raw))
                    extra_dirs.extend([out_dir, out_dir / "instance_masks"])
                scores = self._load_initial_scores(
                    init_mask_id_path=None,
                    init_mask_dir=None,
                    expected_count=len(masks),
                    extra_dirs=extra_dirs,
                )
            state.set_masks = []
            for mi, m in enumerate(masks):
                s = scores[mi] if mi < len(scores) else None
                prompt_data = prompts[mi] if mi < len(prompts) else None
                state.set_masks.append(MaskEntry(mask=m, raw=m.copy(), score=s, prompt_data=prompt_data))
            state.current = None
            state.current_is_raw_display = False
            state.pair_results = []
            state.summary = {}
            state.size_summary = {}
            state.summary_stale = bool(state.set_masks)
            state.selected_idx = None
            state.selected_indices = set()
            state.prompt_points = []
            state.prompt_box = None
            state.polygon_points = []
            state.cluster_labels = []
            state.cluster_stats = {}
            state.undo_stack = []
            state.dirty = bool(state.set_masks) or bool(item.get("dirty", False))
            if state.dirty:
                restored_dirty.add(idx)
            out_dir = item.get("output_dir")
            if out_dir:
                state.output_dir = Path(out_dir)
            try:
                restored_scale = float(item.get("scale_nm_per_px", legacy_scale))
            except Exception:
                restored_scale = legacy_scale
            state.scale_nm_per_px = max(restored_scale, 0.0001)
            restored_adjust = self._normalize_filter_adjustments(item.get("filter_adjustments"))
            state.filter_brightness = int(restored_adjust["brightness"])
            state.filter_contrast = float(restored_adjust["contrast"])
            state.filter_gamma = float(restored_adjust["gamma"])
            restored_spatial = self._normalize_spatial_filter_chain(item.get("spatial_filter_chain"))
            restored_frequency = self._normalize_frequency_filter_chain(item.get("frequency_filter_chain"))
            if not restored_spatial and not restored_frequency:
                restored_spatial, restored_frequency = self._split_filter_chain_by_domain(item.get("filter_chain"))
            state.spatial_filter_chain = restored_spatial
            state.frequency_filter_chain = restored_frequency
            state.filter_chain = [*restored_spatial, *restored_frequency]
            state.spatial_filter_selected_row = (
                int(np.clip(getattr(state, "spatial_filter_selected_row", -1), -1, max(-1, len(restored_spatial) - 1)))
            )
            state.frequency_filter_selected_row = (
                int(np.clip(getattr(state, "frequency_filter_selected_row", -1), -1, max(-1, len(restored_frequency) - 1)))
            )
            domain = str(getattr(state, "filter_selected_domain", "spatial") or "spatial").strip().lower()
            state.filter_selected_domain = "frequency" if domain == "frequency" else "spatial"
            state.filter_fft_mode = bool(item.get("filter_fft_mode", False))
            self._recompute_filtered_image_for_state(state)
        self.unsaved_images = restored_dirty
        self.current_image_idx = int(np.clip(int(payload.get("current_index", 0)), 0, len(self.image_sessions) - 1))
        self.scale_nm_per_px = float(self.image_sessions[self.current_image_idx].scale_nm_per_px)
        max_dist = payload.get("max_distance_nm", self.max_distance_nm)
        self.max_distance_nm = None if max_dist is None else float(max_dist)
        self.realtime_calc_enabled = bool(payload.get("realtime_calc_enabled", self.realtime_calc_enabled))
        source = str(payload.get("filter_input_source", self.filter_input_source)).strip().lower()
        self.filter_input_source = "original" if source == "original" else "filtered"
        # Analysis toggles always prefer default-off policy, even when recovery data exists.
        self.include_zero_distance = False
        self.show_bbox_overlay = bool(payload.get("show_bbox_overlay", self.show_bbox_overlay))
        self.show_axes_overlay = bool(payload.get("show_axes_overlay", self.show_axes_overlay))
        self.show_feret_parallelogram_overlay = bool(
            payload.get("show_feret_parallelogram_overlay", self.show_feret_parallelogram_overlay)
        )
        feret_enabled = bool(self.show_axes_overlay or self.show_feret_parallelogram_overlay)
        self.show_axes_overlay = feret_enabled
        self.show_feret_parallelogram_overlay = feret_enabled
        self.show_ellipse_overlay = bool(payload.get("show_ellipse_overlay", self.show_ellipse_overlay))
        self.cluster_threshold_nm = float(payload.get("cluster_threshold_nm", self.cluster_threshold_nm))
        self.fractal_slides_setting = 0
        self.nearest_hist_metric = self._normalize_nearest_hist_metric(payload.get("nearest_hist_metric", self.nearest_hist_metric))
        self.size_hist_metric = self._normalize_size_hist_metric(payload.get("size_hist_metric", self.size_hist_metric))
        self.area_hist_metric = self._normalize_area_hist_metric(payload.get("area_hist_metric", self.area_hist_metric))
        self.aspect_hist_metric = self._normalize_aspect_hist_metric(payload.get("aspect_hist_metric", self.aspect_hist_metric))
        self.main_graph_metric = self._normalize_main_graph_metric(payload.get("main_graph_metric", self.main_graph_metric))
        dist_metric_raw = payload.get("distribution_metric", self.distribution_metric)
        dist_metric = self._normalize_distribution_metric(dist_metric_raw)
        dist_edges_raw = payload.get("distribution_edges")
        dist_edges: List[float] = []
        if isinstance(dist_edges_raw, list):
            for v in dist_edges_raw:
                try:
                    fv = float(v)
                except Exception:
                    continue
                if np.isfinite(fv):
                    dist_edges.append(fv)
        dist_edges = sorted({float(v) for v in dist_edges})
        if dist_metric == "none" or len(dist_edges) < 2:
            self.distribution_metric = "none"
            self.distribution_edges = []
        else:
            self.distribution_metric = dist_metric
            self.distribution_edges = dist_edges
        self._last_save_note = f"recovered {self._now_text()}"
        self.view.set_scale_text(f"{self.scale_nm_per_px:.3f}")
        self.view.set_cluster_text(f"{self.cluster_threshold_nm:.0f}")
        with QtCore.QSignalBlocker(self.view.realtime_calc_checkbox):
            self.view.set_realtime_calc_checked(self.realtime_calc_enabled)
        self.view.set_include_zero_checked(False)
        self.view.set_show_bbox_checked(self.show_bbox_overlay)
        self.view.set_show_feret_checked(self.show_axes_overlay)
        self.view.set_show_ellipse_checked(self.show_ellipse_overlay)
        self.view.set_fractal_checked(False)
        with QtCore.QSignalBlocker(self.view.nearest_hist_metric_combo):
            self.view.set_nearest_hist_metric(self.nearest_hist_metric)
        with QtCore.QSignalBlocker(self.view.size_hist_metric_combo):
            self.view.set_size_hist_metric(self.size_hist_metric)
        with QtCore.QSignalBlocker(self.view.area_hist_metric_combo):
            self.view.set_area_hist_metric(self.area_hist_metric)
        with QtCore.QSignalBlocker(self.view.aspect_hist_metric_combo):
            self.view.set_aspect_hist_metric(self.aspect_hist_metric)
        with QtCore.QSignalBlocker(self.view.main_graph_metric_combo):
            self.view.set_main_graph_metric(self.main_graph_metric)
        with QtCore.QSignalBlocker(self.view.distribution_metric_combo):
            self.view.set_distribution_metric(self.distribution_metric)
        with QtCore.QSignalBlocker(self.view.filter_input_source_combo):
            self.view.set_filter_input_source(self.filter_input_source)
        self.view.set_distribution_bins_count(max(2, len(self.distribution_edges) - 1) if len(self.distribution_edges) >= 2 else 3)
        self.view.set_distribution_bins_enabled(self.distribution_metric != "none")
        self.view.set_distribution_edges_text(
            self._format_distribution_edges_for_ui(self.distribution_metric, self.distribution_edges)
        )
        if self.distribution_metric != "none" and len(self.distribution_edges) >= 2:
            self.view.set_distribution_status(f"{self.distribution_metric.upper()} bins: {len(self.distribution_edges) - 1}")
        else:
            self.view.set_distribution_status("")
        self._update_save_state_ui()

    @staticmethod
    def _looks_like_saved_session_dir(path: Path) -> bool:
        if not path.exists() or not path.is_dir():
            return False
        if (path / "results.json").is_file():
            return True
        if (path / "instance_masks").is_dir():
            return True
        for ext in ID_MAP_FILE_EXTS:
            if (path / f"instance_ids{ext}").is_file():
                return True
        return False

    @staticmethod
    def _coerce_bool(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, np.integer)):
            return bool(int(value))
        if isinstance(value, float):
            if not np.isfinite(value):
                return None
            return bool(int(value))
        if isinstance(value, str):
            token = value.strip().lower()
            if token in {"1", "true", "yes", "on"}:
                return True
            if token in {"0", "false", "no", "off"}:
                return False
        return None

    @staticmethod
    def _coerce_optional_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            out = float(value)
        except Exception:
            return None
        if not np.isfinite(out):
            return None
        return out

    @staticmethod
    def _resolve_candidate_path(base_dir: Path, value: Any) -> Optional[Path]:
        if not isinstance(value, str):
            return None
        raw = value.strip()
        if not raw:
            return None
        p = Path(raw).expanduser()
        candidates: List[Path] = []
        if p.is_absolute():
            candidates.append(p)
            candidates.append(base_dir / p.name)
        else:
            candidates.append(base_dir / p)
            candidates.append(base_dir / p.name)
        seen: set[str] = set()
        for cand in candidates:
            key = str(cand)
            if key in seen:
                continue
            seen.add(key)
            if cand.exists():
                return cand
        return None

    def _load_saved_results_payload(self, session_dir: Path) -> Dict[str, Any]:
        path = session_dir / "results.json"
        if not path.exists() or not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _resolve_saved_id_map_path(self, session_dir: Path, results_payload: Dict[str, Any]) -> Optional[Path]:
        from_results = self._resolve_candidate_path(session_dir, results_payload.get("instance_id_path"))
        if from_results is not None and from_results.is_file():
            return from_results
        for ext in ID_MAP_FILE_EXTS:
            cand = session_dir / f"instance_ids{ext}"
            if cand.exists() and cand.is_file():
                return cand
        return None

    def _resolve_saved_mask_dir(self, session_dir: Path, results_payload: Dict[str, Any]) -> Optional[Path]:
        from_results = self._resolve_candidate_path(session_dir, results_payload.get("instance_mask_dir"))
        if from_results is not None and from_results.exists() and from_results.is_dir():
            return from_results
        direct = session_dir / "instance_masks"
        if direct.exists() and direct.is_dir():
            return direct
        return None

    def _load_saved_masks_for_state(
        self,
        session_dir: Path,
        state: ImageSessionState,
        results_payload: Dict[str, Any],
    ) -> List[np.ndarray]:
        id_map_path = self._resolve_saved_id_map_path(session_dir, results_payload)
        if id_map_path is not None:
            masks = self._load_masks_from_id_map_for_shape(id_map_path, state.image_bgr.shape[:2])
            if masks:
                return masks

        mask_dir = self._resolve_saved_mask_dir(session_dir, results_payload)
        if mask_dir is not None:
            try:
                masks = load_mask_dir(mask_dir, state.image_bgr.shape[:2])
            except Exception:
                masks = []
            if masks:
                return masks

        if id_map_path is None:
            for ext in ID_MAP_FILE_EXTS:
                cand = session_dir / f"instance_id{ext}"
                if not cand.exists() or not cand.is_file():
                    continue
                masks = self._load_masks_from_id_map_for_shape(cand, state.image_bgr.shape[:2])
                if masks:
                    return masks
        return []

    def _load_saved_prompts(
        self,
        session_dir: Path,
        results_payload: Dict[str, Any],
        expected_count: int,
    ) -> List[Optional[Dict[str, Any]]]:
        prompts: List[Optional[Dict[str, Any]]] = [None] * max(0, int(expected_count))

        def _set_prompt(instance_id_raw: Any, prompt_raw: Any, fallback_idx: int) -> None:
            idx = fallback_idx
            if instance_id_raw is not None:
                try:
                    idx = int(instance_id_raw) - 1
                except Exception:
                    pass
            if idx < 0 or idx >= len(prompts):
                return
            prompt_payload = self._normalize_prompt_payload(prompt_raw)
            if prompt_payload is None:
                return
            prompts[idx] = prompt_payload

        prompt_path = session_dir / "instance_prompts.json"
        if prompt_path.exists() and prompt_path.is_file():
            try:
                payload = json.loads(prompt_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            items = payload.get("instance_prompts", []) if isinstance(payload, dict) else []
            if isinstance(items, list):
                for row_idx, item in enumerate(items):
                    if not isinstance(item, dict):
                        continue
                    inst_id = item.get("instance_id")
                    if inst_id is None:
                        inst_id = item.get("index")
                        if inst_id is not None:
                            try:
                                inst_id = int(inst_id) + 1
                            except Exception:
                                inst_id = None
                    _set_prompt(inst_id, item.get("prompt"), row_idx)

        masks_payload = results_payload.get("masks", []) if isinstance(results_payload, dict) else []
        if isinstance(masks_payload, list):
            for row_idx, item in enumerate(masks_payload):
                if not isinstance(item, dict):
                    continue
                if row_idx < 0 or row_idx >= len(prompts):
                    continue
                if prompts[row_idx] is not None:
                    continue
                _set_prompt(item.get("instance_id"), item.get("prompt"), row_idx)
        return prompts

    def _extract_saved_global_options(self, results_payload: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if not isinstance(results_payload, dict):
            return out

        if "max_distance_nm" in results_payload:
            max_dist = self._coerce_optional_float(results_payload.get("max_distance_nm"))
            if max_dist is not None:
                out["max_distance_nm"] = max(0.0, float(max_dist))
            elif results_payload.get("max_distance_nm") is None:
                out["max_distance_nm"] = None

        include_zero = self._coerce_bool(results_payload.get("include_zero_distance"))
        if include_zero is not None:
            out["include_zero_distance"] = bool(include_zero)

        realtime_calc = self._coerce_bool(results_payload.get("realtime_calc_enabled"))
        if realtime_calc is not None:
            out["realtime_calc_enabled"] = bool(realtime_calc)

        show_bbox = self._coerce_bool(results_payload.get("show_bbox_overlay"))
        if show_bbox is not None:
            out["show_bbox_overlay"] = bool(show_bbox)

        show_axes = self._coerce_bool(results_payload.get("show_axes_overlay"))
        if show_axes is not None:
            out["show_axes_overlay"] = bool(show_axes)

        show_feret_para = self._coerce_bool(results_payload.get("show_feret_parallelogram_overlay"))
        if show_feret_para is not None:
            out["show_feret_parallelogram_overlay"] = bool(show_feret_para)

        show_ellipse = self._coerce_bool(results_payload.get("show_ellipse_overlay"))
        if show_ellipse is not None:
            out["show_ellipse_overlay"] = bool(show_ellipse)

        cluster_th = self._coerce_optional_float(results_payload.get("cluster_threshold_nm"))
        if cluster_th is not None:
            out["cluster_threshold_nm"] = max(0.0, float(cluster_th))

        # Prefer explicit setting value (0/20) to preserve UI toggle semantics.
        fractal_val = results_payload.get("fractal_slides", results_payload.get("fractal_slides_used"))
        fractal_slides = self._coerce_optional_float(fractal_val)
        if fractal_slides is not None:
            out["fractal_slides_setting"] = max(0, int(round(fractal_slides)))

        out["nearest_hist_metric"] = self._normalize_nearest_hist_metric(results_payload.get("nearest_hist_metric", "nearest1"))
        out["size_hist_metric"] = self._normalize_size_hist_metric(results_payload.get("size_hist_metric", "ecd"))
        out["area_hist_metric"] = self._normalize_area_hist_metric(results_payload.get("area_hist_metric", "area"))
        out["aspect_hist_metric"] = self._normalize_aspect_hist_metric(results_payload.get("aspect_hist_metric", "feret"))
        out["main_graph_metric"] = self._normalize_main_graph_metric(results_payload.get("main_graph_metric", "ecd"))
        source = str(results_payload.get("filter_input_source", "filtered")).strip().lower()
        out["filter_input_source"] = "original" if source == "original" else "filtered"

        dist_metric = self._normalize_distribution_metric(results_payload.get("distribution_metric", "none"))
        dist_edges_raw = results_payload.get("distribution_edges")
        dist_edges: List[float] = []
        if isinstance(dist_edges_raw, list):
            for v in dist_edges_raw:
                try:
                    fv = float(v)
                except Exception:
                    continue
                if np.isfinite(fv):
                    dist_edges.append(fv)
        dist_edges = sorted({float(v) for v in dist_edges})
        if dist_metric != "none" and len(dist_edges) >= 2:
            out["distribution_metric"] = dist_metric
            out["distribution_edges"] = dist_edges
        elif dist_metric == "none":
            out["distribution_metric"] = "none"
            out["distribution_edges"] = []
        return out

    def _apply_saved_session_dir_to_state(self, state: ImageSessionState, session_dir: Path) -> Tuple[bool, Dict[str, Any]]:
        if not session_dir.exists() or not session_dir.is_dir():
            return False, {}

        results_payload = self._load_saved_results_payload(session_dir)
        masks = self._load_saved_masks_for_state(session_dir, state, results_payload)
        if not masks:
            return False, {}

        scores = self._load_initial_scores(
            init_mask_id_path=None,
            init_mask_dir=None,
            expected_count=len(masks),
            extra_dirs=[session_dir, session_dir / "instance_masks"],
        )
        prompts = self._load_saved_prompts(session_dir, results_payload, len(masks))

        state.set_masks = []
        for mi, mask in enumerate(masks):
            mask_bin = (mask > 0).astype(np.uint8)
            score = scores[mi] if mi < len(scores) else None
            prompt_data = prompts[mi] if mi < len(prompts) else None
            state.set_masks.append(
                MaskEntry(
                    mask=mask_bin,
                    raw=mask_bin.copy(),
                    score=score,
                    prompt_data=prompt_data,
                )
            )

        pair_results = results_payload.get("pair_results")
        state.pair_results = copy.deepcopy(pair_results) if isinstance(pair_results, list) else []
        summary = results_payload.get("summary")
        state.summary = copy.deepcopy(summary) if isinstance(summary, dict) else {}
        size_summary = results_payload.get("size_summary")
        state.size_summary = copy.deepcopy(size_summary) if isinstance(size_summary, dict) else {}

        cluster_labels_raw = results_payload.get("cluster_labels")
        cluster_labels: List[int] = []
        if isinstance(cluster_labels_raw, list):
            for value in cluster_labels_raw:
                try:
                    cluster_labels.append(int(value))
                except Exception:
                    pass
        if len(cluster_labels) == len(state.set_masks):
            state.cluster_labels = cluster_labels
        else:
            state.cluster_labels = []
        cluster_stats_raw = results_payload.get("cluster_stats")
        state.cluster_stats = copy.deepcopy(cluster_stats_raw) if isinstance(cluster_stats_raw, dict) else {}

        has_calc_outputs = bool(state.pair_results) or bool(state.summary) or bool(state.size_summary)
        state.summary_stale = bool(state.set_masks) and not has_calc_outputs
        state.current = None
        state.current_is_raw_display = False
        state.selected_idx = None
        state.selected_indices = set()
        state.prompt_points = []
        state.prompt_box = None
        state.polygon_points = []
        state.undo_stack = []
        state.dirty = False
        state.output_dir = session_dir

        restored_scale = self._coerce_optional_float(results_payload.get("scale_nm_per_px"))
        if restored_scale is not None and restored_scale > 0:
            state.scale_nm_per_px = float(restored_scale)
        raw_adjust = results_payload.get("filter_adjustments")
        if not isinstance(raw_adjust, dict):
            raw_adjust = {
                "brightness": results_payload.get("filter_brightness", 0),
                "contrast": results_payload.get("filter_contrast", 1.0),
                "gamma": results_payload.get("filter_gamma", 1.0),
            }
        loaded_adjust = self._normalize_filter_adjustments(raw_adjust)
        state.filter_brightness = int(loaded_adjust["brightness"])
        state.filter_contrast = float(loaded_adjust["contrast"])
        state.filter_gamma = float(loaded_adjust["gamma"])
        loaded_spatial = self._normalize_spatial_filter_chain(results_payload.get("spatial_filter_chain"))
        loaded_frequency = self._normalize_frequency_filter_chain(results_payload.get("frequency_filter_chain"))
        if not loaded_spatial and not loaded_frequency:
            loaded_spatial, loaded_frequency = self._split_filter_chain_by_domain(results_payload.get("filter_chain"))
        state.spatial_filter_chain = loaded_spatial
        state.frequency_filter_chain = loaded_frequency
        state.filter_chain = [*loaded_spatial, *loaded_frequency]
        state.spatial_filter_selected_row = -1
        state.frequency_filter_selected_row = -1
        state.filter_selected_domain = "spatial"
        state.filter_fft_mode = False
        self._recompute_filtered_image_for_state(state)

        ih, iw = state.image_bgr.shape[:2]
        state.view_center_x = float(iw) / 2.0
        state.view_center_y = float(ih) / 2.0
        state.zoom_factor = 1.0
        return True, self._extract_saved_global_options(results_payload)

    def _read_saved_index_mapping_from_summary(self, root_dir: Path) -> Dict[int, Path]:
        mapping: Dict[int, Path] = {}

        image_path_map: Dict[str, int] = {}
        for idx, image_path in enumerate(self.image_paths):
            try:
                key = str(image_path.resolve())
            except Exception:
                key = str(image_path)
            image_path_map[key] = idx

        stem_to_indices: Dict[str, List[int]] = {}
        for idx, image_path in enumerate(self.image_paths):
            stem_to_indices.setdefault(image_path.stem.lower(), []).append(idx)

        csv_candidates = [
            root_dir / "all_images_index.csv",
            root_dir / "all_images_summary.csv",  # backward-compatible fallback
        ]
        rows: List[Dict[str, str]] = []
        for csv_path in csv_candidates:
            if not csv_path.exists() or not csv_path.is_file():
                continue
            try:
                with csv_path.open("r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    loaded = [r for r in reader if isinstance(r, dict)]
            except Exception:
                loaded = []
            if loaded:
                rows = loaded
                break
        if not rows:
            return mapping

        for row in rows:
            idx: Optional[int] = None

            image_path_raw = (row.get("image_path") or "").strip()
            if image_path_raw:
                try:
                    image_key = str(Path(image_path_raw).expanduser().resolve())
                except Exception:
                    image_key = image_path_raw
                idx = image_path_map.get(image_key)

            if idx is None:
                image_name_raw = (row.get("image_name") or "").strip()
                if image_name_raw:
                    stem = Path(image_name_raw).stem.lower()
                    candidates = stem_to_indices.get(stem, [])
                    if len(candidates) == 1:
                        idx = candidates[0]

            if idx is None:
                image_index_raw = (row.get("image_index") or "").strip()
                try:
                    guess = int(float(image_index_raw)) - 1
                except Exception:
                    guess = -1
                if 0 <= guess < len(self.image_paths):
                    idx = guess

            if idx is None or idx in mapping:
                continue

            out_dir = self._resolve_candidate_path(root_dir, row.get("output_dir"))
            if out_dir is None or not out_dir.exists() or not out_dir.is_dir():
                continue
            mapping[idx] = out_dir
        return mapping

    def _find_saved_dir_for_image(self, root_dir: Path, image_idx: int) -> Optional[Path]:
        if not root_dir.exists() or not root_dir.is_dir():
            return None
        image_path = self.image_paths[image_idx]
        prefix = f"{image_idx + 1:04d}_{image_path.stem}"
        candidates = [
            p for p in root_dir.iterdir()
            if p.is_dir() and p.name.startswith(prefix) and self._looks_like_saved_session_dir(p)
        ]
        if not candidates:
            stem_l = image_path.stem.lower()
            candidates = [
                p for p in root_dir.iterdir()
                if p.is_dir() and stem_l in p.name.lower() and self._looks_like_saved_session_dir(p)
            ]
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    def _discover_saved_session_dirs(self, selected_dir: Path) -> Dict[int, Path]:
        if self._looks_like_saved_session_dir(selected_dir):
            return {int(self.current_image_idx): selected_dir}

        mapping = self._read_saved_index_mapping_from_summary(selected_dir)
        for idx in range(len(self.image_sessions)):
            if idx in mapping:
                continue
            found = self._find_saved_dir_for_image(selected_dir, idx)
            if found is not None:
                mapping[idx] = found
        return mapping

    def on_load_session(self) -> None:
        start_dir = str(self.output_dir if self.output_dir.exists() else self.base_output_dir)
        selected = QFileDialog.getExistingDirectory(self.view, "Select saved session directory", start_dir)
        if not selected:
            return
        selected_dir = Path(selected)
        mapping = self._discover_saved_session_dirs(selected_dir)
        if not mapping:
            QMessageBox.warning(
                self.view,
                "Load Session",
                f"No loadable session data found in:\n{selected_dir}\n"
                "Expected results.json with instance_ids.* or instance_masks/.",
            )
            return

        self._persist_current_session_state()
        loaded_indices: List[int] = []
        first_options: Dict[str, Any] = {}
        for idx, session_dir in sorted(mapping.items()):
            if idx < 0 or idx >= len(self.image_sessions):
                continue
            ok, options = self._apply_saved_session_dir_to_state(self.image_sessions[idx], session_dir)
            if not ok:
                continue
            loaded_indices.append(idx)
            if not first_options:
                first_options = options

        if not loaded_indices:
            QMessageBox.warning(
                self.view,
                "Load Session",
                f"Session directories were found, but no masks could be loaded from:\n{selected_dir}",
            )
            return

        for idx in loaded_indices:
            self.unsaved_images.discard(idx)

        if "max_distance_nm" in first_options:
            self.max_distance_nm = first_options["max_distance_nm"]
        if "realtime_calc_enabled" in first_options:
            self.realtime_calc_enabled = bool(first_options["realtime_calc_enabled"])
        if "include_zero_distance" in first_options:
            self.include_zero_distance = bool(first_options["include_zero_distance"])
        if "show_bbox_overlay" in first_options:
            self.show_bbox_overlay = bool(first_options["show_bbox_overlay"])
        if "show_axes_overlay" in first_options:
            self.show_axes_overlay = bool(first_options["show_axes_overlay"])
        if "show_feret_parallelogram_overlay" in first_options:
            self.show_feret_parallelogram_overlay = bool(first_options["show_feret_parallelogram_overlay"])
        feret_enabled = bool(self.show_axes_overlay or self.show_feret_parallelogram_overlay)
        self.show_axes_overlay = feret_enabled
        self.show_feret_parallelogram_overlay = feret_enabled
        if "show_ellipse_overlay" in first_options:
            self.show_ellipse_overlay = bool(first_options["show_ellipse_overlay"])
        if "cluster_threshold_nm" in first_options:
            self.cluster_threshold_nm = float(first_options["cluster_threshold_nm"])
        if "fractal_slides_setting" in first_options:
            self.fractal_slides_setting = int(first_options["fractal_slides_setting"])
        if "nearest_hist_metric" in first_options:
            self.nearest_hist_metric = self._normalize_nearest_hist_metric(first_options["nearest_hist_metric"])
        if "size_hist_metric" in first_options:
            self.size_hist_metric = self._normalize_size_hist_metric(first_options["size_hist_metric"])
        if "area_hist_metric" in first_options:
            self.area_hist_metric = self._normalize_area_hist_metric(first_options["area_hist_metric"])
        if "aspect_hist_metric" in first_options:
            self.aspect_hist_metric = self._normalize_aspect_hist_metric(first_options["aspect_hist_metric"])
        if "main_graph_metric" in first_options:
            self.main_graph_metric = self._normalize_main_graph_metric(first_options["main_graph_metric"])
        if "filter_input_source" in first_options:
            source = str(first_options["filter_input_source"]).strip().lower()
            self.filter_input_source = "original" if source == "original" else "filtered"
        if "distribution_metric" in first_options:
            self.distribution_metric = self._normalize_distribution_metric(first_options["distribution_metric"])
        if "distribution_edges" in first_options and isinstance(first_options["distribution_edges"], list):
            vals: List[float] = []
            for v in first_options["distribution_edges"]:
                try:
                    fv = float(v)
                except Exception:
                    continue
                if np.isfinite(fv):
                    vals.append(fv)
            vals = sorted({float(v) for v in vals})
            self.distribution_edges = vals if len(vals) >= 2 else []
            if len(self.distribution_edges) < 2:
                self.distribution_metric = "none"

        self._eval_gt_source_cache.clear()
        self._eval_gt_instances_cache.clear()
        self._eval_current_result = {}
        self._eval_all_result = {}
        self._update_eval_plot_preview()
        self.view.set_eval_status_text("Idle")
        self._last_save_note = f"loaded {self._now_text()}"

        self._apply_image_session(self.current_image_idx)
        with QtCore.QSignalBlocker(self.view.realtime_calc_checkbox):
            self.view.set_realtime_calc_checked(self.realtime_calc_enabled)
        with QtCore.QSignalBlocker(self.view.include_zero_checkbox):
            self.view.set_include_zero_checked(self.include_zero_distance)
        with QtCore.QSignalBlocker(self.view.fractal_checkbox):
            self.view.set_fractal_checked(self.fractal_slides_setting >= 20)
        with QtCore.QSignalBlocker(self.view.nearest_hist_metric_combo):
            self.view.set_nearest_hist_metric(self.nearest_hist_metric)
        with QtCore.QSignalBlocker(self.view.show_bbox_checkbox):
            self.view.set_show_bbox_checked(self.show_bbox_overlay)
        with QtCore.QSignalBlocker(self.view.show_feret_checkbox):
            self.view.set_show_feret_checked(self.show_axes_overlay)
        with QtCore.QSignalBlocker(self.view.show_ellipse_checkbox):
            self.view.set_show_ellipse_checked(self.show_ellipse_overlay)
        with QtCore.QSignalBlocker(self.view.size_hist_metric_combo):
            self.view.set_size_hist_metric(self.size_hist_metric)
        with QtCore.QSignalBlocker(self.view.area_hist_metric_combo):
            self.view.set_area_hist_metric(self.area_hist_metric)
        with QtCore.QSignalBlocker(self.view.aspect_hist_metric_combo):
            self.view.set_aspect_hist_metric(self.aspect_hist_metric)
        with QtCore.QSignalBlocker(self.view.main_graph_metric_combo):
            self.view.set_main_graph_metric(self.main_graph_metric)
        with QtCore.QSignalBlocker(self.view.distribution_metric_combo):
            self.view.set_distribution_metric(self.distribution_metric)
        with QtCore.QSignalBlocker(self.view.filter_input_source_combo):
            self.view.set_filter_input_source(self.filter_input_source)
        self.view.set_distribution_bins_count(max(2, len(self.distribution_edges) - 1) if len(self.distribution_edges) >= 2 else 3)
        self.view.set_distribution_bins_enabled(self.distribution_metric != "none")
        self.view.set_distribution_edges_text(
            self._format_distribution_edges_for_ui(self.distribution_metric, self.distribution_edges)
        )
        if self.distribution_metric != "none" and len(self.distribution_edges) >= 2:
            self.view.set_distribution_status(f"{self.distribution_metric.upper()} bins: {len(self.distribution_edges) - 1}")
        else:
            self.view.set_distribution_status("")
        self._sync_cluster_controls()
        self._update_save_state_ui()
        self._refresh()

        QMessageBox.information(
            self.view,
            "Load Session",
            f"Loaded {len(loaded_indices)} image session(s) from:\n{selected_dir}",
        )

    def _on_autosave_timer(self) -> None:
        if not self.unsaved_images:
            return
        try:
            self._write_recovery_snapshot()
            self.view.set_save_state(f"Unsaved ({len(self.unsaved_images)}) · Autosaved", True)
        except Exception:
            # Keep UI functional even if snapshot I/O fails.
            pass

    def _persist_current_session_state(self) -> None:
        state = self.image_sessions[self.current_image_idx]
        state.output_dir = self.output_dir
        state.scale_nm_per_px = float(self.scale_nm_per_px)
        state.set_masks = self.set_masks
        state.current = self.current
        state.current_is_raw_display = self.current_is_raw_display
        state.pair_results = self.pair_results
        state.summary = self.summary
        state.size_summary = self.size_summary
        state.summary_stale = self.summary_stale
        state.selected_idx = self.selected_idx
        state.selected_indices = set(self.selected_indices)
        state.prompt_points = self.prompt_points
        state.prompt_box = self.prompt_box
        state.polygon_points = self.polygon_points
        state.cluster_labels = self.cluster_labels
        state.cluster_stats = self.cluster_stats
        state.undo_stack = list(self._undo_stack)
        state.view_center_x = self._view_center_x
        state.view_center_y = self._view_center_y
        state.zoom_factor = self.zoom_factor
        state.dirty = bool(self.current_image_idx in self.unsaved_images)
        state.filter_brightness = int(self.filter_brightness)
        state.filter_contrast = float(self.filter_contrast)
        state.filter_gamma = float(self.filter_gamma)
        state.spatial_filter_chain = copy.deepcopy(self.spatial_filter_chain)
        state.frequency_filter_chain = copy.deepcopy(self.frequency_filter_chain)
        state.spatial_filter_selected_row = int(self.spatial_filter_selected_row)
        state.frequency_filter_selected_row = int(self.frequency_filter_selected_row)
        state.filter_selected_domain = str(self.filter_selected_domain)
        state.filter_chain = copy.deepcopy(self._combined_filter_chain())
        state.filtered_image_bgr = self.filtered_image_bgr.copy() if isinstance(self.filtered_image_bgr, np.ndarray) else None
        state.filter_fft_mode = bool(self.filter_fft_mode)

    def _apply_image_session(self, index: int) -> None:
        state = self.image_sessions[index]
        self.current_image_idx = index
        self.base_image_path = state.image_path
        self.image_bgr = state.image_bgr
        self.output_dir = state.output_dir
        self.scale_nm_per_px = float(max(state.scale_nm_per_px, 0.0001))
        self.set_masks = state.set_masks
        self.current = state.current
        self.current_is_raw_display = state.current_is_raw_display
        self.pair_results = state.pair_results
        self.summary = state.summary
        self.size_summary = state.size_summary
        self.summary_stale = state.summary_stale
        self.selected_idx = state.selected_idx
        self.selected_indices = {
            int(i) for i in state.selected_indices
            if 0 <= int(i) < len(state.set_masks)
        }
        if self.selected_idx is not None and 0 <= self.selected_idx < len(state.set_masks):
            self.selected_indices.add(int(self.selected_idx))
        elif self.selected_indices:
            self.selected_idx = max(self.selected_indices)
        else:
            self.selected_idx = None
        self.prompt_points = state.prompt_points
        self.prompt_box = state.prompt_box
        self.drag_prompt_box = None
        self.polygon_points = state.polygon_points
        self.scale_bar_points = []
        self.scale_bar_px_length = None
        self.cluster_labels = state.cluster_labels
        self.cluster_stats = state.cluster_stats
        self._undo_stack = list(state.undo_stack)
        self._last_undoable_action = ""
        self.filter_brightness = int(state.filter_brightness)
        self.filter_contrast = float(state.filter_contrast)
        self.filter_gamma = float(state.filter_gamma)
        spatial_chain = self._normalize_spatial_filter_chain(getattr(state, "spatial_filter_chain", []))
        frequency_chain = self._normalize_frequency_filter_chain(getattr(state, "frequency_filter_chain", []))
        if not spatial_chain and not frequency_chain:
            spatial_chain, frequency_chain = self._split_filter_chain_by_domain(state.filter_chain)
        self.spatial_filter_chain = spatial_chain
        self.frequency_filter_chain = frequency_chain
        state.spatial_filter_chain = copy.deepcopy(self.spatial_filter_chain)
        state.frequency_filter_chain = copy.deepcopy(self.frequency_filter_chain)
        self.filter_chain = self._combined_filter_chain()
        state.filter_chain = copy.deepcopy(self.filter_chain)
        self.spatial_filter_selected_row = int(getattr(state, "spatial_filter_selected_row", -1))
        self.frequency_filter_selected_row = int(getattr(state, "frequency_filter_selected_row", -1))
        self.filter_selected_domain = str(getattr(state, "filter_selected_domain", "spatial") or "spatial").strip().lower()
        if self.filter_selected_domain not in {"spatial", "frequency"}:
            self.filter_selected_domain = "spatial"
        if isinstance(state.filtered_image_bgr, np.ndarray):
            self.filtered_image_bgr = state.filtered_image_bgr
        else:
            self._recompute_filtered_image_for_state(state)
            self.filtered_image_bgr = state.filtered_image_bgr if state.filtered_image_bgr is not None else state.image_bgr
        self.filter_fft_mode = bool(state.filter_fft_mode)
        self.zoom_factor = float(np.clip(state.zoom_factor, self.zoom_min, self.zoom_max))
        self._view_center_x = state.view_center_x
        self._view_center_y = state.view_center_y
        if state.dirty:
            self.unsaved_images.add(index)
        else:
            self.unsaved_images.discard(index)

        h, w = self.image_bgr.shape[:2]
        self._view_x0 = 0.0
        self._view_y0 = 0.0
        self._view_w = float(w)
        self._view_h = float(h)
        self._disp_size = (0, 0)
        self._disp_origin = (0, 0)
        self._scale_x = 1.0
        self._scale_y = 1.0
        self._mark_overlay_dirty()
        self._mark_analysis_dirty()
        self._refresh_predictor_image()
        self.view.set_filter_fft_mode(self.filter_fft_mode)
        self._refresh_filter_ui()
        self.view.set_scale_text(f"{self.scale_nm_per_px:.3f}")
        self._sync_scale_preset_ui_from_scale()
        self.view.set_scale_calibration_pixels(None)

    def _switch_image(self, index: int) -> None:
        if index < 0 or index >= len(self.image_sessions):
            return
        if index == self.current_image_idx:
            return
        self._persist_current_session_state()
        self._apply_image_session(index)
        self._update_save_state_ui()
        self._refresh()

    def _update_image_status_ui(self) -> None:
        if self.workspace_tab == self.WORKSPACE_TRAIN:
            train_note = self._train_preview_note if self._train_preview_note else "Train preview"
            self.view.set_image_status_text(train_note)
            self.view.set_image_nav_enabled(False, False)
            self._sync_eval_default_paths()
            return
        if self.workspace_tab == self.WORKSPACE_TRACK:
            track_note = self._track_preview_note if self._track_preview_note else "Track preview"
            self.view.set_image_status_text(track_note)
            self.view.set_image_nav_enabled(False, False)
            self._sync_eval_default_paths()
            return
        total = len(self.image_sessions)
        idx = self.current_image_idx + 1
        self.view.set_image_status_text(f"Image {idx}/{total}: {self.base_image_path.name}")
        self.view.set_image_nav_enabled(self.current_image_idx > 0, self.current_image_idx < total - 1)
        self._sync_eval_default_paths()

    def _sync_eval_default_paths(self) -> None:
        self._update_gt_overlay_availability()

    def _update_gt_overlay_availability(self) -> None:
        enabled = self._resolve_eval_roi_source_for_image(self.current_image_idx, allow_shared_dir=True) is not None
        self.view.set_show_gt_enabled(enabled)
        if not enabled and self.show_gt_overlay:
            self.show_gt_overlay = False
            self.view.set_show_gt_checked(False)

    def _resolve_eval_roi_source_for_image(self, image_idx: int, allow_shared_dir: bool) -> Optional[Path]:
        roi_text = self.view.eval_gt_roi_edit.text().strip() if hasattr(self.view, "eval_gt_roi_edit") else ""
        cache_key = (int(image_idx), roi_text, bool(allow_shared_dir))
        if cache_key in self._eval_gt_source_cache:
            return self._eval_gt_source_cache[cache_key]
        if not roi_text:
            self._eval_gt_source_cache[cache_key] = None
            return None
        root = Path(roi_text)
        image_stem = self.image_sessions[image_idx].image_path.stem
        source = resolve_eval_gt_source(root, image_stem, allow_shared_dir=allow_shared_dir)
        self._eval_gt_source_cache[cache_key] = source
        return source

    def _get_eval_gt_instances(
        self,
        image_idx: int,
        allow_shared_dir: bool,
        silent: bool = True,
    ) -> Optional[List[np.ndarray]]:
        roi_text = self.view.eval_gt_roi_edit.text().strip() if hasattr(self.view, "eval_gt_roi_edit") else ""
        cache_key = (int(image_idx), roi_text, bool(allow_shared_dir))
        if cache_key in self._eval_gt_instances_cache:
            return self._eval_gt_instances_cache[cache_key]

        source = self._resolve_eval_roi_source_for_image(image_idx, allow_shared_dir=allow_shared_dir)
        if source is None:
            self._eval_gt_instances_cache[cache_key] = None
            return None

        shape = self.image_sessions[image_idx].image_bgr.shape[:2]
        image_stem = self.image_sessions[image_idx].image_path.stem
        try:
            instances = load_eval_gt_instances(
                source=source,
                image_stem=image_stem,
                shape=shape,
                min_area=self._eval_gt_min_area,
                tmp_root=self.base_output_dir / ".eval_tmp",
                allow_shared_dir=allow_shared_dir,
            )
            out = [m.astype(bool) for m in instances if int(m.sum()) >= self._eval_gt_min_area]
        except Exception as exc:
            if not silent:
                QMessageBox.warning(self.view, "GT Data Error", f"Failed to load GT data ({source}):\n{exc}")
            out = None
        self._eval_gt_instances_cache[cache_key] = out
        return out

    def _sync_workspace_ui(self) -> None:
        seg_workspace_active = self._is_segmentation_workspace_active()
        preview_tab_active = seg_workspace_active
        preview_editable = preview_tab_active and not (self.workspace_tab == self.WORKSPACE_FILTERS and self.filter_fft_mode)
        analyze_tab_active = self.workspace_tab == self.WORKSPACE_ANALYZE and self.view.is_analyze_tab_active()
        evaluate_tab_active = self.workspace_tab == self.WORKSPACE_ANALYZE and self.view.is_evaluate_tab_active()
        self.view.set_action_mode(self.mode)
        self.view.preview_control_card.setVisible(preview_tab_active)
        if hasattr(self.view, "image_actions_card"):
            self.view.image_actions_card.setVisible(preview_tab_active)
        self.view.btn_set.setVisible(preview_tab_active)
        self.view.btn_remove.setVisible(preview_tab_active)
        self.view.btn_undo.setVisible(preview_tab_active)
        self.view.btn_reset.setVisible(False)
        self.view.btn_save.setVisible(preview_tab_active)
        self.view.btn_load_session.setVisible(preview_tab_active)
        self.view.btn_save_all.setVisible(preview_tab_active)
        self.view.btn_calc.setVisible(False)
        if hasattr(self.view, "btn_calc_current"):
            self.view.btn_calc_current.setVisible(analyze_tab_active)
        if hasattr(self.view, "btn_calc_all"):
            self.view.btn_calc_all.setVisible(analyze_tab_active)
        if hasattr(self.view, "eval_run_current_btn"):
            self.view.eval_run_current_btn.setVisible(evaluate_tab_active)
        if hasattr(self.view, "eval_run_all_btn"):
            self.view.eval_run_all_btn.setVisible(evaluate_tab_active)
        for w in [self.view.lora_path_edit, self.view.lora_browse_btn]:
            w.setEnabled(preview_tab_active)
        for btn in [
            self.view.btn_set,
            self.view.btn_remove,
            self.view.btn_undo,
            self.view.btn_reset,
            self.view.btn_save,
            self.view.btn_save_all,
            self.view.btn_load_session,
        ]:
            btn.setEnabled(preview_editable)
        for btn in [self.view.mode_sam_btn, self.view.mode_lora_btn, self.view.mode_polygon_btn, self.view.scale_calib_toggle_btn]:
            btn.setEnabled(preview_editable and (btn is not self.view.mode_lora_btn or self.lora_mode_available))
        self.view.btn_calc.setEnabled(analyze_tab_active)
        if hasattr(self.view, "btn_calc_current"):
            self.view.btn_calc_current.setEnabled(analyze_tab_active)
        if hasattr(self.view, "btn_calc_all"):
            self.view.btn_calc_all.setEnabled(analyze_tab_active)
        if hasattr(self.view, "eval_run_current_btn"):
            self.view.eval_run_current_btn.setEnabled(evaluate_tab_active)
        if hasattr(self.view, "eval_run_all_btn"):
            self.view.eval_run_all_btn.setEnabled(evaluate_tab_active)
        action_states = [
            ("act_set", preview_editable),
            ("act_undo", preview_editable),
            ("act_remove", preview_editable),
            ("act_reset", preview_editable),
            ("act_calc", analyze_tab_active),
        ]
        for action_name, enabled in action_states:
            action = getattr(self.view, action_name, None)
            if action is not None:
                action.setEnabled(enabled)

    def on_workspace_tab_changed(self, index: int) -> None:
        self.workspace_tab = int(index)
        self._clear_sym_notch_drag_state()
        if self.workspace_tab == self.WORKSPACE_FILTERS:
            self._refresh_filter_ui()
        self._mark_overlay_dirty()
        self._sync_workspace_ui()
        self._refresh()

    def on_preview_analyze_tab_changed(self, _index: int) -> None:
        self._sync_workspace_ui()
        if self.workspace_tab == self.WORKSPACE_ANALYZE and self.view.is_evaluate_tab_active():
            self._update_eval_plot_preview()
        self._refresh()

    def on_shortcut_open_evaluate(self) -> None:
        if self.workspace_tab != self.WORKSPACE_ANALYZE:
            self.workspace_tab = self.WORKSPACE_ANALYZE
            self.view.set_workspace_tab(self.WORKSPACE_ANALYZE)
        self.view.set_preview_analyze_tab("evaluate")
        self._sync_workspace_ui()
        self._update_eval_plot_preview()
        self._refresh()

    def on_eval_plot_scope_changed(self, scope: str) -> None:
        self.view.set_eval_scope(scope)
        if self.workspace_tab == self.WORKSPACE_ANALYZE and self.view.is_evaluate_tab_active():
            self._refresh()

    def on_shortcut_open_analyze_and_calc(self) -> None:
        if self.workspace_tab != self.WORKSPACE_ANALYZE:
            self.workspace_tab = self.WORKSPACE_ANALYZE
            self.view.set_workspace_tab(self.WORKSPACE_ANALYZE)
        self.view.set_preview_analyze_tab("analyze")
        self._sync_workspace_ui()
        self.analysis_scope = "current"
        self.on_calc()

    def on_analysis_scope_changed(self, scope: str) -> None:
        target = "all" if (scope or "").strip().lower() == "all" else "current"
        if target == self.analysis_scope:
            return
        self.analysis_scope = target
        self._mark_analysis_dirty()
        self._refresh()

    @staticmethod
    def _stats_from_values(values: Sequence[float]) -> Dict[str, Optional[float]]:
        if not values:
            return {
                "count": 0,
                "mean": None,
                "median": None,
                "std": None,
                "cv_pct": None,
                "min": None,
                "max": None,
            }
        arr_np = np.asarray(values, dtype=np.float32)
        if arr_np.size == 0:
            return {
                "count": 0,
                "mean": None,
                "median": None,
                "std": None,
                "cv_pct": None,
                "min": None,
                "max": None,
            }
        mean = float(arr_np.mean())
        median = float(np.median(arr_np))
        std = float(arr_np.std())
        cv_pct = float((std / mean) * 100.0) if abs(mean) > 1e-12 else None
        return {
            "count": int(arr_np.size),
            "mean": mean,
            "median": median,
            "std": std,
            "cv_pct": cv_pct,
            "min": float(arr_np.min()),
            "max": float(arr_np.max()),
        }

    def _analysis_scope_payload(self) -> Dict[str, Any]:
        scope = "all" if self.analysis_scope == "all" else "current"
        if scope != "all":
            return {
                "scope": "current",
                "mask_count": len(self.set_masks),
                "summary": self.summary if self.summary else {},
                "size_summary": self.size_summary if self.size_summary else {},
                "pending": bool(self.summary_stale and len(self.set_masks) > 0),
            }

        # Keep aggregate view in sync with unswitched edits on the current image.
        self._persist_current_session_state()
        mask_count = 0
        pending = False
        hist_first: List[float] = []
        hist_second: List[float] = []
        hist_c1: List[float] = []
        hist_c2: List[float] = []
        hist_ecd: List[float] = []
        hist_vesd: List[float] = []
        hist_area: List[float] = []
        hist_bbox_area: List[float] = []
        hist_feret_rect_area: List[float] = []
        hist_area_vesd: List[float] = []
        hist_major: List[float] = []
        hist_minor: List[float] = []
        hist_ellipse_major: List[float] = []
        hist_ellipse_minor: List[float] = []
        hist_aspect: List[float] = []
        hist_shape: List[float] = []
        hist_fractal: List[float] = []
        fg_eps_all: List[float] = []
        fg_counts_all: List[float] = []
        fractal_per_image: List[Dict[str, Any]] = []

        for idx, state in enumerate(self.image_sessions):
            mask_count += len(state.set_masks)
            if state.summary_stale and len(state.set_masks) > 0:
                pending = True
            s = state.summary if isinstance(state.summary, dict) else {}
            sz = state.size_summary if isinstance(state.size_summary, dict) else {}
            hist_first.extend([float(v) for v in (s.get("hist_first") or [])])
            hist_second.extend([float(v) for v in (s.get("hist_second") or [])])
            hist_c1.extend([float(v) for v in (s.get("hist_centroid1") or [])])
            hist_c2.extend([float(v) for v in (s.get("hist_centroid2") or [])])
            hist_ecd.extend([float(v) for v in (sz.get("hist_ecd") or [])])
            hist_vesd.extend([float(v) for v in (sz.get("hist_vesd") or [])])
            hist_area.extend([float(v) for v in (sz.get("hist_area") or [])])
            hist_bbox_area.extend([float(v) for v in (sz.get("hist_bbox_area") or [])])
            hist_feret_rect_area.extend([float(v) for v in (sz.get("hist_feret_rect_area") or [])])
            hist_area_vesd.extend([float(v) for v in (sz.get("hist_area_vesd") or [])])
            hist_major.extend([float(v) for v in (sz.get("hist_major_axis") or [])])
            hist_minor.extend([float(v) for v in (sz.get("hist_minor_axis") or [])])
            hist_ellipse_major.extend([float(v) for v in (sz.get("hist_ellipse_major_axis") or [])])
            hist_ellipse_minor.extend([float(v) for v in (sz.get("hist_ellipse_minor_axis") or [])])
            hist_aspect.extend([float(v) for v in (sz.get("hist_aspect_ratio") or [])])
            hist_shape.extend([float(v) for v in (sz.get("hist_shape_ratio") or [])])
            hist_fractal.extend([float(v) for v in (sz.get("hist_fractal") or [])])
            fg = sz.get("fractal_global", {}) if isinstance(sz, dict) else {}
            eps_raw = fg.get("log_eps") if isinstance(fg, dict) else None
            cnt_raw = fg.get("log_counts") if isinstance(fg, dict) else None
            fg_eps_one: List[float] = []
            fg_counts_one: List[float] = []
            if isinstance(eps_raw, list) and isinstance(cnt_raw, list) and len(eps_raw) == len(cnt_raw):
                for e, c in zip(eps_raw, cnt_raw):
                    try:
                        ev = float(e)
                        cv = float(c)
                    except Exception:
                        continue
                    if not np.isfinite(ev) or not np.isfinite(cv):
                        continue
                    fg_eps_all.append(ev)
                    fg_counts_all.append(cv)
                    fg_eps_one.append(ev)
                    fg_counts_one.append(cv)
            if len(fg_eps_one) >= 2 and len(fg_counts_one) >= 2:
                slope_raw = fg.get("slope") if isinstance(fg, dict) else None
                slope_v: Optional[float]
                try:
                    slope_v = float(slope_raw)
                    if not np.isfinite(slope_v):
                        slope_v = None
                except Exception:
                    slope_v = None
                fractal_per_image.append(
                    {
                        "image_index": idx + 1,
                        "image_name": state.image_path.name,
                        "log_eps": fg_eps_one,
                        "log_counts": fg_counts_one,
                        "slope": slope_v,
                    }
                )

        summary_all = {
            "nearest1": self._stats_from_values(hist_first),
            "nearest2": self._stats_from_values(hist_second),
            "hist_first": hist_first,
            "hist_second": hist_second,
            "centroid1": self._stats_from_values(hist_c1),
            "centroid2": self._stats_from_values(hist_c2),
            "hist_centroid1": hist_c1,
            "hist_centroid2": hist_c2,
        }

        fg_slope: Optional[float] = None
        fg_value: Optional[float] = None
        fg_eps: List[float] = []
        fg_counts: List[float] = []
        if len(fg_eps_all) >= 2 and len(fg_counts_all) >= 2:
            eps_arr = np.asarray(fg_eps_all, dtype=np.float32)
            cnt_arr = np.asarray(fg_counts_all, dtype=np.float32)
            order = np.argsort(eps_arr)
            eps_arr = eps_arr[order]
            cnt_arr = cnt_arr[order]
            if eps_arr.size >= 2:
                slope, _ = np.polyfit(eps_arr, cnt_arr, 1)
                fg_slope = float(slope)
                fg_value = -float(slope)
                fg_eps = list(map(float, eps_arr.tolist()))
                fg_counts = list(map(float, cnt_arr.tolist()))

        size_all = {
            "ecd": self._stats_from_values(hist_ecd),
            "hist_ecd": hist_ecd,
            "vesd": self._stats_from_values(hist_vesd),
            "hist_vesd": hist_vesd,
            "area": self._stats_from_values(hist_area),
            "hist_area": hist_area,
            "bbox_area": self._stats_from_values(hist_bbox_area),
            "hist_bbox_area": hist_bbox_area,
            "feret_rect_area": self._stats_from_values(hist_feret_rect_area),
            "hist_feret_rect_area": hist_feret_rect_area,
            "area_vesd": self._stats_from_values(hist_area_vesd),
            "hist_area_vesd": hist_area_vesd,
            "major_axis": self._stats_from_values(hist_major),
            "hist_major_axis": hist_major,
            "minor_axis": self._stats_from_values(hist_minor),
            "hist_minor_axis": hist_minor,
            "ellipse_major_axis": self._stats_from_values(hist_ellipse_major),
            "hist_ellipse_major_axis": hist_ellipse_major,
            "ellipse_minor_axis": self._stats_from_values(hist_ellipse_minor),
            "hist_ellipse_minor_axis": hist_ellipse_minor,
            "aspect_ratio": self._stats_from_values(hist_aspect),
            "hist_aspect_ratio": hist_aspect,
            "shape_ratio": self._stats_from_values(hist_shape),
            "hist_shape_ratio": hist_shape,
            "fractal": self._stats_from_values(hist_fractal),
            "hist_fractal": hist_fractal,
            "fractal_curve": {"log_eps": [], "log_counts": [], "slope": None, "sizes_px": [], "counts_mean": []},
            "fractal_per_image": fractal_per_image,
            "fractal_global": {
                "value": fg_value,
                "log_eps": fg_eps,
                "log_counts": fg_counts,
                "slope": fg_slope,
                "sizes_px": [],
                "counts": [],
            },
        }
        return {
            "scope": "all",
            "mask_count": int(mask_count),
            "summary": summary_all,
            "size_summary": size_all,
            "pending": bool(pending),
        }

    def _selected_hist_markers(self, analysis_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Optional[float]]:
        payload = analysis_payload if isinstance(analysis_payload, dict) else self._analysis_scope_payload()
        if str(payload.get("scope", "current")) != "current":
            return {}
        if self.summary_stale:
            return {}
        if self.selected_idx is None or not (0 <= int(self.selected_idx) < len(self.set_masks)):
            return {}
        idx = int(self.selected_idx)
        rec = self.set_masks[idx]
        metrics = compute_mask_shape_metrics(rec.mask.astype(np.uint8), self.scale_nm_per_px)
        out: Dict[str, Optional[float]] = {
            "nearest1": None,
            "nearest2": None,
            "ecd": float(metrics.get("ecd_nm")) if metrics.get("ecd_nm") is not None else None,
            "vesd": float(metrics.get("vesd_nm")) if metrics.get("vesd_nm") is not None else None,
            "area": float(metrics.get("area_nm2")) if metrics.get("area_nm2") is not None else None,
            "area_bbox": float(metrics.get("bbox_area_nm2")) if metrics.get("bbox_area_nm2") is not None else None,
            "area_vesd": float(metrics.get("area_vesd_nm2")) if metrics.get("area_vesd_nm2") is not None else None,
            "aspect": float(metrics.get("aspect_ratio")) if metrics.get("aspect_ratio") is not None else None,
            "aspect_ellipse": (
                float(metrics.get("ellipse_major_axis_nm")) / float(metrics.get("ellipse_minor_axis_nm"))
                if metrics.get("ellipse_major_axis_nm") is not None
                and metrics.get("ellipse_minor_axis_nm") is not None
                and float(metrics.get("ellipse_minor_axis_nm")) > 0.0
                else None
            ),
        }
        if 0 <= idx < len(self.pair_results):
            near = self.pair_results[idx].get("nearest", []) if isinstance(self.pair_results[idx], dict) else []
            if len(near) >= 1:
                try:
                    out["nearest1"] = float(near[0].get("distance_nm"))
                except Exception:
                    out["nearest1"] = None
            if len(near) >= 2:
                try:
                    out["nearest2"] = float(near[1].get("distance_nm"))
                except Exception:
                    out["nearest2"] = None
        return out

    def _run_calc_for_state(self, state: ImageSessionState) -> None:
        masks = [rec.mask for rec in state.set_masks]
        if not masks:
            state.pair_results = []
            state.summary = {}
            state.size_summary = {}
            state.summary_stale = False
            state.cluster_labels = []
            state.cluster_stats = {}
            return
        if len(masks) >= 2:
            pair_results = compute_two_nearest(
                masks,
                float(state.scale_nm_per_px),
                self.max_distance_nm,
                include_zero=self.include_zero_distance,
            )
            summary = summarize(pair_results)
        else:
            pair_results = []
            summary = {}
        size_summary = summarize_sizes(
            masks,
            float(state.scale_nm_per_px),
            fractal_slides=self._fractal_slides_effective(),
        )
        state.pair_results = pair_results
        state.summary = summary
        state.size_summary = size_summary
        state.summary_stale = False

    def _mark_all_sessions_summary_stale(self) -> None:
        for state in self.image_sessions:
            state.summary_stale = bool(state.set_masks)
        self.summary_stale = bool(self.set_masks)

    def on_train_preview_inputs_changed(self, *_args) -> None:
        self._train_preview_key = None
        self._train_preview_overlay = None
        self._train_preview_note = "Train preview: set image/mask dirs."
        if self.workspace_tab == self.WORKSPACE_TRAIN:
            self._refresh()

    def on_track_preview_input_changed(self, *_args) -> None:
        self._track_preview_key = None
        self._track_preview_image = None
        self._track_preview_note = "Track preview: import image."
        if self.workspace_tab == self.WORKSPACE_TRACK:
            self._refresh()

    def _ensure_train_preview_overlay(self) -> None:
        image_dir_text = self.view.train_image_dir_edit.text().strip()
        mask_dir_text = self.view.train_mask_dir_edit.text().strip()
        key = (image_dir_text, mask_dir_text)
        if self._train_preview_key == key and self._train_preview_overlay is not None:
            return
        self._train_preview_key = key
        self._train_preview_overlay = None

        if not image_dir_text or not mask_dir_text:
            self._train_preview_note = "Train preview: set image/mask dirs."
            return
        image_dir = Path(image_dir_text)
        mask_dir = Path(mask_dir_text)
        if not image_dir.exists() or not image_dir.is_dir():
            self._train_preview_note = f"Train preview: image dir not found ({image_dir})."
            return
        if not mask_dir.exists() or not mask_dir.is_dir():
            self._train_preview_note = f"Train preview: mask dir not found ({mask_dir})."
            return

        image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        mask_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        image_files = sorted([p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts])
        mask_files = sorted([p for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() in mask_exts])
        if not image_files:
            self._train_preview_note = f"Train preview: no images in {image_dir}."
            return
        if not mask_files:
            self._train_preview_note = f"Train preview: no masks in {mask_dir}."
            return

        mask_by_stem: Dict[str, Path] = {p.stem: p for p in mask_files}
        chosen_image = image_files[0]
        chosen_mask = mask_by_stem.get(chosen_image.stem)
        if chosen_mask is None:
            for img in image_files:
                m = mask_by_stem.get(img.stem)
                if m is not None:
                    chosen_image = img
                    chosen_mask = m
                    break
        if chosen_mask is None:
            chosen_mask = mask_files[0]

        try:
            image_bgr = load_image_bgr(chosen_image)
        except Exception as exc:
            self._train_preview_note = f"Train preview: failed to load image ({exc})."
            return

        raw_mask = cv2.imread(str(chosen_mask), cv2.IMREAD_UNCHANGED)
        if raw_mask is None:
            self._train_preview_note = f"Train preview: failed to load mask ({chosen_mask.name})."
            return

        ih, iw = image_bgr.shape[:2]
        color_mask = np.zeros((ih, iw, 3), dtype=np.uint8)
        active = np.zeros((ih, iw), dtype=bool)

        if raw_mask.ndim == 3:
            if raw_mask.shape[2] == 4:
                raw_mask = cv2.cvtColor(raw_mask, cv2.COLOR_BGRA2BGR)
            if raw_mask.shape[:2] != (ih, iw):
                raw_mask = cv2.resize(raw_mask, (iw, ih), interpolation=cv2.INTER_NEAREST)
            color_mask = raw_mask[:, :, :3].astype(np.uint8, copy=False)
            active = np.any(color_mask > 0, axis=2)
        else:
            mask_2d = raw_mask
            if mask_2d.shape != (ih, iw):
                mask_2d = cv2.resize(mask_2d, (iw, ih), interpolation=cv2.INTER_NEAREST)
            mask_ids = mask_2d.astype(np.int64, copy=False)
            active = mask_ids > 0
            uniq = [int(v) for v in np.unique(mask_ids) if int(v) > 0]
            if len(uniq) <= 1:
                color_mask[active] = np.array([80, 220, 100], dtype=np.uint8)
            else:
                for inst_id in uniq:
                    hue = int((inst_id * 37) % 180)
                    hsv = np.uint8([[[hue, 200, 255]]])
                    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
                    color_mask[mask_ids == inst_id] = bgr

        overlay = image_bgr.astype(np.float32)
        if active.any():
            overlay[active] = overlay[active] * 0.65 + color_mask[active].astype(np.float32) * 0.35
        self._train_preview_overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        self._train_preview_note = (
            f"Train preview: {chosen_image.name} + {chosen_mask.name}"
            f" ({int(active.sum())} mask px)"
        )

    def _ensure_track_preview_image(self) -> None:
        path_text = self.view.track_image_edit.text().strip()
        key = path_text
        if self._track_preview_key == key and self._track_preview_image is not None:
            return
        self._track_preview_key = key
        self._track_preview_image = None

        if not path_text:
            self._track_preview_note = "Track preview: import image."
            return
        path = Path(path_text)
        if not path.exists() or not path.is_file():
            self._track_preview_note = f"Track preview: file not found ({path})."
            return

        ext = path.suffix.lower()
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        if ext in video_exts:
            cap = cv2.VideoCapture(str(path))
            ok, frame = cap.read()
            cap.release()
            if not ok or frame is None:
                self._track_preview_note = f"Track preview: failed to read video ({path.name})."
                return
            self._track_preview_image = frame
            self._track_preview_note = f"Track preview: {path.name} (first frame)"
            return

        try:
            img = load_image_bgr(path)
        except Exception as exc:
            self._track_preview_note = f"Track preview: failed to load image ({exc})."
            return
        self._track_preview_image = img
        self._track_preview_note = f"Track preview: {path.name}"

    def _build_workspace_placeholder(self, title: str, note: str) -> np.ndarray:
        h, w = self.image_bgr.shape[:2]
        canvas = np.full((h, w, 3), 236, dtype=np.uint8)
        cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), (208, 218, 232), 1)
        cv2.putText(
            canvas,
            title,
            (max(16, w // 18), max(32, h // 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            (56, 72, 98),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            note,
            (max(16, w // 18), max(58, h // 10 + 34)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (78, 92, 114),
            1,
            cv2.LINE_AA,
        )
        return canvas

    def _browse_directory_into(self, edit: QtWidgets.QLineEdit, title: str) -> None:
        current = edit.text().strip() or str(Path.cwd())
        picked = QFileDialog.getExistingDirectory(self.view, title, current)
        if picked:
            edit.setText(picked)

    def _browse_file_into(self, edit: QtWidgets.QLineEdit, title: str, file_filter: str = "All Files (*)") -> None:
        current = edit.text().strip()
        start_dir = str(Path(current).parent) if current else str(Path.cwd())
        picked, _ = QFileDialog.getOpenFileName(self.view, title, start_dir, file_filter)
        if picked:
            edit.setText(picked)

    def on_train_browse_image_dir(self) -> None:
        self._browse_directory_into(self.view.train_image_dir_edit, "Select image directory")

    def on_train_browse_mask_dir(self) -> None:
        self._browse_directory_into(self.view.train_mask_dir_edit, "Select mask directory")

    def on_train_browse_output_dir(self) -> None:
        self._browse_directory_into(self.view.train_output_dir_edit, "Select training output directory")

    def on_train_browse_sweep_json(self) -> None:
        self._browse_file_into(self.view.train_sweep_edit, "Select sweep JSON", "JSON Files (*.json);;All Files (*)")

    def on_train_browse_sam_checkpoint(self) -> None:
        self._browse_file_into(
            self.view.train_sam_ckpt_edit,
            "Select SAM checkpoint",
            "Model Files (*.pth *.pt *.bin *.safetensors);;All Files (*)",
        )

    def on_eval_browse_gt_roi_path(self) -> None:
        current = self.view.eval_gt_roi_edit.text().strip()
        start_path = Path(current).expanduser() if current else Path.cwd()
        if start_path.exists() and start_path.is_file():
            start_dir = str(start_path.parent)
        else:
            start_dir = str(start_path if start_path.exists() else Path.cwd())
        id_glob = " ".join(f"*{ext}" for ext in ID_MAP_FILE_EXTS)
        picked_file, _ = QFileDialog.getOpenFileName(
            self.view,
            "Select GT ROI / ID map file",
            start_dir,
            f"GT Files (*.roi *.zip {id_glob});;ROI/ZIP Files (*.roi *.zip);;ID Map Images ({id_glob});;All Files (*)",
        )
        if picked_file:
            self.view.eval_gt_roi_edit.setText(picked_file)
            return
        picked_dir = QFileDialog.getExistingDirectory(
            self.view,
            "Select GT ROI directory",
            start_dir,
        )
        if picked_dir:
            self.view.eval_gt_roi_edit.setText(picked_dir)

    def on_eval_gt_roi_path_changed(self, *_args) -> None:
        self._eval_gt_source_cache.clear()
        self._eval_gt_instances_cache.clear()
        self._eval_current_result = {}
        self._eval_all_result = {}
        self._update_eval_plot_preview()
        self.view.set_eval_status_text("Idle")
        self._update_gt_overlay_availability()
        if self._is_segmentation_workspace_active():
            self._mark_overlay_dirty()
            self._refresh()

    def on_track_browse_image(self) -> None:
        self._browse_file_into(
            self.view.track_image_edit,
            "Select track image or video",
            "Media Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.mp4 *.avi *.mov *.mkv);;All Files (*)",
        )

    def _require_text(self, text: str, field_name: str) -> str:
        value = text.strip()
        if not value:
            raise ValueError(f"{field_name} is required.")
        return value

    def _parse_float_text(self, text: str, field_name: str) -> float:
        try:
            return float(text.strip())
        except Exception as exc:
            raise ValueError(f"Invalid {field_name}: {text}") from exc

    def _reset_train_monitor_ui(self, clear_graph: bool = False) -> None:
        self._train_monitor.reset(expected_epochs=int(self.view.train_epochs_spin.value()))
        self.view.set_train_status_text("Idle", level="idle")
        self.view.set_train_progress(0, self._train_monitor.expected_epochs)
        self.view.set_train_run_dir_text("-")
        self.view.set_train_metrics_summary_text("train: -, val: -, test: -")
        if clear_graph:
            self.view.set_train_graph_pixmap(None)

    def _sync_train_monitor_status(self) -> None:
        run_dir = self._train_monitor.run_dir
        self.view.set_train_run_dir_text(str(run_dir) if run_dir is not None else "-")
        self.view.set_train_progress(self._train_monitor.last_epoch, self._train_monitor.expected_epochs)
        if self._train_monitor.last_epoch > 0 or self._train_monitor.expected_epochs > 0:
            self.view.set_train_status_text(
                f"Running (epoch {self._train_monitor.last_epoch}/{max(1, self._train_monitor.expected_epochs)})",
                level="running",
            )

    def _update_train_monitor_from_metrics(self, force: bool = False) -> None:
        payload = self._train_monitor.build_metrics_payload(force=force)
        if payload is None:
            return
        self.view.set_train_metrics_summary_text(payload.summary_text)

        if not (payload.train_epochs or payload.val_epochs or payload.test_epochs):
            self.view.set_train_graph_pixmap(None)
            return

        target_w = self.view.train_graph_label.width() if hasattr(self.view, "train_graph_label") else 900
        target_h = self.view.train_graph_label.height() if hasattr(self.view, "train_graph_label") else 300
        rgb = render_train_metrics_panel(
            train_epochs=payload.train_epochs,
            train_loss=payload.train_loss,
            train_iou=payload.train_iou,
            train_dice=payload.train_dice,
            val_epochs=payload.val_epochs,
            val_loss=payload.val_loss,
            val_iou=payload.val_iou,
            val_dice=payload.val_dice,
            test_epochs=payload.test_epochs,
            test_loss=payload.test_loss,
            test_iou=payload.test_iou,
            test_dice=payload.test_dice,
            width=max(720, int(target_w)),
            height=max(240, int(target_h)),
        )
        self.view.set_train_graph_pixmap(rgb_to_qpixmap(rgb))

    def _build_train_command(self) -> List[str]:
        image_dir = self._require_text(self.view.train_image_dir_edit.text(), "Train image dir")
        mask_dir = self._require_text(self.view.train_mask_dir_edit.text(), "Train mask dir")
        output_dir = self._require_text(self.view.train_output_dir_edit.text(), "Train output dir")
        backend = self.view.train_backend_combo.currentText().strip().lower()
        if backend not in ("hf", "meta"):
            backend = "hf"

        args: List[str] = [
            "--image-dir",
            image_dir,
            "--mask-dir",
            mask_dir,
            "--output-dir",
            output_dir,
            "--backend",
            backend,
            "--epochs",
            str(int(self.view.train_epochs_spin.value())),
            "--batch-size",
            str(int(self.view.train_batch_spin.value())),
            "--lr",
            str(self._parse_float_text(self.view.train_lr_edit.text(), "LR")),
            "--weight-decay",
            str(self._parse_float_text(self.view.train_weight_decay_edit.text(), "weight decay")),
            "--lora-rank",
            str(int(self.view.train_rank_spin.value())),
            "--lora-alpha",
            str(self._parse_float_text(self.view.train_alpha_edit.text(), "LoRA alpha")),
            "--lora-dropout",
            str(self._parse_float_text(self.view.train_dropout_edit.text(), "LoRA dropout")),
            "--sample-mode",
            self.view.train_sample_mode_combo.currentText().strip(),
            "--freeze-config",
            self.view.train_freeze_combo.currentText().strip(),
            "--eval-every",
            str(int(self.view.train_eval_every_spin.value())),
        ]
        if backend == "meta":
            ckpt = self.view.train_sam_ckpt_edit.text().strip()
            if not ckpt:
                raise ValueError("SAM checkpoint is required when backend=meta.")
            args.extend(["--sam-checkpoint", ckpt])
            args.extend(["--sam-model-type", self.view.train_sam_model_combo.currentText().strip()])
        else:
            args.extend(["--hf-model-id", self.view.train_hf_model_edit.text().strip() or "facebook/sam-vit-base"])
            args.extend(["--hf-input-size", str(int(self.view.train_hf_input_size_spin.value()))])

        targets_raw = self.view.train_lora_targets_edit.text().replace(",", " ").strip()
        targets = [tok for tok in targets_raw.split() if tok]
        if targets:
            args.extend(["--lora-targets", *targets])

        sweep = self.view.train_sweep_edit.text().strip()
        if sweep:
            args.extend(["--sweep-config", sweep])

        extra_raw = self.view.train_extra_args_edit.text().strip()
        if extra_raw:
            args.extend(shlex.split(extra_raw))

        return args

    def _on_train_started(self) -> None:
        self.view.set_train_status_text("Running", level="running")
        self.view.append_train_log("[INFO] Train process started.")

    def _on_train_ready_read(self) -> None:
        if self._train_process is None:
            return
        data = bytes(self._train_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            self.view.append_train_log(data)
            self._train_monitor.consume_output(data, self._project_root)
            self._sync_train_monitor_status()
            self._update_train_monitor_from_metrics(force=False)

    def _on_train_finished(self, exit_code: int, _status: QtCore.QProcess.ExitStatus) -> None:
        self.view.set_train_running(False)
        self._train_monitor.flush_pending_output(self._project_root)
        self._sync_train_monitor_status()
        self._update_train_monitor_from_metrics(force=True)
        if exit_code == 0:
            self.view.append_train_log("[INFO] Train finished.")
            if self._train_monitor.expected_epochs > 0:
                self.view.set_train_progress(self._train_monitor.expected_epochs, self._train_monitor.expected_epochs)
            self.view.set_train_status_text("Completed", level="done")
        else:
            self.view.append_train_log(f"[WARN] Train exited with code {exit_code}.")
            self.view.set_train_status_text(f"Failed (code {exit_code})", level="error")
        if self._train_process is not None:
            self._train_process.deleteLater()
            self._train_process = None

    def _on_train_error(self, _err: QtCore.QProcess.ProcessError) -> None:
        if self._train_process is None:
            return
        self.view.set_train_running(False)
        self.view.append_train_log(f"[ERROR] Train process error: {self._train_process.errorString()}")
        self._train_monitor.flush_pending_output(self._project_root)
        self._sync_train_monitor_status()
        self.view.set_train_status_text("Process error", level="error")
        self._update_train_monitor_from_metrics(force=True)
        self._train_process.deleteLater()
        self._train_process = None

    def on_train_run(self) -> None:
        if getattr(sys, "frozen", False):
            QMessageBox.warning(
                self.view,
                "Train Unavailable",
                "Train workspace is not available in frozen app builds. "
                "Run from source Python environment for training.",
            )
            return
        if self._train_process is not None and self._train_process.state() != QtCore.QProcess.NotRunning:
            QMessageBox.information(self.view, "Train Running", "Training is already running.")
            return
        try:
            args = self._build_train_command()
        except Exception as exc:
            QMessageBox.warning(self.view, "Train Config Error", str(exc))
            return
        entry_args = ["-m", "microseg.train.cli", *args]
        cmd = [sys.executable, *entry_args]
        self.view.set_train_command_text(f"Command: {' '.join(shlex.quote(c) for c in cmd)}")
        self._reset_train_monitor_ui(clear_graph=True)
        self.view.set_train_status_text("Launching...", level="running")
        self.view.clear_train_log()
        self.view.append_train_log("[INFO] Launching train command...")
        proc = QtCore.QProcess(self.view)
        proc.setWorkingDirectory(str(self._project_root))
        proc.setProgram(sys.executable)
        proc.setArguments(entry_args)
        proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        proc.started.connect(self._on_train_started)
        proc.readyReadStandardOutput.connect(self._on_train_ready_read)
        proc.finished.connect(self._on_train_finished)
        proc.errorOccurred.connect(self._on_train_error)
        self._train_process = proc
        self.view.set_train_running(True)
        proc.start()

    def _force_kill_train_if_running(self) -> None:
        if self._train_process is None:
            return
        if self._train_process.state() != QtCore.QProcess.NotRunning:
            self._train_process.kill()

    def on_train_stop(self) -> None:
        if self._train_process is None or self._train_process.state() == QtCore.QProcess.NotRunning:
            return
        self.view.append_train_log("[INFO] Stopping train process...")
        self.view.set_train_status_text("Stopping...", level="running")
        self._train_process.terminate()
        QtCore.QTimer.singleShot(2500, self._force_kill_train_if_running)

    def on_eval_run_current(self) -> None:
        self._run_eval(scope="current")

    def on_eval_run_all(self) -> None:
        self._run_eval(scope="all")

    def on_eval_run(self) -> None:
        # Legacy entry point; keep behavior deterministic.
        self.on_eval_run_current()

    def _run_eval(self, scope: str) -> None:
        scope_key = "all" if (scope or "").strip().lower() == "all" else "current"
        roi_text = self.view.eval_gt_roi_edit.text().strip()
        if not roi_text:
            QMessageBox.warning(self.view, "Evaluation Config Error", "GT ROI path is required.")
            return
        roi_root = Path(roi_text)
        if not roi_root.exists():
            QMessageBox.warning(self.view, "Evaluation Config Error", f"GT ROI path not found:\n{roi_root}")
            return

        self.view.set_eval_scope(scope_key)
        self.view.set_eval_running(True)
        self.view.set_eval_status_text(f"Running evaluation ({scope_key})...")
        self._eval_gt_source_cache.clear()
        self._eval_gt_instances_cache.clear()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            if scope_key == "all":
                all_images = self._run_eval_scope(current_only=False)
                self._eval_all_result = all_images
                self._apply_eval_scope_result("all", all_images)
            else:
                current = self._run_eval_scope(current_only=True)
                self._eval_current_result = current
                self._apply_eval_scope_result("current", current)
            self.view.set_eval_status_text(f"Done ({scope_key}, {self._now_text()})")
        except Exception as exc:
            self.view.set_eval_status_text(f"Failed ({self._now_text()})")
            QMessageBox.warning(self.view, "Evaluation Error", str(exc))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.view.set_eval_running(False)
            self._update_gt_overlay_availability()
            self._mark_overlay_dirty()
            self._refresh()

    def _run_eval_scope(self, current_only: bool) -> Dict[str, Any]:
        indices: Sequence[int]
        if current_only:
            indices = [self.current_image_idx]
        else:
            indices = list(range(len(self.image_sessions)))

        # Allow shared GT roots for batch evaluation across multiple images.
        allow_shared_for_all = True
        return run_eval_scope(
            indices=indices,
            image_sessions=self.image_sessions,
            pred_min_area=self._eval_pred_min_area,
            current_only=current_only,
            allow_shared_for_all=allow_shared_for_all,
            get_gt_instances=self._get_eval_gt_instances,
            match_iou=self._eval_match_iou,
            boundary_ratio=self._eval_boundary_ratio,
        )

    def _eval_scope_plot_pixmap(self, scope: str, iou_vals: Sequence[float], dice_vals: Sequence[float], bf1_vals: Sequence[float]) -> QtGui.QPixmap:
        label = self.view.eval_current_plot_label if scope == "current" else self.view.eval_all_plot_label
        target_w = label.width() if label.width() > 0 else 920
        target_h = label.height() if label.height() > 0 else 520
        rgb = render_eval_metrics_panel(
            iou_vals,
            dice_vals,
            bf1_vals,
            width=max(640, int(target_w)),
            height=max(280, int(target_h)),
        )
        return rgb_to_qpixmap(rgb)

    def _apply_eval_scope_result(self, scope: str, result: Dict[str, Any]) -> None:
        iou_vals = result.get("iou", []) or []
        dice_vals = result.get("dice", []) or []
        bf1_vals = result.get("bf1", []) or []
        if not iou_vals and not dice_vals and not bf1_vals:
            placeholder = (
                "No evaluable data in this scope.\n"
                "Check GT ROI mapping and whether masks are loaded."
            )
            self.view.set_eval_scope_plot(scope, None, placeholder)
            return
        pix = self._eval_scope_plot_pixmap(scope, iou_vals, dice_vals, bf1_vals)
        self.view.set_eval_scope_plot(scope, pix, "")

    def _update_eval_plot_preview(self, run_dir: Optional[Path] = None) -> None:
        _ = run_dir
        if self._eval_current_result:
            self._apply_eval_scope_result("current", self._eval_current_result)
        else:
            self.view.set_eval_scope_plot("current", None, "Run Compare to view current-image metrics")
        if self._eval_all_result:
            self._apply_eval_scope_result("all", self._eval_all_result)
        else:
            self.view.set_eval_scope_plot("all", None, "Run Compare to view all-images metrics")

    def on_review_row_clicked(self, row: int, _col: int) -> None:
        idx = int(row)
        if 0 <= idx < len(self._review_row_to_mask_idx):
            idx = int(self._review_row_to_mask_idx[idx])
        if 0 <= idx < len(self.set_masks):
            self._set_single_selection(idx)
            self.current = None
            self.current_is_raw_display = False
            self._mark_overlay_dirty()
            self._refresh()

    # Core calculations ----------------------------------------------------
    def _fractal_slides_effective(self) -> int:
        return max(1, self.fractal_slides_setting)

    def _use_lora_adapter(self) -> bool:
        return self.mode == "lora" and self.lora_mode_available

    def _apply_no_overlap(self, mask: np.ndarray) -> np.ndarray:
        if not self.set_masks:
            return mask
        union = np.zeros_like(mask, dtype=np.uint8)
        for rec in self.set_masks:
            union |= rec.mask.astype(np.uint8)
        return np.where(union > 0, 0, mask).astype(np.uint8)

    def _generate_from_prompts(self) -> Optional[MaskEntry]:
        if not self.prompt_points and self.prompt_box is None:
            return None
        if self.prompt_box is None and not any(lbl == 1 for _, _, lbl in self.prompt_points):
            return None
        pts = [(p[0], p[1]) for p in self.prompt_points]
        lbls = [p[2] for p in self.prompt_points]
        box = self.prompt_box
        masks, scores = self.predictor.predict(
            pts,
            labels=lbls,
            box=box,
            multimask_output=False,
            use_adapter=self._use_lora_adapter(),
        )
        from microseg.compute import choose_best_mask  # avoid circular import

        best_mask, best_score = choose_best_mask(masks, scores)
        raw = binary_fill_holes(best_mask > 0).astype(np.uint8)
        cleaned = clean_mask(raw, min_area=20, largest_only=True)
        cleaned = binary_fill_holes(cleaned > 0).astype(np.uint8)
        candidate = cleaned
        if not self.use_original_mode and self.set_masks:
            trimmed = self._apply_no_overlap(cleaned)
            if trimmed.sum() > 0:
                candidate = trimmed
        if self.use_original_mode:
            candidate = raw
        prompt_data = self._build_point_box_prompt_payload()
        return MaskEntry(candidate, raw, best_score, prompt_data=prompt_data)

    def _generate_from_polygon(self) -> Optional[MaskEntry]:
        if len(self.polygon_points) < 3:
            return None
        h, w = self.image_bgr.shape[:2]
        pts = np.array(
            [[int(round(p[0])), int(round(p[1]))] for p in self.polygon_points],
            dtype=np.int32,
        )
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        raw = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(raw, [pts], 1)
        cleaned = raw.copy()
        candidate = cleaned
        if not self.use_original_mode and self.set_masks:
            trimmed = self._apply_no_overlap(cleaned)
            if trimmed.sum() > 0:
                candidate = trimmed
        if self.use_original_mode:
            candidate = raw
        prompt_data = self._build_polygon_prompt_payload()
        return MaskEntry(candidate, raw, None, prompt_data=prompt_data)

    def _update_current_preview(self, entry: Optional[MaskEntry]) -> None:
        self.current = entry
        self.current_is_raw_display = self.use_original_mode
        self._clear_selection()
        self._mark_overlay_dirty()
        self._refresh()

    def _update_prompt_preview(self) -> None:
        self._update_current_preview(self._generate_from_prompts())

    def _update_polygon_preview(self) -> None:
        self._update_current_preview(self._generate_from_polygon())

    def _run_calc_and_summary(self) -> None:
        masks = [m.mask for m in self.set_masks]
        self.pair_results = compute_two_nearest(
            masks,
            self.scale_nm_per_px,
            self.max_distance_nm,
            include_zero=self.include_zero_distance,
        )
        self.summary = summarize(self.pair_results)
        self.size_summary = summarize_sizes(masks, self.scale_nm_per_px, fractal_slides=self._fractal_slides_effective())
        self._compute_clusters()
        self.summary_stale = False
        self._mark_analysis_dirty()

    def _reset_calc_outputs(self) -> None:
        self.pair_results = []
        self.summary = {}
        self.size_summary = summarize_sizes(
            [m.mask for m in self.set_masks],
            self.scale_nm_per_px,
            fractal_slides=self._fractal_slides_effective(),
        ) if self.set_masks else {}
        self.cluster_labels = []
        self.cluster_stats = {}
        self.summary_stale = False
        self._mark_analysis_dirty()

    def _request_calc_async(self) -> None:
        # Kept as a compatibility entry point; calculation is intentionally synchronous.
        if len(self.set_masks) < 2:
            self._reset_calc_outputs()
            return
        self.view.set_calc_running(True, "Calculating...")
        status_text = f"Calc done ({self._now_text()})"
        try:
            self._run_calc_and_summary()
        except Exception as exc:
            status_text = "Calc failed"
            QMessageBox.warning(self.view, "Calc Error", f"Calc failed:\n{exc}")
        finally:
            self.view.set_calc_running(False, status_text)
        self._refresh()

    def _auto_calc_if_ready(self) -> None:
        if len(self.set_masks) == 0:
            self._clear_analysis_state()
            self.summary_stale = False
            return
        if not self.realtime_calc_enabled:
            # Keep the previous calc outputs visible until user explicitly presses Calc.
            self.summary_stale = True
            return
        self._request_calc_async()

    def _sync_cluster_controls(self) -> None:
        unit = self._normalize_length_unit(self.display_length_unit)
        disp_val = self._convert_length_from_nm(self.cluster_threshold_nm, unit)
        if disp_val is None:
            cluster_text = ""
        else:
            cluster_text = f"{float(disp_val):.6g}"
        self.view.set_cluster_label_text(f"Cluster th (N1, {unit}):")
        self.view.set_cluster_text(cluster_text)
        if not self.cluster_stats:
            self.view.set_cluster_stats_summary(None, None, None, None)
            return
        self.view.set_cluster_stats_summary(
            self.cluster_stats.get("count"),
            self.cluster_stats.get("mean"),
            self.cluster_stats.get("min"),
            self.cluster_stats.get("max"),
        )

    def _compute_clusters(self) -> None:
        self.cluster_labels = []
        self.cluster_stats = {}
        if not self.pair_results:
            return
        n = len(self.set_masks)
        if n == 0:
            return
        if len(self.pair_results) != n:
            return
        parent = list(range(n))
        centroids_px = [mask_centroid(rec.mask) for rec in self.set_masks]

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        kernel = np.ones((3, 3), np.uint8)
        dilated = [cv2.dilate(rec.mask.astype(np.uint8), kernel, iterations=1) for rec in self.set_masks]
        for i in range(n):
            mi = self.set_masks[i].mask.astype(np.uint8)
            for j in range(i + 1, n):
                mj = self.set_masks[j].mask.astype(np.uint8)
                touching = (
                    np.logical_and(dilated[i] > 0, mj > 0).any()
                    or np.logical_and(dilated[j] > 0, mi > 0).any()
                )
                if touching:
                    union(i, j)

        thr = float(self.cluster_threshold_nm)
        k = max(1, int(self.cluster_k))
        for i, res in enumerate(self.pair_results):
            near = res.get("nearest_all") or res.get("nearest", [])
            if not near:
                continue
            for item in near[:k]:
                j = item.get("index")
                dist = item.get("distance_nm")
                if j is None or dist is None:
                    continue
                if (thr <= 0 or dist <= thr) and 0 <= j < n:
                    union(i, int(j))

        def cluster_centroid(root: int) -> Tuple[float, float]:
            members = [idx for idx in range(n) if find(idx) == root]
            if not members:
                return (0.0, 0.0)
            xs = [centroids_px[m][0] for m in members]
            ys = [centroids_px[m][1] for m in members]
            return (float(np.mean(xs)), float(np.mean(ys)))

        roots = {find(i) for i in range(n)}
        root_list = list(roots)
        centroids_root = {r: cluster_centroid(r) for r in root_list}
        thr_px = thr / float(self.scale_nm_per_px) if thr > 0 else None
        for i in range(len(root_list)):
            for j in range(i + 1, len(root_list)):
                ri = root_list[i]
                rj = root_list[j]
                ci = centroids_root[ri]
                cj = centroids_root[rj]
                dist_px = np.hypot(ci[0] - cj[0], ci[1] - cj[1])
                if thr_px is None or dist_px <= thr_px:
                    union(ri, rj)

        labels = []
        mapping: Dict[int, int] = {}
        for i in range(n):
            root = find(i)
            if root not in mapping:
                mapping[root] = len(mapping)
            labels.append(mapping[root])
        self.cluster_labels = labels
        counts = Counter(labels)
        sizes = list(counts.values())

        def _stats(arr: List[int]) -> Dict[str, Optional[float]]:
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

        self.cluster_stats = _stats(sizes)
        self.cluster_stats["sizes"] = sizes
        self.cluster_stats["threshold_nm"] = thr
        self.cluster_stats["k_used"] = k

    # Palette ------------------------------------------------------------------
    def _cluster_palette(self, k: int) -> Dict[int, Tuple[int, int, int]]:
        k = max(1, int(k))
        colors: Dict[int, Tuple[int, int, int]] = {}
        for i in range(k):
            h = i / max(1, k)
            r, g, b = colorsys.hsv_to_rgb(h, 0.65, 1.0)
            colors[i] = (int(b * 255), int(g * 255), int(r * 255))
        return colors

    # Distribution bins --------------------------------------------------------
    @staticmethod
    def _normalize_distribution_metric(metric: Any) -> str:
        token = str(metric or "none").strip().lower()
        aliases = {
            "none": "none",
            "off": "none",
            "vesd": "vesd",
            "ecd": "ecd",
            "aspect": "aspect",
            "aspect_ratio": "aspect",
            "aspect(feret)": "aspect",
        }
        return aliases.get(token, "none")

    @staticmethod
    def _normalize_nearest_hist_metric(metric: Any) -> str:
        token = str(metric or "nearest1").strip().lower()
        if token in {"nearest2", "n2", "second", "2"}:
            return "nearest2"
        return "nearest1"

    @staticmethod
    def _normalize_size_hist_metric(metric: Any) -> str:
        token = str(metric or "ecd").strip().lower()
        if token == "vesd":
            return "vesd"
        return "ecd"

    @staticmethod
    def _normalize_area_hist_metric(metric: Any) -> str:
        token = str(metric or "area").strip().lower()
        if token in {"bbox", "area_bbox", "bbox_area"}:
            return "bbox"
        if token in {"vesd", "area_vesd"}:
            return "vesd"
        return "area"

    @staticmethod
    def _normalize_aspect_hist_metric(metric: Any) -> str:
        token = str(metric or "feret").strip().lower()
        if token in {"ellipse", "ell"}:
            return "ellipse"
        return "feret"

    @staticmethod
    def _normalize_main_graph_metric(metric: Any) -> str:
        token = str(metric or "ecd").strip().lower()
        aliases = {
            "nearest1": "nearest1",
            "n1": "nearest1",
            "first": "nearest1",
            "nearest2": "nearest2",
            "n2": "nearest2",
            "second": "nearest2",
            "ecd": "ecd",
            "vesd": "vesd",
            "area": "area",
            "bbox": "bbox",
            "area_bbox": "bbox",
            "bbox_area": "bbox",
            "area_vesd": "area_vesd",
            "vesd_area": "area_vesd",
            "aspect_feret": "aspect_feret",
            "aspect": "aspect_feret",
            "feret": "aspect_feret",
            "aspect_ellipse": "aspect_ellipse",
            "ellipse": "aspect_ellipse",
            "fractal": "fractal",
        }
        return aliases.get(token, "ecd")

    @classmethod
    def _distribution_metric_is_length(cls, metric: str) -> bool:
        return cls._normalize_distribution_metric(metric) in {"vesd", "ecd"}

    def _format_distribution_edges_for_ui(self, metric: str, edges: Sequence[float]) -> str:
        if not edges:
            return ""
        values: List[float] = []
        for v in edges:
            try:
                fv = float(v)
            except Exception:
                continue
            if not np.isfinite(fv):
                continue
            values.append(fv)
        if not values:
            return ""
        if self._distribution_metric_is_length(metric):
            converted = [self._convert_length_from_nm(v, self.display_length_unit) for v in values]
            values = [float(v) for v in converted if v is not None and np.isfinite(float(v))]
        return ", ".join(f"{float(v):.1f}" for v in values)

    def _distribution_values_for_metric(
        self,
        metric: str,
        analysis_payload: Optional[Dict[str, Any]] = None,
    ) -> List[float]:
        m = self._normalize_distribution_metric(metric)
        if m == "none":
            return []
        payload = analysis_payload if isinstance(analysis_payload, dict) else self._analysis_scope_payload()
        size_data = payload.get("size_summary", {}) if isinstance(payload.get("size_summary"), dict) else {}
        if m == "vesd":
            raw = size_data.get("hist_vesd", [])
        elif m == "ecd":
            raw = size_data.get("hist_ecd", [])
        elif m == "aspect":
            if self._normalize_aspect_hist_metric(self.aspect_hist_metric) == "ellipse":
                major_raw = size_data.get("hist_ellipse_major_axis", [])
                minor_raw = size_data.get("hist_ellipse_minor_axis", [])
                out_aspect: List[float] = []
                if isinstance(major_raw, list) and isinstance(minor_raw, list):
                    for ma, mi in zip(major_raw, minor_raw):
                        try:
                            ma_f = float(ma)
                            mi_f = float(mi)
                        except Exception:
                            continue
                        if np.isfinite(ma_f) and np.isfinite(mi_f) and mi_f > 0.0:
                            out_aspect.append(float(ma_f / mi_f))
                return out_aspect
            raw = size_data.get("hist_aspect_ratio", [])
        else:
            raw = []
        out: List[float] = []
        if isinstance(raw, list):
            for v in raw:
                try:
                    fv = float(v)
                except Exception:
                    continue
                if np.isfinite(fv):
                    out.append(fv)
        return out

    def _distribution_value_to_display(self, metric: str, value_internal: float) -> Optional[float]:
        m = self._normalize_distribution_metric(metric)
        try:
            v = float(value_internal)
        except Exception:
            return None
        if not np.isfinite(v):
            return None
        if self._distribution_metric_is_length(m):
            return self._convert_length_from_nm(v, self.display_length_unit)
        return v

    def _distribution_value_from_display(self, metric: str, value_display: float) -> Optional[float]:
        m = self._normalize_distribution_metric(metric)
        try:
            v = float(value_display)
        except Exception:
            return None
        if not np.isfinite(v):
            return None
        if self._distribution_metric_is_length(m):
            return self._length_to_nm(v, self.display_length_unit)
        return v

    @staticmethod
    def _default_distribution_edges(values: Sequence[float], bin_count: int = 3) -> List[float]:
        vals = [float(v) for v in values if np.isfinite(float(v))]
        if not vals:
            return []
        arr = np.asarray(vals, dtype=np.float32)
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        if not np.isfinite(lo) or not np.isfinite(hi):
            return []
        bins = max(1, int(bin_count))
        if hi <= lo:
            hi = lo + max(abs(lo) * 1e-3, 1e-6)
        try:
            q = np.linspace(0.0, 1.0, bins + 1)
            edges_arr = np.quantile(arr, q)
            edges = [float(v) for v in edges_arr.tolist()]
        except Exception:
            edges = []
        uniq = sorted({float(v) for v in edges if np.isfinite(float(v))})
        if len(uniq) < 2:
            uniq = [lo, hi]
        if len(uniq) < bins + 1:
            uniq = [float(v) for v in np.linspace(lo, hi, bins + 1).tolist()]
        return uniq

    @staticmethod
    def _sanitize_distribution_edges(
        edges: Sequence[float],
        fallback_min: float,
        fallback_max: float,
    ) -> List[float]:
        vals: List[float] = []
        for v in edges:
            try:
                fv = float(v)
            except Exception:
                continue
            if np.isfinite(fv):
                vals.append(fv)
        vals = sorted(vals)
        if len(vals) < 2:
            lo = float(fallback_min)
            hi = float(fallback_max)
            if hi <= lo:
                hi = lo + max(abs(lo) * 1e-3, 1e-6)
            vals = [lo, hi]
        lo = min(float(fallback_min), float(vals[0]))
        hi = max(float(fallback_max), float(vals[-1]))
        if hi <= lo:
            hi = lo + max(abs(lo) * 1e-3, 1e-6)
        span = max(hi - lo, 1e-12)
        eps = span * 1e-6
        interior: List[float] = []
        for v in vals[1:-1]:
            if v <= lo + eps or v >= hi - eps:
                continue
            if interior and abs(v - interior[-1]) <= eps:
                continue
            interior.append(v)
        return [lo] + interior + [hi]

    def _sync_distribution_slider_ui(self, analysis_payload: Optional[Dict[str, Any]] = None) -> None:
        metric = self._normalize_distribution_metric(self.distribution_metric)
        if metric == "none":
            self._distribution_slider_min_internal = None
            self._distribution_slider_max_internal = None
            self.view.set_distribution_bins_enabled(False)
            self.view.set_distribution_slider_state(0.0, 1.0, [], enabled=False)
            if not self.view.get_distribution_edges_text().strip():
                self.view.set_distribution_status("")
            return

        values = self._distribution_values_for_metric(metric, analysis_payload=analysis_payload)
        if not values:
            self._distribution_slider_min_internal = None
            self._distribution_slider_max_internal = None
            self.distribution_edges = []
            self.view.set_distribution_bins_enabled(False)
            self.view.set_distribution_slider_state(0.0, 1.0, [], enabled=False)
            self.view.set_distribution_status("No data", error=False)
            return

        self.view.set_distribution_bins_enabled(True)
        data_min = float(min(values))
        data_max = float(max(values))
        target_bins = max(2, int(self.view.get_distribution_bins_count()))
        if len(self.distribution_edges) < 2:
            self.distribution_edges = self._default_distribution_edges(values, bin_count=target_bins)
        edges = self._sanitize_distribution_edges(self.distribution_edges, data_min, data_max)
        self.distribution_edges = edges
        self.view.set_distribution_bins_count(max(2, len(edges) - 1))
        self._distribution_slider_min_internal = float(edges[0])
        self._distribution_slider_max_internal = float(edges[-1])

        min_disp = self._distribution_value_to_display(metric, self._distribution_slider_min_internal)
        max_disp = self._distribution_value_to_display(metric, self._distribution_slider_max_internal)
        if min_disp is None or max_disp is None:
            self.view.set_distribution_slider_state(0.0, 1.0, [], enabled=False)
            self.view.set_distribution_status("Invalid unit", error=True)
            return
        interior_disp: List[float] = []
        for v in edges[1:-1]:
            dv = self._distribution_value_to_display(metric, float(v))
            if dv is None or not np.isfinite(float(dv)):
                continue
            interior_disp.append(float(dv))
        self.view.set_distribution_slider_state(float(min_disp), float(max_disp), interior_disp, enabled=True)
        self.view.set_distribution_edges_text(self._format_distribution_edges_for_ui(metric, edges))
        self.view.set_distribution_status(f"{metric.upper()} bins: {len(edges) - 1}", error=False)

    def _apply_distribution_bin_count(self, bin_count: int) -> None:
        metric = self._normalize_distribution_metric(self.distribution_metric)
        if metric == "none":
            return
        values = self._distribution_values_for_metric(metric)
        if not values:
            self.view.set_distribution_status("No data", error=False)
            return
        target = max(2, min(20, int(bin_count)))
        self.view.set_distribution_bins_count(target)
        edges = self._default_distribution_edges(values, bin_count=target)
        self.distribution_edges = [float(v) for v in edges]
        self._sync_distribution_slider_ui()
        self._mark_overlay_dirty()
        self._mark_analysis_dirty()
        self._refresh()

    def _parse_distribution_edges_from_text(self, text: str, metric: str) -> Tuple[List[float], Optional[str]]:
        raw = str(text or "")
        if not raw.strip():
            return [], "Enter at least two edges"
        tokens = [t for t in re.split(r"[,\s;、]+", raw.strip()) if t]
        values: List[float] = []
        for token in tokens:
            token_norm = token.strip().replace("−", "-")
            try:
                val = float(token_norm)
            except Exception:
                return [], f"Invalid value: {token}"
            if not np.isfinite(val):
                return [], f"Invalid value: {token}"
            if self._distribution_metric_is_length(metric):
                val_nm = self._length_to_nm(val, self.display_length_unit)
                if val_nm is None:
                    return [], "Invalid unit conversion"
                values.append(float(val_nm))
            else:
                values.append(float(val))
        uniq = sorted({float(v) for v in values})
        if len(uniq) < 2:
            return [], "Need at least two unique edges"
        return uniq, None

    @staticmethod
    def _distribution_bin_index(value: Optional[float], edges: Sequence[float]) -> Optional[int]:
        if value is None or len(edges) < 2:
            return None
        fv = float(value)
        if not np.isfinite(fv):
            return None
        arr = np.asarray(edges, dtype=np.float32)
        if arr.size < 2:
            return None
        if fv < float(arr[0]) or fv > float(arr[-1]):
            return None
        idx = int(np.searchsorted(arr, fv, side="right") - 1)
        if idx == int(arr.size - 1) and np.isclose(fv, float(arr[-1])):
            idx -= 1
        if idx < 0 or idx >= int(arr.size - 1):
            return None
        return idx

    @staticmethod
    def _distribution_palette_rgb(count: int) -> List[Tuple[int, int, int]]:
        n = max(0, int(count))
        if n <= 0:
            return []
        base = [
            (31, 119, 180),
            (255, 127, 14),
            (44, 160, 44),
            (214, 39, 40),
            (148, 103, 189),
            (140, 86, 75),
            (227, 119, 194),
            (127, 127, 127),
            (188, 189, 34),
            (23, 190, 207),
        ]
        if n <= len(base):
            return base[:n]
        out: List[Tuple[int, int, int]] = []
        for i in range(n):
            h = i / max(1, n)
            r, g, b = colorsys.hsv_to_rgb(h, 0.72, 0.95)
            out.append((int(r * 255), int(g * 255), int(b * 255)))
        return out

    @classmethod
    def _distribution_palette_hex(cls, count: int) -> List[str]:
        return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in cls._distribution_palette_rgb(count)]

    @classmethod
    def _distribution_palette_bgr(cls, count: int) -> List[Tuple[int, int, int]]:
        return [(int(b), int(g), int(r)) for r, g, b in cls._distribution_palette_rgb(count)]

    def _distribution_value_from_metrics(self, metrics: Dict[str, Any], metric: str) -> Optional[float]:
        key = self._normalize_distribution_metric(metric)
        if key == "vesd":
            value = metrics.get("vesd_nm")
        elif key == "ecd":
            value = metrics.get("ecd_nm")
        elif key == "aspect":
            value = metrics.get("aspect_ratio")
        else:
            return None
        if value is None:
            return None
        try:
            fv = float(value)
        except Exception:
            return None
        if not np.isfinite(fv):
            return None
        return fv

    def _distribution_overlay_assignments(self) -> Tuple[Dict[int, int], List[Tuple[int, int, int]]]:
        metric = self._normalize_distribution_metric(self.distribution_metric)
        edges = [float(v) for v in self.distribution_edges]
        if metric == "none" or len(edges) < 2:
            return {}, []
        bin_count = len(edges) - 1
        palette = self._distribution_palette_bgr(bin_count)
        if not palette:
            return {}, []
        out: Dict[int, int] = {}
        for idx, rec in enumerate(self.set_masks):
            metrics = compute_mask_shape_metrics(rec.mask.astype(np.uint8), self.scale_nm_per_px)
            value = self._distribution_value_from_metrics(metrics, metric)
            bin_idx = self._distribution_bin_index(value, edges)
            if bin_idx is None:
                continue
            out[int(idx)] = int(bin_idx)
        return out, palette

    def _distribution_hist_style(self) -> Dict[str, Any]:
        metric = self._normalize_distribution_metric(self.distribution_metric)
        edges = [float(v) for v in self.distribution_edges]
        if metric == "none" or len(edges) < 2:
            return {"metric": "none", "edges": [], "colors": []}
        colors = self._distribution_palette_hex(len(edges) - 1)
        return {"metric": metric, "edges": edges, "colors": colors}

    # Event handlers -----------------------------------------------------------
    def _compute_view_size_for_zoom(self, zoom: float) -> Tuple[int, int]:
        h, w = self.image_bgr.shape[:2]
        z = float(np.clip(zoom, self.zoom_min, self.zoom_max))
        view_w = max(1, int(round(w / z)))
        view_h = max(1, int(round(h / z)))
        view_w = min(view_w, w)
        view_h = min(view_h, h)
        return view_w, view_h

    def _compute_view_rect(self) -> Tuple[int, int, int, int]:
        h, w = self.image_bgr.shape[:2]
        self.zoom_factor = float(np.clip(self.zoom_factor, self.zoom_min, self.zoom_max))
        view_w, view_h = self._compute_view_size_for_zoom(self.zoom_factor)
        max_x0 = max(0, w - view_w)
        max_y0 = max(0, h - view_h)
        cx = min(max(self._view_center_x, 0.0), float(w - 1))
        cy = min(max(self._view_center_y, 0.0), float(h - 1))
        x0 = int(round(cx - view_w / 2.0))
        y0 = int(round(cy - view_h / 2.0))
        x0 = min(max(x0, 0), max_x0)
        y0 = min(max(y0, 0), max_y0)
        self._view_center_x = float(x0) + float(view_w) / 2.0
        self._view_center_y = float(y0) + float(view_h) / 2.0
        return x0, y0, view_w, view_h

    def _display_to_image_xy(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        disp_w, disp_h = self._disp_size
        org_x, org_y = self._disp_origin
        if disp_w == 0 or disp_h == 0:
            return None
        lx = int(x) - int(org_x)
        ly = int(y) - int(org_y)
        if lx < 0 or ly < 0 or lx >= disp_w or ly >= disp_h:
            return None
        xx = int(round(self._view_x0 + lx * self._scale_x))
        yy = int(round(self._view_y0 + ly * self._scale_y))
        xx = min(max(xx, 0), self.image_bgr.shape[1] - 1)
        yy = min(max(yy, 0), self.image_bgr.shape[0] - 1)
        return xx, yy

    def _hit_mask_index(self, xx: int, yy: int) -> Optional[int]:
        for idx in range(len(self.set_masks) - 1, -1, -1):
            if self.set_masks[idx].mask[yy, xx] > 0:
                return idx
        return None

    def _valid_selected_indices(self) -> set[int]:
        return {idx for idx in self.selected_indices if 0 <= idx < len(self.set_masks)}

    def _clear_selection(self) -> None:
        self.selected_idx = None
        self.selected_indices.clear()

    def _set_single_selection(self, idx: int) -> None:
        self.selected_idx = int(idx)
        self.selected_indices = {int(idx)}

    def _toggle_multi_selection(self, idx: int) -> None:
        idx = int(idx)
        selected = self._valid_selected_indices()
        if idx in selected:
            selected.discard(idx)
            if self.selected_idx == idx:
                self.selected_idx = max(selected) if selected else None
        else:
            selected.add(idx)
            self.selected_idx = idx
        self.selected_indices = selected

    def _add_prompt_point(self, xx: int, yy: int, label: int) -> None:
        self.prompt_points.append((float(xx), float(yy), int(label)))
        self._last_undoable_action = "prompt_add"
        self._update_prompt_preview()

    def _add_polygon_point(self, xx: int, yy: int) -> None:
        self.polygon_points.append((float(xx), float(yy)))
        self._last_undoable_action = "polygon_add"
        self._update_polygon_preview()

    @staticmethod
    def _clone_mask_entry(entry: MaskEntry) -> MaskEntry:
        return MaskEntry(
            mask=entry.mask.copy(),
            raw=entry.raw.copy(),
            score=entry.score,
            prompt_data=copy.deepcopy(entry.prompt_data),
        )

    def _capture_undo_snapshot(self) -> Dict[str, Any]:
        return {
            "set_masks": [self._clone_mask_entry(rec) for rec in self.set_masks],
            "current": self._clone_mask_entry(self.current) if self.current is not None else None,
            "current_is_raw_display": bool(self.current_is_raw_display),
            "pair_results": copy.deepcopy(self.pair_results),
            "summary": copy.deepcopy(self.summary),
            "size_summary": copy.deepcopy(self.size_summary),
            "summary_stale": bool(self.summary_stale),
            "selected_idx": self.selected_idx,
            "selected_indices": set(int(i) for i in self.selected_indices),
            "prompt_points": list(self.prompt_points),
            "prompt_box": tuple(self.prompt_box) if self.prompt_box is not None else None,
            "polygon_points": list(self.polygon_points),
            "cluster_labels": list(self.cluster_labels),
            "cluster_stats": copy.deepcopy(self.cluster_stats),
        }

    def _push_undo_snapshot(self) -> None:
        self._undo_stack.append(self._capture_undo_snapshot())
        if len(self._undo_stack) > self._undo_limit:
            self._undo_stack = self._undo_stack[-self._undo_limit :]
        self._last_undoable_action = "snapshot"

    def _restore_undo_snapshot(self, snapshot: Dict[str, Any]) -> None:
        set_masks = snapshot.get("set_masks") or []
        self.set_masks = [self._clone_mask_entry(rec) for rec in set_masks]
        current = snapshot.get("current")
        self.current = self._clone_mask_entry(current) if isinstance(current, MaskEntry) else None
        self.current_is_raw_display = bool(snapshot.get("current_is_raw_display", False))
        self.pair_results = copy.deepcopy(snapshot.get("pair_results") or [])
        self.summary = copy.deepcopy(snapshot.get("summary") or {})
        self.size_summary = copy.deepcopy(snapshot.get("size_summary") or {})
        self.summary_stale = bool(snapshot.get("summary_stale", False))
        self.selected_idx = snapshot.get("selected_idx")
        self.selected_indices = {
            int(i) for i in (snapshot.get("selected_indices") or set())
            if 0 <= int(i) < len(self.set_masks)
        }
        if self.selected_idx is not None and 0 <= int(self.selected_idx) < len(self.set_masks):
            self.selected_idx = int(self.selected_idx)
            self.selected_indices.add(int(self.selected_idx))
        elif self.selected_indices:
            self.selected_idx = max(self.selected_indices)
        else:
            self.selected_idx = None
        self.prompt_points = list(snapshot.get("prompt_points") or [])
        box = snapshot.get("prompt_box")
        if box is None or len(box) != 4:
            self.prompt_box = None
        else:
            self.prompt_box = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        self.drag_prompt_box = None
        self.polygon_points = list(snapshot.get("polygon_points") or [])
        self.cluster_labels = [int(v) for v in (snapshot.get("cluster_labels") or [])]
        self.cluster_stats = copy.deepcopy(snapshot.get("cluster_stats") or {})
        self._last_undoable_action = ""
        self._mark_dirty_current()
        self._mark_analysis_dirty()
        self._mark_overlay_dirty()

    def _can_edit_in_preview(self) -> bool:
        if not self._is_segmentation_workspace_active():
            return False
        if self.workspace_tab == self.WORKSPACE_FILTERS and self.filter_fft_mode:
            return False
        if hasattr(self, "view") and hasattr(self.view, "is_preview_tab_active"):
            return bool(self.view.is_preview_tab_active())
        return True

    def on_click_image(self, x: int, y: int) -> None:
        if not self._can_edit_in_preview():
            return
        if self.drag_prompt_box is not None:
            self.drag_prompt_box = None
        img_xy = self._display_to_image_xy(x, y)
        if img_xy is None:
            return
        xx, yy = img_xy
        if self.scale_calibration_mode:
            if self.polygon_mode:
                self._add_polygon_point(xx, yy)
            else:
                self._add_prompt_point(xx, yy, 1)
            return
        if self.polygon_mode:
            self._add_polygon_point(xx, yy)
            return
        # Multi-point mode: left click always adds positive prompt point.
        if self.prompt_points or self.prompt_box is not None:
            self._add_prompt_point(xx, yy, 1)
            return
        # selection check: prefer latest set mask
        hit_idx = self._hit_mask_index(xx, yy)
        if hit_idx is not None:
            self._set_single_selection(hit_idx)
            self.current = None
            self.current_is_raw_display = False
            self._mark_overlay_dirty()
            self._refresh()
            return
        self._add_prompt_point(xx, yy, 1)

    def on_right_click_image(self, x: int, y: int) -> None:
        if not self._can_edit_in_preview():
            return
        if self.drag_prompt_box is not None:
            self.drag_prompt_box = None
        img_xy = self._display_to_image_xy(x, y)
        if img_xy is None:
            return
        xx, yy = img_xy
        if self.scale_calibration_mode:
            if self.polygon_mode:
                if len(self.polygon_points) >= 3:
                    self._update_polygon_preview()
            else:
                self._add_prompt_point(xx, yy, 0)
            return
        if self.polygon_mode:
            if len(self.polygon_points) >= 3:
                self._update_polygon_preview()
            return
        hit_idx = self._hit_mask_index(xx, yy)
        # When no prompt is being edited, right-click on a mask toggles multi-selection.
        if hit_idx is not None and not self.prompt_points and self.prompt_box is None:
            self._toggle_multi_selection(hit_idx)
            self.current = None
            self.current_is_raw_display = False
            self._mark_overlay_dirty()
            self._refresh()
            return
        # SAM prompt mode: right click adds negative point.
        self._add_prompt_point(xx, yy, 0)

    def on_box_dragging_image(self, x0: int, y0: int, x1: int, y1: int) -> None:
        if self._handle_sym_notch_dragging_event(x0, y0, x1, y1, finalize=False):
            return
        if not self._can_edit_in_preview():
            return
        if self.polygon_mode:
            return
        p0 = self._display_to_image_xy(x0, y0)
        p1 = self._display_to_image_xy(x1, y1)
        if p0 is None or p1 is None:
            return
        x0i, y0i = p0
        x1i, y1i = p1
        bx0 = int(min(x0i, x1i))
        by0 = int(min(y0i, y1i))
        bx1 = int(max(x0i, x1i))
        by1 = int(max(y0i, y1i))
        if (bx1 - bx0) < 1 or (by1 - by0) < 1:
            return
        next_box = (float(bx0), float(by0), float(bx1), float(by1))
        if self.drag_prompt_box == next_box:
            return
        self.drag_prompt_box = next_box
        self._mark_overlay_dirty()
        self._refresh()

    def on_box_drag_image(self, x0: int, y0: int, x1: int, y1: int) -> None:
        if self._handle_sym_notch_dragging_event(x0, y0, x1, y1, finalize=True):
            return
        if not self._can_edit_in_preview():
            return
        if self.polygon_mode:
            return
        self.drag_prompt_box = None
        p0 = self._display_to_image_xy(x0, y0)
        p1 = self._display_to_image_xy(x1, y1)
        if p0 is None or p1 is None:
            return
        x0i, y0i = p0
        x1i, y1i = p1
        bx0 = int(min(x0i, x1i))
        by0 = int(min(y0i, y1i))
        bx1 = int(max(x0i, x1i))
        by1 = int(max(y0i, y1i))
        if (bx1 - bx0) < 3 or (by1 - by0) < 3:
            self._mark_overlay_dirty()
            self._refresh()
            return
        self.prompt_box = (float(bx0), float(by0), float(bx1), float(by1))
        self._last_undoable_action = "box_set"
        self._update_prompt_preview()

    def on_wheel_image(self, delta: int, x: int, y: int) -> None:
        if delta == 0:
            return
        old_zoom = self.zoom_factor
        step = 1.15 ** (float(delta) / 120.0)
        new_zoom = float(np.clip(old_zoom * step, self.zoom_min, self.zoom_max))
        if abs(new_zoom - old_zoom) < 1e-9:
            return
        self.zoom_factor = new_zoom
        disp_w, disp_h = self._disp_size
        org_x, org_y = self._disp_origin
        if disp_w <= 0 or disp_h <= 0 or self._view_w <= 0 or self._view_h <= 0:
            self._refresh()
            return
        lx = int(x) - int(org_x)
        ly = int(y) - int(org_y)
        if lx < 0 or ly < 0 or lx >= disp_w or ly >= disp_h:
            return
        rx = min(max(float(lx) / float(max(1, disp_w)), 0.0), 1.0)
        ry = min(max(float(ly) / float(max(1, disp_h)), 0.0), 1.0)
        anchor_x = self._view_x0 + rx * self._view_w
        anchor_y = self._view_y0 + ry * self._view_h
        vw_int, vh_int = self._compute_view_size_for_zoom(self.zoom_factor)
        new_view_w = float(vw_int)
        new_view_h = float(vh_int)
        img_h, img_w = self.image_bgr.shape[:2]
        x0 = anchor_x - rx * new_view_w
        y0 = anchor_y - ry * new_view_h
        x0 = min(max(x0, 0.0), max(0.0, float(img_w) - new_view_w))
        y0 = min(max(y0, 0.0), max(0.0, float(img_h) - new_view_h))
        self._view_center_x = x0 + new_view_w / 2.0
        self._view_center_y = y0 + new_view_h / 2.0
        self._refresh()

    def _set_scale_measure_from_mask(self, mask: np.ndarray) -> bool:
        comp = (mask > 0).astype(np.uint8)
        if int(comp.sum()) <= 0:
            return False
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(comp, connectivity=8)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            if areas.size > 0:
                largest = int(np.argmax(areas)) + 1
                comp = (labels == largest).astype(np.uint8)
        fit = self._fit_bar_segment(comp)
        if fit is None:
            return False
        p0, p1, px_len, _short_len, _aspect = fit
        self.scale_bar_points = [(float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))]
        self.scale_bar_px_length = float(px_len)
        self.view.set_scale_calibration_pixels(self.scale_bar_px_length)
        return True

    def on_set(self) -> None:
        if not self._can_edit_in_preview():
            return
        if self.scale_calibration_mode:
            def _finish_measure_mode() -> None:
                self.scale_calibration_mode = False
                self.view.set_scale_calibration_mode(False)
                self.prompt_points.clear()
                self.prompt_box = None
                self.drag_prompt_box = None
                self.polygon_points.clear()
                self.current = None
                self.current_is_raw_display = False
                self._last_undoable_action = ""

            candidate: Optional[MaskEntry] = None
            if self.polygon_mode and self.polygon_points:
                candidate = self._generate_from_polygon()
                self.polygon_points.clear()
            elif self.prompt_points or self.prompt_box is not None:
                candidate = self._generate_from_prompts()
                self.prompt_points.clear()
                self.prompt_box = None
            elif self.current is not None:
                candidate = self.current

            if candidate is not None and self._set_scale_measure_from_mask(candidate.mask):
                _finish_measure_mode()
                self._mark_overlay_dirty()
                self._refresh()
                return

            if len(self.scale_bar_points) >= 2:
                (x0, y0), (x1, y1) = self.scale_bar_points[-2], self.scale_bar_points[-1]
                self.scale_bar_px_length = float(np.hypot(float(x1 - x0), float(y1 - y0)))
                self.view.set_scale_calibration_pixels(self.scale_bar_px_length)
                _finish_measure_mode()
                self._mark_overlay_dirty()
                self._refresh()
                return

            QMessageBox.information(
                self.view,
                "Scale Calibration",
                "No measurable bar yet. Add prompts/polygon, then press Set.",
            )
            return

        new_entry = None
        if self.polygon_mode and self.polygon_points:
            gen = self._generate_from_polygon()
            if gen is not None:
                new_entry = gen
            self.polygon_points.clear()
        elif self.prompt_points or self.prompt_box is not None:
            gen = self._generate_from_prompts()
            if gen is not None:
                new_entry = gen
            self.prompt_points.clear()
            self.prompt_box = None
        if new_entry is None and self.current:
            new_entry = self.current
        if new_entry:
            self._push_undo_snapshot()
            self.set_masks.append(new_entry)
            self.current = None
            self.current_is_raw_display = False
            self.drag_prompt_box = None
            self._set_single_selection(len(self.set_masks) - 1)
            self._mark_dirty_current()
            self._mark_overlay_dirty()
            self._auto_calc_if_ready()
            self._refresh()

    def on_toggle_raw(self) -> None:
        pass  # legacy shortcut removed

    def on_trim_overlap(self) -> None:
        if not self.current or not self.set_masks:
            return
        trimmed = self._apply_no_overlap(self.current.mask)
        if trimmed.sum() == 0:
            return
        self._push_undo_snapshot()
        self.current = MaskEntry(
            trimmed,
            trimmed,
            self.current.score,
            prompt_data=copy.deepcopy(self.current.prompt_data),
        )
        self.current_is_raw_display = False
        self._clear_selection()
        self._mark_dirty_current()
        self._mark_overlay_dirty()
        self._refresh()

    def on_prev_image(self) -> None:
        self._switch_image(self.current_image_idx - 1)

    def on_next_image(self) -> None:
        self._switch_image(self.current_image_idx + 1)

    def _build_aggregate_row(self, size_summary_save: Dict, length_unit: str) -> Dict:
        unit = self._normalize_length_unit(length_unit)
        area_unit = f"{unit}2"

        def _length_stat(stats: Dict[str, Any], key: str) -> Optional[float]:
            if not isinstance(stats, dict):
                return None
            return self._convert_length_from_nm(stats.get(key), unit)

        def _area_stat(stats: Dict[str, Any], key: str) -> Optional[float]:
            if not isinstance(stats, dict):
                return None
            return self._convert_area_from_nm2(stats.get(key), unit)

        n1 = self.summary.get("nearest1", {}) if self.summary else {}
        n2 = self.summary.get("nearest2", {}) if self.summary else {}
        c1 = self.summary.get("centroid1", {}) if self.summary else {}
        c2 = self.summary.get("centroid2", {}) if self.summary else {}
        ecd = size_summary_save.get("ecd", {}) if size_summary_save else {}
        vesd = size_summary_save.get("vesd", {}) if size_summary_save else {}
        area = size_summary_save.get("area", {}) if size_summary_save else {}
        bbox_area = size_summary_save.get("bbox_area", {}) if size_summary_save else {}
        area_vesd = size_summary_save.get("area_vesd", {}) if size_summary_save else {}
        major = size_summary_save.get("major_axis", {}) if size_summary_save else {}
        minor = size_summary_save.get("minor_axis", {}) if size_summary_save else {}
        feret_rect_area = size_summary_save.get("feret_rect_area", {}) if size_summary_save else {}
        ellipse_major = size_summary_save.get("ellipse_major_axis", {}) if size_summary_save else {}
        ellipse_minor = size_summary_save.get("ellipse_minor_axis", {}) if size_summary_save else {}
        aspect = size_summary_save.get("aspect_ratio", {}) if size_summary_save else {}
        shape = size_summary_save.get("shape_ratio", {}) if size_summary_save else {}
        return {
            "image_index": self.current_image_idx + 1,
            "image_name": self.base_image_path.name,
            "image_path": str(self.base_image_path),
            "output_dir": str(self.output_dir),
            "mask_count": len(self.set_masks),
            "length_unit": unit,
            "area_unit": area_unit,
            "n1_mean": _length_stat(n1, "mean"),
            "n1_std": _length_stat(n1, "std"),
            "n1_cv_pct": n1.get("cv_pct"),
            "n2_mean": _length_stat(n2, "mean"),
            "n2_std": _length_stat(n2, "std"),
            "n2_cv_pct": n2.get("cv_pct"),
            "cent1_mean": _length_stat(c1, "mean"),
            "cent1_std": _length_stat(c1, "std"),
            "cent1_cv_pct": c1.get("cv_pct"),
            "cent2_mean": _length_stat(c2, "mean"),
            "cent2_std": _length_stat(c2, "std"),
            "cent2_cv_pct": c2.get("cv_pct"),
            "ecd_mean": _length_stat(ecd, "mean"),
            "ecd_std": _length_stat(ecd, "std"),
            "ecd_cv_pct": ecd.get("cv_pct"),
            "vesd_mean": _length_stat(vesd, "mean"),
            "vesd_std": _length_stat(vesd, "std"),
            "vesd_cv_pct": vesd.get("cv_pct"),
            "volume_mean_diameter": self._convert_length_from_nm(size_summary_save.get("volume_mean_diameter"), unit) if size_summary_save else None,
            "area_mean": _area_stat(area, "mean"),
            "area_std": _area_stat(area, "std"),
            "area_cv_pct": area.get("cv_pct"),
            "bbox_area_mean": _area_stat(bbox_area, "mean"),
            "bbox_area_std": _area_stat(bbox_area, "std"),
            "bbox_area_cv_pct": bbox_area.get("cv_pct"),
            "area_vesd_mean": _area_stat(area_vesd, "mean"),
            "area_vesd_std": _area_stat(area_vesd, "std"),
            "area_vesd_cv_pct": area_vesd.get("cv_pct"),
            "major_axis_mean": _length_stat(major, "mean"),
            "major_axis_std": _length_stat(major, "std"),
            "major_axis_cv_pct": major.get("cv_pct"),
            "minor_axis_mean": _length_stat(minor, "mean"),
            "minor_axis_std": _length_stat(minor, "std"),
            "minor_axis_cv_pct": minor.get("cv_pct"),
            "feret_rect_area_mean": _area_stat(feret_rect_area, "mean"),
            "feret_rect_area_std": _area_stat(feret_rect_area, "std"),
            "feret_rect_area_cv_pct": feret_rect_area.get("cv_pct"),
            "ellipse_major_axis_mean": _length_stat(ellipse_major, "mean"),
            "ellipse_major_axis_std": _length_stat(ellipse_major, "std"),
            "ellipse_major_axis_cv_pct": ellipse_major.get("cv_pct"),
            "ellipse_minor_axis_mean": _length_stat(ellipse_minor, "mean"),
            "ellipse_minor_axis_std": _length_stat(ellipse_minor, "std"),
            "ellipse_minor_axis_cv_pct": ellipse_minor.get("cv_pct"),
            "aspect_ratio_mean": aspect.get("mean"),
            "aspect_ratio_std": aspect.get("std"),
            "aspect_ratio_cv_pct": aspect.get("cv_pct"),
            "shape_ratio_mean": shape.get("mean"),
            "shape_ratio_std": shape.get("std"),
            "shape_ratio_cv_pct": shape.get("cv_pct"),
            "cluster_count": self.cluster_stats.get("count") if self.cluster_stats else None,
            "cluster_mean_size": self.cluster_stats.get("mean") if self.cluster_stats else None,
            "cluster_std_size": self.cluster_stats.get("std") if self.cluster_stats else None,
            "cluster_cv_pct": self.cluster_stats.get("cv_pct") if self.cluster_stats else None,
        }

    def _save_current_outputs(self, show_message: bool = True) -> Dict:
        if len(self.set_masks) >= 2 and (self.summary_stale or not self.pair_results):
            self._run_calc_and_summary()
        masks = [m.mask for m in self.set_masks]
        # Save fractal metrics with the current UI setting (default: x20 OFF).
        size_summary_save = summarize_sizes(
            masks,
            self.scale_nm_per_px,
            fractal_slides=self._fractal_slides_effective(),
        ) if masks else {}
        export_unit = self._normalize_length_unit(self.display_length_unit)
        export_area_unit = f"{export_unit}2"
        ensure_outdir(self.output_dir)
        instance_dir = self.output_dir / "instance_masks"
        id_path = self.output_dir / "instance_ids.tiff"
        if self.set_masks:
            instance_dir.mkdir(parents=True, exist_ok=True)
            h, w = self.image_bgr.shape[:2]
            id_map = np.zeros((h, w), dtype=np.uint16)
            for idx, rec in enumerate(self.set_masks):
                inst_id = idx + 1
                mask_bool = rec.mask.astype(bool)
                id_map[mask_bool] = inst_id
                mask_u8 = (mask_bool.astype(np.uint8) * 255)
                cv2.imwrite(str(instance_dir / f"mask_{inst_id:04d}.png"), mask_u8)
            cv2.imwrite(str(id_path), id_map)
        filter_adjustments = self._current_filter_adjustments()
        adjust_slug = self._filter_adjust_slug(filter_adjustments)
        self.filter_chain = self._combined_filter_chain()
        spatial_slug = self._filter_chain_slug(self.spatial_filter_chain)
        frequency_slug = self._filter_chain_slug(self.frequency_filter_chain)
        filter_slug = f"sp_{spatial_slug}__fq_{frequency_slug}"
        filtered_name = f"filtered__{adjust_slug}__{filter_slug}.png"
        if isinstance(self.filtered_image_bgr, np.ndarray):
            cv2.imwrite(str(self.output_dir / filtered_name), self.filtered_image_bgr)
            cv2.imwrite(str(self.output_dir / "filtered_latest.png"), self.filtered_image_bgr)
        overlay = self._build_overlay(draw_pairs=bool(self.pair_results))
        cv2.imwrite(str(self.output_dir / "overlay.png"), overlay)
        hist_first = self.summary.get("hist_first", []) if self.summary else []
        hist_second = self.summary.get("hist_second", []) if self.summary else []
        hist_c1 = self.summary.get("hist_centroid1", []) if self.summary else []
        hist_c2 = self.summary.get("hist_centroid2", []) if self.summary else []
        save_histograms(self.output_dir, hist_first, hist_second, hist_c1, hist_c2)
        if size_summary_save:
            save_size_area_combo(self.output_dir, size_summary_save.get("hist_ecd", []), size_summary_save.get("hist_area", []))
            curve = size_summary_save.get("fractal_curve", {})
            gcurve = size_summary_save.get("fractal_global", {})
            save_fractal_loglog(
                self.output_dir,
                curve.get("log_eps", []),
                curve.get("log_counts", []),
                curve.get("slope"),
                name="fractal_loglog_curve",
            )
            save_fractal_loglog(
                self.output_dir,
                gcurve.get("log_eps", []),
                gcurve.get("log_counts", []),
                gcurve.get("slope"),
                name="fractal_loglog_global",
            )
        # CSV exports for full reconstruction
        write_hist_nearest_csv(
            self.output_dir,
            hist_first,
            hist_second,
            hist_c1,
            hist_c2,
            length_unit=export_unit,
        )
        if size_summary_save:
            write_hist_size_csv(
                self.output_dir,
                size_summary_save.get("hist_ecd", []),
                size_summary_save.get("hist_area", []),
                None,
                size_summary_save.get("hist_major_axis", []),
                size_summary_save.get("hist_minor_axis", []),
                size_summary_save.get("hist_feret_rect_area", []),
                size_summary_save.get("hist_aspect_ratio", []),
                size_summary_save.get("hist_shape_ratio", []),
                length_unit=export_unit,
            )
            gcurve = size_summary_save.get("fractal_global", {})
            write_fractal_global_csv(
                self.output_dir,
                gcurve.get("log_eps", []),
                gcurve.get("log_counts", []),
                gcurve.get("slope"),
                gcurve.get("value"),
            )
            write_boxcount_global_csv(
                self.output_dir,
                gcurve.get("sizes_px", []),
                gcurve.get("counts", []),
            )
        mask_info = []
        prompt_info = []
        for idx, rec in enumerate(self.set_masks):
            metrics = compute_mask_shape_metrics(rec.mask, self.scale_nm_per_px)
            prompt_payload = self._normalize_prompt_payload(rec.prompt_data)
            area_nm2 = metrics.get("area_nm2")
            bbox_area_nm2 = metrics.get("bbox_area_nm2")
            area_vesd_nm2 = metrics.get("area_vesd_nm2")
            feret_rect_area_nm2 = metrics.get("feret_rect_area_nm2")
            ecd_nm = metrics.get("ecd_nm")
            vesd_nm = metrics.get("vesd_nm")
            major_nm = metrics.get("major_axis_nm")
            minor_nm = metrics.get("minor_axis_nm")
            ellipse_major_nm = metrics.get("ellipse_major_axis_nm")
            ellipse_minor_nm = metrics.get("ellipse_minor_axis_nm")
            mask_info.append(
                {
                    "index": idx,
                    "instance_id": idx + 1,
                    "area": self._convert_area_from_nm2(area_nm2, export_unit),
                    "bbox_area": self._convert_area_from_nm2(bbox_area_nm2, export_unit),
                    "area_vesd": self._convert_area_from_nm2(area_vesd_nm2, export_unit),
                    "feret_rect_area": self._convert_area_from_nm2(feret_rect_area_nm2, export_unit),
                    "ecd": self._convert_length_from_nm(ecd_nm, export_unit),
                    "vesd": self._convert_length_from_nm(vesd_nm, export_unit),
                    "major_axis": self._convert_length_from_nm(major_nm, export_unit),
                    "minor_axis": self._convert_length_from_nm(minor_nm, export_unit),
                    "ellipse_major_axis": self._convert_length_from_nm(ellipse_major_nm, export_unit),
                    "ellipse_minor_axis": self._convert_length_from_nm(ellipse_minor_nm, export_unit),
                    "aspect_ratio": metrics.get("aspect_ratio"),
                    "score": self._score_value(rec.score),
                    "centroid_px": metrics.get("centroid_px"),
                    "bbox_xywh_px": metrics.get("bbox_xywh_px"),
                    "shape_ratio": metrics.get("shape_ratio"),
                    "length_unit": export_unit,
                    "area_unit": export_area_unit,
                    "area_px": metrics.get("area_px"),
                    "bbox_area_px": metrics.get("bbox_area_px"),
                    "major_axis_px": metrics.get("major_axis_px"),
                    "minor_axis_px": metrics.get("minor_axis_px"),
                    "feret_rect_area_px": metrics.get("feret_rect_area_px"),
                    "ellipse_major_axis_px": metrics.get("ellipse_major_axis_px"),
                    "ellipse_minor_axis_px": metrics.get("ellipse_minor_axis_px"),
                    "area_nm2": area_nm2,
                    "bbox_area_nm2": bbox_area_nm2,
                    "area_vesd_nm2": area_vesd_nm2,
                    "feret_rect_area_nm2": feret_rect_area_nm2,
                    "ecd_nm": ecd_nm,
                    "vesd_nm": vesd_nm,
                    "major_axis_nm": major_nm,
                    "minor_axis_nm": minor_nm,
                    "ellipse_major_axis_nm": ellipse_major_nm,
                    "ellipse_minor_axis_nm": ellipse_minor_nm,
                    "prompt": prompt_payload,
                }
            )
            prompt_info.append(
                {
                    "index": idx,
                    "instance_id": idx + 1,
                    "prompt": prompt_payload,
                }
            )
        summary_payload: Dict[str, Any] = {}
        summary_export: Dict[str, Any] = {}
        if isinstance(self.summary, dict):
            summary_order = (
                "nearest1",
                "nearest2",
                "centroid1",
                "centroid2",
                "hist_first",
                "hist_second",
                "hist_centroid1",
                "hist_centroid2",
            )
            for key in summary_order:
                if key in self.summary:
                    summary_payload[key] = copy.deepcopy(self.summary.get(key))
            for key in self.summary.keys():
                if key not in summary_payload:
                    summary_payload[key] = copy.deepcopy(self.summary.get(key))
            summary_export = copy.deepcopy(summary_payload)
            for key in ("nearest1", "nearest2", "centroid1", "centroid2"):
                stats = summary_payload.get(key)
                if not isinstance(stats, dict):
                    continue
                converted = copy.deepcopy(stats)
                for skey in ("mean", "median", "std", "min", "max"):
                    if skey in converted:
                        converted[skey] = self._convert_length_from_nm(converted.get(skey), export_unit)
                summary_export[key] = converted
            for key in ("hist_first", "hist_second", "hist_centroid1", "hist_centroid2"):
                values = summary_payload.get(key)
                if isinstance(values, list):
                    summary_export[key] = [self._convert_length_from_nm(v, export_unit) for v in values]

        size_summary_payload: Dict[str, Any] = {}
        size_summary_export: Dict[str, Any] = {}
        if isinstance(size_summary_save, dict):
            size_order = (
                "ecd",
                "vesd",
                "area",
                "bbox_area",
                "area_vesd",
                "major_axis",
                "minor_axis",
                "feret_rect_area",
                "ellipse_major_axis",
                "ellipse_minor_axis",
                "aspect_ratio",
                "shape_ratio",
                "volume_mean_diameter",
                "hist_ecd",
                "hist_vesd",
                "hist_area",
                "hist_bbox_area",
                "hist_area_vesd",
                "hist_major_axis",
                "hist_minor_axis",
                "hist_feret_rect_area",
                "hist_ellipse_major_axis",
                "hist_ellipse_minor_axis",
                "hist_aspect_ratio",
                "hist_shape_ratio",
                "fractal_global",
            )
            for key in size_order:
                if key in size_summary_save:
                    size_summary_payload[key] = copy.deepcopy(size_summary_save.get(key))
            for key in size_summary_save.keys():
                if key in {"fractal", "hist_fractal", "fractal_curve"}:
                    continue
                if key not in size_summary_payload:
                    size_summary_payload[key] = copy.deepcopy(size_summary_save.get(key))

            size_summary_export = copy.deepcopy(size_summary_payload)
            for key in ("ecd", "vesd", "major_axis", "minor_axis", "ellipse_major_axis", "ellipse_minor_axis"):
                stats = size_summary_payload.get(key)
                if not isinstance(stats, dict):
                    continue
                converted = copy.deepcopy(stats)
                for skey in ("mean", "median", "std", "min", "max"):
                    if skey in converted:
                        converted[skey] = self._convert_length_from_nm(converted.get(skey), export_unit)
                size_summary_export[key] = converted
            for key in ("area", "bbox_area", "area_vesd", "feret_rect_area"):
                stats = size_summary_payload.get(key)
                if not isinstance(stats, dict):
                    continue
                converted = copy.deepcopy(stats)
                for skey in ("mean", "median", "std", "min", "max"):
                    if skey in converted:
                        converted[skey] = self._convert_area_from_nm2(converted.get(skey), export_unit)
                size_summary_export[key] = converted
            for key in ("hist_ecd", "hist_vesd", "hist_major_axis", "hist_minor_axis", "hist_ellipse_major_axis", "hist_ellipse_minor_axis"):
                values = size_summary_payload.get(key)
                if isinstance(values, list):
                    size_summary_export[key] = [self._convert_length_from_nm(v, export_unit) for v in values]
            for key in ("hist_area", "hist_bbox_area", "hist_area_vesd", "hist_feret_rect_area"):
                values = size_summary_payload.get(key)
                if isinstance(values, list):
                    size_summary_export[key] = [self._convert_area_from_nm2(v, export_unit) for v in values]
            if "volume_mean_diameter" in size_summary_export:
                size_summary_export["volume_mean_diameter"] = self._convert_length_from_nm(
                    size_summary_payload.get("volume_mean_diameter"),
                    export_unit,
                )

        payload = {
            "pair_results": self.pair_results,
            "summary": summary_payload,
            "size_summary": size_summary_payload,
            "summary_export": summary_export,
            "size_summary_export": size_summary_export,
            "export_length_unit": export_unit,
            "export_area_unit": export_area_unit,
            "scale_per_px": self._convert_length_from_nm(self.scale_nm_per_px, export_unit),
            "max_distance": self._convert_length_from_nm(self.max_distance_nm, export_unit),
            "cluster_threshold": self._convert_length_from_nm(self.cluster_threshold_nm, export_unit),
            "scale_nm_per_px": self.scale_nm_per_px,
            "max_distance_nm": self.max_distance_nm,
            "fractal_slides": int(self.fractal_slides_setting),
            "fractal_slides_used": int(self._fractal_slides_effective()),
            "nearest_hist_metric": self.nearest_hist_metric,
            "size_hist_metric": self.size_hist_metric,
            "area_hist_metric": self.area_hist_metric,
            "aspect_hist_metric": self.aspect_hist_metric,
            "main_graph_metric": self.main_graph_metric,
            "set_count": len(self.set_masks),
            "cluster_threshold_nm": self.cluster_threshold_nm,
            "cluster_labels": self.cluster_labels,
            "cluster_stats": self.cluster_stats,
            "realtime_calc_enabled": bool(self.realtime_calc_enabled),
            "include_zero_distance": self.include_zero_distance,
            "show_bbox_overlay": self.show_bbox_overlay,
            "show_axes_overlay": self.show_axes_overlay,
            "show_feret_parallelogram_overlay": self.show_feret_parallelogram_overlay,
            "show_ellipse_overlay": self.show_ellipse_overlay,
            "distribution_metric": self.distribution_metric,
            "distribution_edges": [float(v) for v in self.distribution_edges],
            "filter_input_source": self.filter_input_source,
            "filter_adjustments": filter_adjustments,
            "filter_brightness": int(filter_adjustments["brightness"]),
            "filter_contrast": float(filter_adjustments["contrast"]),
            "filter_gamma": float(filter_adjustments["gamma"]),
            "spatial_filter_chain": copy.deepcopy(self.spatial_filter_chain),
            "frequency_filter_chain": copy.deepcopy(self.frequency_filter_chain),
            "spatial_filter_chain_slug": spatial_slug,
            "frequency_filter_chain_slug": frequency_slug,
            "filter_chain": copy.deepcopy(self.filter_chain),
            "filter_chain_slug": filter_slug,
            "filtered_image_path": filtered_name if isinstance(self.filtered_image_bgr, np.ndarray) else None,
            "masks": mask_info,
            "instance_mask_dir": str(instance_dir.name) if self.set_masks else None,
            "instance_id_path": str(id_path.name) if self.set_masks else None,
        }
        (self.output_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (self.output_dir / "instance_scores.json").write_text(
            json.dumps(
                {
                    "scores": [self._score_value(rec.score) for rec in self.set_masks],
                    "instance_ids": [idx + 1 for idx in range(len(self.set_masks))],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (self.output_dir / "instance_prompts.json").write_text(
            json.dumps(
                {
                    "instance_prompts": prompt_info,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        write_csv_summary(
            self.output_dir,
            self.summary,
            size_summary_save,
            self.cluster_stats,
            length_unit=export_unit,
        )
        write_mask_csv(self.output_dir, mask_info, length_unit=export_unit)
        row = self._build_aggregate_row(size_summary_save, export_unit)
        self._mark_saved_current()
        self._persist_current_session_state()
        try:
            self._write_recovery_snapshot()
        except Exception:
            pass
        if show_message:
            QtWidgets.QMessageBox.information(self.view, "Saved", f"Saved to {self.output_dir}")
        return row

    def on_save(self) -> None:
        dir_path = str(self.output_dir)
        new_dir = QFileDialog.getExistingDirectory(self.view, "Select output directory", dir_path)
        if not new_dir:
            return
        candidate = Path(new_dir)
        if candidate.exists() and candidate.is_dir() and candidate != self.output_dir and any(candidate.iterdir()):
            answer = QMessageBox.question(
                self.view,
                "Confirm Save",
                f"Directory {candidate} is not empty. Continue saving (files may be overwritten)?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        self.output_dir = candidate
        self._save_current_outputs(show_message=True)
        self._refresh()

    def on_save_all(self) -> None:
        base_default = str(self.base_output_dir)
        new_base = QFileDialog.getExistingDirectory(self.view, "Select base output directory (all images)", base_default)
        if not new_base:
            return
        base_dir = Path(new_base)
        if base_dir.exists() and base_dir.is_dir() and any(base_dir.iterdir()):
            answer = QMessageBox.question(
                self.view,
                "Confirm Save All",
                f"Directory {base_dir} is not empty. Continue saving all images (files may be overwritten)?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        prev_idx = self.current_image_idx
        self._persist_current_session_state()
        aggregate_rows = []
        for idx in range(len(self.image_sessions)):
            self._apply_image_session(idx)
            target = base_dir / f"{idx + 1:04d}_{self.base_image_path.stem}"
            self.output_dir = target
            aggregate_rows.append(self._save_current_outputs(show_message=False))
        write_all_images_index_csv(base_dir, aggregate_rows)
        prev_scope = self.analysis_scope
        try:
            self.analysis_scope = "all"
            payload_all = self._analysis_scope_payload()
        finally:
            self.analysis_scope = prev_scope
        summary_all = payload_all.get("summary", {}) if isinstance(payload_all.get("summary"), dict) else {}
        size_all = payload_all.get("size_summary", {}) if isinstance(payload_all.get("size_summary"), dict) else {}
        export_unit = self._normalize_length_unit(self.display_length_unit)
        write_csv_summary(
            base_dir,
            summary_all,
            size_all,
            {},
            length_unit=export_unit,
            filename="all_images_summary.csv",
        )
        self._mark_saved_all()
        try:
            self._write_recovery_snapshot()
        except Exception:
            pass
        self._apply_image_session(prev_idx)
        self._refresh()
        QtWidgets.QMessageBox.information(
            self.view,
            "Saved All",
            (
                f"Saved {len(self.image_sessions)} images under {base_dir}\n"
                f"Summary: {base_dir / 'all_images_summary.csv'}\n"
                f"Index: {base_dir / 'all_images_index.csv'}"
            ),
        )

    def on_undo(self) -> None:
        if not self._can_edit_in_preview():
            return
        if self._last_undoable_action == "polygon_add" and self.polygon_points:
            self.polygon_points.pop()
            self._last_undoable_action = "polygon_add" if self.polygon_points else ""
            self._mark_overlay_dirty()
            self._update_polygon_preview()
            return
        if self._last_undoable_action == "prompt_add" and self.prompt_points:
            self.prompt_points.pop()
            if self.prompt_points:
                self._last_undoable_action = "prompt_add"
            elif self.prompt_box is not None:
                self._last_undoable_action = "box_set"
            else:
                self._last_undoable_action = ""
            self._mark_overlay_dirty()
            self._update_prompt_preview()
            return
        if self._last_undoable_action == "box_set" and self.prompt_box is not None:
            self.prompt_box = None
            self._last_undoable_action = "prompt_add" if self.prompt_points else ""
            self._mark_overlay_dirty()
            self._update_prompt_preview()
            return
        if self._undo_stack:
            snapshot = self._undo_stack.pop()
            self._restore_undo_snapshot(snapshot)
            self._refresh()
            return
        # fallback: legacy point-wise undo when no snapshot exists
        if self.polygon_points:
            self.polygon_points.pop()
            self._mark_overlay_dirty()
            self._update_polygon_preview()
            return
        if self.prompt_points:
            self.prompt_points.pop()
            self._mark_overlay_dirty()
            self._update_prompt_preview()
            return
        if self.set_masks:
            self.set_masks.pop()
            self.summary_stale = True
            if self.selected_idx is not None:
                if self.selected_idx >= len(self.set_masks):
                    self.selected_idx = len(self.set_masks) - 1 if self.set_masks else None
            self.selected_indices = {
                idx for idx in self._valid_selected_indices()
                if idx < len(self.set_masks)
            }
            if self.selected_idx is not None:
                self.selected_indices.add(int(self.selected_idx))
            self._mark_dirty_current()
            self._mark_overlay_dirty()
            self._auto_calc_if_ready()
            self._refresh()

    def on_reset(self) -> None:
        if not self._can_edit_in_preview():
            return
        if self.polygon_points:
            self._push_undo_snapshot()
            self.polygon_points.clear()
            self.drag_prompt_box = None
            self.current = None
            self.current_is_raw_display = False
            self._mark_overlay_dirty()
            self._refresh()
            return
        if self.prompt_points or self.prompt_box is not None:
            self._push_undo_snapshot()
            self.prompt_points.clear()
            self.prompt_box = None
            self.drag_prompt_box = None
            self.current = None
            self.current_is_raw_display = False
            self._mark_overlay_dirty()
            self._refresh()
            return
        if self.set_masks:
            self._push_undo_snapshot()
        self.set_masks.clear()
        self.current = None
        self.current_is_raw_display = False
        self.drag_prompt_box = None
        self._clear_analysis_state()
        self._clear_selection()
        self._mark_dirty_current()
        self._mark_overlay_dirty()
        self._refresh()

    @staticmethod
    def _parse_magnification_label_to_value(label: str) -> Optional[float]:
        token = str(label or "").strip().lower().replace(",", "")
        if not token:
            return None
        if token.endswith("m"):
            token_num = token[:-1].strip()
            try:
                v = float(token_num)
            except Exception:
                return None
            if not np.isfinite(v) or v <= 0:
                return None
            return float(v * 1000.0 * 1000.0)
        if token.endswith("k"):
            token_num = token[:-1].strip()
            try:
                v = float(token_num)
            except Exception:
                return None
            if not np.isfinite(v) or v <= 0:
                return None
            return float(v * 1000.0)
        try:
            v = float(token)
        except Exception:
            return None
        if not np.isfinite(v) or v <= 0:
            return None
        return float(v)

    def _build_scale_preset_norm_positions(self) -> List[float]:
        n = len(self._scale_preset_labels)
        if n <= 0:
            self._scale_mag_min = 0.0
            self._scale_mag_max = 0.0
            self._scale_mag_log_span = 0.0
            return []
        if n == 1:
            only = self._scale_preset_magnifications[0]
            v = float(only) if only is not None and np.isfinite(float(only)) and float(only) > 0 else 1.0
            self._scale_mag_min = v
            self._scale_mag_max = v
            self._scale_mag_log_span = 0.0
            return [0.0]

        mags: List[Optional[float]] = []
        valid: List[float] = []
        for m in self._scale_preset_magnifications:
            if m is None:
                mags.append(None)
                continue
            mv = float(m)
            if not np.isfinite(mv) or mv <= 0:
                mags.append(None)
                continue
            mags.append(mv)
            valid.append(mv)
        if len(valid) < 2:
            self._scale_mag_min = 1.0
            self._scale_mag_max = 1.0
            self._scale_mag_log_span = 0.0
            return [float(i) / float(n - 1) for i in range(n)]

        mag_min = float(min(valid))
        mag_max = float(max(valid))
        log_min = float(np.log(mag_min))
        log_span = float(np.log(mag_max) - log_min)
        if not np.isfinite(log_span) or log_span <= 1e-12:
            self._scale_mag_min = mag_min
            self._scale_mag_max = mag_max
            self._scale_mag_log_span = 0.0
            return [float(i) / float(n - 1) for i in range(n)]

        self._scale_mag_min = mag_min
        self._scale_mag_max = mag_max
        self._scale_mag_log_span = log_span
        out: List[float] = []
        for i, mv in enumerate(mags):
            if mv is None:
                out.append(float(i) / float(n - 1))
                continue
            pos = (float(np.log(mv)) - log_min) / log_span
            out.append(float(np.clip(pos, 0.0, 1.0)))
        return out

    def _scale_from_slider_norm(self, norm_t: float) -> Optional[float]:
        if not np.isfinite(norm_t):
            return None
        if self._scale_preset_factor_nm is None or not np.isfinite(self._scale_preset_factor_nm) or self._scale_preset_factor_nm <= 0:
            return None
        if self._scale_mag_min <= 0 or self._scale_mag_max <= 0:
            return None
        t = float(np.clip(norm_t, 0.0, 1.0))
        if self._scale_mag_log_span <= 1e-12:
            mag = float(self._scale_mag_min)
        else:
            mag = float(self._scale_mag_min * np.exp(self._scale_mag_log_span * t))
        if not np.isfinite(mag) or mag <= 0:
            return None
        return float(self._scale_preset_factor_nm / mag)

    def _magnification_from_slider_norm(self, norm_t: float) -> Optional[float]:
        if not np.isfinite(norm_t):
            return None
        if self._scale_mag_min <= 0 or self._scale_mag_max <= 0:
            return None
        t = float(np.clip(norm_t, 0.0, 1.0))
        if self._scale_mag_log_span <= 1e-12:
            return float(self._scale_mag_min)
        mag = float(self._scale_mag_min * np.exp(self._scale_mag_log_span * t))
        if not np.isfinite(mag) or mag <= 0:
            return None
        return mag

    def _slider_norm_from_magnification(self, magnification: float) -> Optional[float]:
        if not np.isfinite(magnification) or magnification <= 0:
            return None
        if self._scale_mag_min <= 0 or self._scale_mag_max <= 0:
            return None
        mag = float(np.clip(float(magnification), self._scale_mag_min, self._scale_mag_max))
        if self._scale_mag_log_span <= 1e-12:
            return 0.0
        t = (float(np.log(mag)) - float(np.log(self._scale_mag_min))) / self._scale_mag_log_span
        return float(np.clip(t, 0.0, 1.0))

    def _slider_norm_from_scale(self, scale_nm_per_px: float) -> Optional[float]:
        if not np.isfinite(scale_nm_per_px) or scale_nm_per_px <= 0:
            return None
        if self._scale_preset_factor_nm is None or not np.isfinite(self._scale_preset_factor_nm) or self._scale_preset_factor_nm <= 0:
            return None
        if self._scale_mag_min <= 0 or self._scale_mag_max <= 0:
            return None
        mag = float(self._scale_preset_factor_nm / float(scale_nm_per_px))
        if not np.isfinite(mag) or mag <= 0:
            return None
        mag = float(np.clip(mag, self._scale_mag_min, self._scale_mag_max))
        if self._scale_mag_log_span <= 1e-12:
            return 0.0
        t = (float(np.log(mag)) - float(np.log(self._scale_mag_min))) / self._scale_mag_log_span
        return float(np.clip(t, 0.0, 1.0))

    def _nearest_scale_preset_index_from_norm(self, norm_t: float) -> int:
        if not self._scale_preset_norm_positions:
            return 0
        t = float(np.clip(norm_t, 0.0, 1.0))
        best_idx = 0
        best_dist = float("inf")
        for idx, p in enumerate(self._scale_preset_norm_positions):
            d = abs(float(p) - t)
            if d < best_dist:
                best_dist = d
                best_idx = idx
        return int(best_idx)

    @staticmethod
    def _format_magnification_label(mag: float) -> str:
        if not np.isfinite(mag) or mag <= 0:
            return "-"
        if mag >= 1000.0 * 1000.0:
            m = int(round(mag / (1000.0 * 1000.0)))
            return f"{m}M"
        if mag >= 1000.0:
            k = int(round(mag / 1000.0))
            return f"{k}k"
        return f"{mag:.0f}"

    def _sync_scale_preset_ui_from_scale(self) -> None:
        if not hasattr(self, "view"):
            return
        if not self._scale_preset_labels:
            self.view.set_scale_preset_hint_text("")
            return
        tick_data = list(zip(self._scale_preset_norm_positions, self._scale_preset_labels))
        self.view.set_scale_preset_ticks(tick_data)

        norm_t = self._slider_norm_from_scale(self.scale_nm_per_px)
        if norm_t is None:
            norm_t = 0.0
        slider_value = int(round(float(np.clip(norm_t, 0.0, 1.0)) * float(self._scale_slider_steps)))
        self.view.set_scale_preset_slider_value(slider_value)

        nearest_idx = self._nearest_scale_preset_index_from_norm(norm_t)
        nearest_label = self._scale_preset_labels[nearest_idx] if 0 <= nearest_idx < len(self._scale_preset_labels) else "-"
        nearest_pos = (
            float(self._scale_preset_norm_positions[nearest_idx])
            if 0 <= nearest_idx < len(self._scale_preset_norm_positions)
            else norm_t
        )
        mag_now = float(self._scale_preset_factor_nm / max(float(self.scale_nm_per_px), 1e-9))
        mag_label = self._format_magnification_label(mag_now)
        if abs(norm_t - nearest_pos) <= self._scale_slider_snap_norm:
            self.view.set_scale_preset_hint_text(f"{nearest_label} ({self.scale_nm_per_px:.3f})")
        else:
            self.view.set_scale_preset_hint_text(f"{mag_label} ({self.scale_nm_per_px:.3f})")

    def on_scale_preset_slider_changed(self, slider_value: int) -> None:
        if not self._scale_preset_labels:
            return
        raw_t = float(np.clip(float(slider_value) / float(max(1, self._scale_slider_steps)), 0.0, 1.0))
        t = raw_t
        snapped = False
        nearest_idx = self._nearest_scale_preset_index_from_norm(raw_t)
        if 0 <= nearest_idx < len(self._scale_preset_norm_positions):
            snap_t = float(self._scale_preset_norm_positions[nearest_idx])
            if abs(raw_t - snap_t) <= self._scale_slider_snap_norm:
                t = snap_t
                snapped = True
                snapped_value = int(round(snap_t * float(self._scale_slider_steps)))
                if snapped_value != int(slider_value):
                    self.view.set_scale_preset_slider_value(snapped_value)
        if not snapped and self._scale_nonpreset_mag_step > 0:
            mag = self._magnification_from_slider_norm(raw_t)
            if mag is not None:
                mag_q = round(float(mag) / float(self._scale_nonpreset_mag_step)) * float(self._scale_nonpreset_mag_step)
                mag_q = float(np.clip(mag_q, self._scale_mag_min, self._scale_mag_max))
                q_t = self._slider_norm_from_magnification(mag_q)
                if q_t is not None:
                    t = q_t
                    snapped_value = int(round(float(q_t) * float(self._scale_slider_steps)))
                    if snapped_value != int(slider_value):
                        self.view.set_scale_preset_slider_value(snapped_value)
        value = self._scale_from_slider_norm(t)
        if value is None or not np.isfinite(value) or value <= 0:
            return
        self.on_scale_preset_clicked(float(value))

    def on_apply_scale(self) -> None:
        was_pending = bool(self.summary_stale)
        try:
            v = float(self.view.get_scale_text())
            self.scale_nm_per_px = max(v, 0.0001)
        except ValueError:
            pass
        self.image_sessions[self.current_image_idx].scale_nm_per_px = float(self.scale_nm_per_px)
        self.view.set_scale_text(f"{self.scale_nm_per_px:.3f}")
        self._sync_scale_preset_ui_from_scale()
        self._mark_dirty_current()
        self._refresh_scale_dependent_stats_after_scale_change(was_pending)
        self._mark_analysis_dirty()
        self._refresh()

    @staticmethod
    def _length_to_nm(length_value: float, unit: str) -> Optional[float]:
        unit_norm = (unit or "").strip().lower().replace("µ", "u").replace("μ", "u")
        scale_map = {
            "nm": 1.0,
            "um": 1000.0,
            "mm": 1000.0 * 1000.0,
        }
        mult = scale_map.get(unit_norm)
        if mult is None:
            return None
        return float(length_value) * float(mult)

    @staticmethod
    def _normalize_length_unit(unit: str) -> str:
        u = (unit or "nm").strip().lower().replace("µ", "u").replace("μ", "u")
        if u not in {"nm", "um", "mm"}:
            return "nm"
        return u

    @classmethod
    def _length_scale_nm(cls, unit: str) -> float:
        u = cls._normalize_length_unit(unit)
        if u == "um":
            return 1000.0
        if u == "mm":
            return 1000.0 * 1000.0
        return 1.0

    @classmethod
    def _convert_length_from_nm(cls, value_nm: Optional[float], unit: str) -> Optional[float]:
        if value_nm is None:
            return None
        scale = cls._length_scale_nm(unit)
        return float(value_nm) / scale

    @classmethod
    def _convert_area_from_nm2(cls, value_nm2: Optional[float], unit: str) -> Optional[float]:
        if value_nm2 is None:
            return None
        scale = cls._length_scale_nm(unit)
        return float(value_nm2) / (scale * scale)

    def _convert_length_series_from_nm(self, values: Sequence[Any]) -> List[float]:
        out: List[float] = []
        for v in values:
            try:
                fv = float(v)
            except Exception:
                continue
            if not np.isfinite(fv):
                continue
            conv = self._convert_length_from_nm(fv, self.display_length_unit)
            if conv is None:
                continue
            conv_f = float(conv)
            if np.isfinite(conv_f):
                out.append(conv_f)
        return out

    def _convert_area_series_from_nm2(self, values: Sequence[Any]) -> List[float]:
        out: List[float] = []
        for v in values:
            try:
                fv = float(v)
            except Exception:
                continue
            if not np.isfinite(fv):
                continue
            conv = self._convert_area_from_nm2(fv, self.display_length_unit)
            if conv is None:
                continue
            conv_f = float(conv)
            if np.isfinite(conv_f):
                out.append(conv_f)
        return out

    def _convert_optional_length_from_nm(self, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        try:
            fv = float(value)
        except Exception:
            return None
        if not np.isfinite(fv):
            return None
        conv = self._convert_length_from_nm(fv, self.display_length_unit)
        if conv is None:
            return None
        conv_f = float(conv)
        return conv_f if np.isfinite(conv_f) else None

    def _convert_optional_area_from_nm2(self, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        try:
            fv = float(value)
        except Exception:
            return None
        if not np.isfinite(fv):
            return None
        conv = self._convert_area_from_nm2(fv, self.display_length_unit)
        if conv is None:
            return None
        conv_f = float(conv)
        return conv_f if np.isfinite(conv_f) else None

    @staticmethod
    def _fit_bar_segment(component_u8: np.ndarray) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], float, float, float]]:
        contours, _ = cv2.findContours(component_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area <= 1.0:
            return None
        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]
        long_len = float(max(rw, rh))
        short_len = float(min(rw, rh))
        if not np.isfinite(long_len) or not np.isfinite(short_len):
            return None
        if long_len <= 1.0 or short_len <= 0.0:
            return None
        box = cv2.boxPoints(rect).astype(np.float32)
        best_vec: Optional[np.ndarray] = None
        best_len = 0.0
        for i in range(4):
            p0 = box[i]
            p1 = box[(i + 1) % 4]
            vec = p1 - p0
            length = float(np.hypot(float(vec[0]), float(vec[1])))
            if length > best_len:
                best_len = length
                if length > 1e-6:
                    best_vec = vec / length
        if best_vec is None or best_len <= 1.0:
            return None
        center = box.mean(axis=0)
        half = best_len * 0.5
        pa = center - best_vec * half
        pb = center + best_vec * half
        aspect = float(long_len / max(short_len, 1e-6))
        return (float(pa[0]), float(pa[1])), (float(pb[0]), float(pb[1])), float(best_len), float(short_len), float(aspect)

    @staticmethod
    def _mask_major_minor_segments(
        mask_u8: np.ndarray,
    ) -> Optional[
        Tuple[
            Tuple[float, float],
            Tuple[float, float],
            Tuple[float, float],
            Tuple[float, float],
        ]
    ]:
        return mask_feret_major_minor_segments_px(mask_u8)

    @staticmethod
    def _feret_parallelogram_from_segments(
        major_minor_segments: Tuple[
            Tuple[float, float],
            Tuple[float, float],
            Tuple[float, float],
            Tuple[float, float],
        ],
        center_xy: Tuple[float, float],
    ) -> Optional[np.ndarray]:
        ma0, ma1, mi0, mi1 = major_minor_segments
        v_major = np.array([float(ma1[0]) - float(ma0[0]), float(ma1[1]) - float(ma0[1])], dtype=np.float32)
        v_minor = np.array([float(mi1[0]) - float(mi0[0]), float(mi1[1]) - float(mi0[1])], dtype=np.float32)
        len_major = float(np.hypot(float(v_major[0]), float(v_major[1])))
        len_minor = float(np.hypot(float(v_minor[0]), float(v_minor[1])))
        if len_major <= 1e-6 or len_minor <= 1e-6:
            return None
        center = np.array([float(center_xy[0]), float(center_xy[1])], dtype=np.float32)
        p0 = center - 0.5 * v_major - 0.5 * v_minor
        p1 = p0 + v_major
        p2 = p1 + v_minor
        p3 = p0 + v_minor
        pts = np.array([p0, p1, p2, p3], dtype=np.float32)
        if not np.isfinite(pts).all():
            return None
        return pts

    def on_display_unit_changed(self, *_args) -> None:
        unit = self._normalize_length_unit(self.view.get_display_unit())
        if unit == self.display_length_unit:
            self.view.set_display_unit(unit)
            return
        self.display_length_unit = unit
        self.view.set_display_unit(unit)
        if len(self.distribution_edges) >= 2:
            self.view.set_distribution_edges_text(
                self._format_distribution_edges_for_ui(self.distribution_metric, self.distribution_edges)
            )
        self._review_cache_key = None
        self._stats_cache_revision = -1
        self._mark_analysis_dirty()
        if not self._try_apply_scale_from_inputs(silent=True, finish_measure=False, allow_fallback_px=False):
            self._refresh()

    def on_toggle_scale_calibration_mode(self) -> None:
        enabled = bool(self.view.scale_calib_toggle_btn.isChecked())
        if enabled:
            # Keep current manual px input and restore it if measure mode is canceled.
            self._scale_px_before_measure = self.view.get_scale_px_text()
        self.scale_calibration_mode = enabled
        self.scale_bar_points = []
        self.scale_bar_px_length = None
        self.view.set_scale_calibration_mode(enabled)
        if enabled:
            self.view.set_scale_calibration_pixels(None)
        else:
            current_px_text = self.view.get_scale_px_text().strip()
            if not current_px_text and self._scale_px_before_measure.strip():
                self.view.set_scale_px_text(self._scale_px_before_measure)
            self._scale_px_before_measure = ""
        if enabled:
            # Avoid accidental prompt edits while measuring scale bars.
            self.prompt_points.clear()
            self.prompt_box = None
            self.drag_prompt_box = None
            self.polygon_points.clear()
            self.current = None
            self.current_is_raw_display = False
            self._last_undoable_action = ""
        self._mark_overlay_dirty()
        self._refresh()

    def _try_apply_scale_from_inputs(
        self,
        *,
        silent: bool,
        finish_measure: bool,
        allow_fallback_px: bool,
    ) -> bool:
        was_pending = bool(self.summary_stale)
        px_len: Optional[float] = None
        px_raw = self.view.get_scale_px_text().strip().replace(",", ".") if hasattr(self.view, "get_scale_px_text") else ""
        if px_raw:
            try:
                px_len = float(px_raw)
            except Exception:
                if not silent:
                    QMessageBox.warning(self.view, "Scale Calibration", f"Invalid px value: {px_raw}")
                return False
        if px_len is None and allow_fallback_px:
            px_len = self.scale_bar_px_length
        if px_len is None or not np.isfinite(px_len) or px_len <= 0:
            if not silent:
                QMessageBox.information(self.view, "Scale Calibration", "Set px first (Measure + Set, or manual input).")
            return False
        raw = self.view.get_scale_bar_length_text().strip().replace(",", ".")
        if not raw:
            if not silent:
                QMessageBox.information(self.view, "Scale Calibration", "Enter the known bar length.")
            return False
        try:
            known_length = float(raw)
        except Exception:
            if not silent:
                QMessageBox.warning(self.view, "Scale Calibration", f"Invalid bar length: {raw}")
            return False
        if not np.isfinite(known_length) or known_length <= 0:
            if not silent:
                QMessageBox.warning(self.view, "Scale Calibration", "Bar length must be a positive number.")
            return False
        known_nm = self._length_to_nm(known_length, self.view.get_display_unit())
        if known_nm is None:
            if not silent:
                QMessageBox.warning(self.view, "Scale Calibration", "Unsupported unit. Use nm, um, or mm.")
            return False
        self.scale_bar_px_length = float(px_len)
        self.view.set_scale_calibration_pixels(self.scale_bar_px_length)
        self.scale_nm_per_px = max(float(known_nm) / float(px_len), 0.0001)
        self.image_sessions[self.current_image_idx].scale_nm_per_px = float(self.scale_nm_per_px)
        self.view.set_scale_text(f"{self.scale_nm_per_px:.3f}")
        self._sync_scale_preset_ui_from_scale()
        # Finish measurement workflow automatically after successful apply.
        if finish_measure:
            self.scale_calibration_mode = False
            self.scale_bar_points = []
            self.view.set_scale_calibration_mode(False)
            self._scale_px_before_measure = ""
        self._mark_dirty_current()
        self._refresh_scale_dependent_stats_after_scale_change(was_pending)
        self._mark_analysis_dirty()
        self._refresh()
        return True

    def on_apply_scale_from_bar(self) -> None:
        self._try_apply_scale_from_inputs(silent=False, finish_measure=True, allow_fallback_px=True)

    def on_scale_fields_edited(self) -> None:
        self._try_apply_scale_from_inputs(silent=True, finish_measure=False, allow_fallback_px=False)

    def on_scale_preset_clicked(self, value_nm_per_px: float) -> None:
        try:
            val = float(value_nm_per_px)
        except Exception:
            return
        if not np.isfinite(val) or val <= 0:
            return
        if self.display_length_unit != "nm":
            self.display_length_unit = "nm"
            self.view.set_display_unit("nm")
            self._review_cache_key = None
            self._stats_cache_revision = -1
        text = f"{val:.4f}".rstrip("0").rstrip(".")
        self.view.set_scale_bar_length_text(text)
        self.view.set_scale_px_text("1")
        self._try_apply_scale_from_inputs(silent=False, finish_measure=True, allow_fallback_px=False)

    def on_set_scale(self, v: float) -> None:
        was_pending = bool(self.summary_stale)
        self.scale_nm_per_px = max(float(v), 0.0001)
        self.image_sessions[self.current_image_idx].scale_nm_per_px = float(self.scale_nm_per_px)
        self.view.set_scale_text(f"{self.scale_nm_per_px:.3f}")
        self._sync_scale_preset_ui_from_scale()
        self._mark_dirty_current()
        self._refresh_scale_dependent_stats_after_scale_change(was_pending)
        self._mark_analysis_dirty()
        self._refresh()

    def _refresh_scale_dependent_stats_after_scale_change(self, was_pending: bool) -> None:
        masks = [m.mask for m in self.set_masks]
        if not masks:
            if not was_pending:
                self.pair_results = []
                self.summary = {}
            self.size_summary = {}
            self.cluster_labels = []
            self.cluster_stats = {}
            self.summary_stale = False
            return

        # Size/area metrics are deterministic from masks + scale.
        self.size_summary = summarize_sizes(
            masks,
            self.scale_nm_per_px,
            fractal_slides=self._fractal_slides_effective(),
        )

        if not was_pending:
            if len(masks) >= 2:
                self.pair_results = compute_two_nearest(
                    masks,
                    self.scale_nm_per_px,
                    self.max_distance_nm,
                    include_zero=self.include_zero_distance,
                )
                self.summary = summarize(self.pair_results)
                self._compute_clusters()
            else:
                self.pair_results = []
                self.summary = {}
                self.cluster_labels = []
                self.cluster_stats = {}

        self.summary_stale = bool(was_pending and len(masks) > 0)
        state = self.image_sessions[self.current_image_idx]
        state.pair_results = self.pair_results
        state.summary = self.summary
        state.size_summary = self.size_summary
        state.cluster_labels = self.cluster_labels
        state.cluster_stats = self.cluster_stats
        state.summary_stale = self.summary_stale

    def on_apply_cluster_threshold(self) -> None:
        try:
            v = float(self.view.get_cluster_text().strip().replace(",", "."))
            v_nm = self._length_to_nm(v, self.display_length_unit)
            if v_nm is None:
                raise ValueError("invalid unit")
            self.cluster_threshold_nm = max(float(v_nm), 0.0)
        except ValueError:
            pass
        self._sync_cluster_controls()
        self._mark_dirty_all()
        self.cluster_labels = []
        self.cluster_stats = {}
        self.summary_stale = bool(self.set_masks)
        self._mark_analysis_dirty()
        self._refresh()

    def on_toggle_fractal_slides(self) -> None:
        self.fractal_slides_setting = 20 if self.view.fractal_checkbox.isChecked() else 0
        self._mark_dirty_all()
        self._mark_all_sessions_summary_stale()
        self._mark_analysis_dirty()
        self._auto_calc_if_ready()
        self._refresh()

    def on_toggle_overlay_centroid(self) -> None:
        self.overlay_use_centroid = self.view.overlay_centroid_checkbox.isChecked()
        self._mark_overlay_dirty()
        self._refresh()

    def on_toggle_realtime_calc(self) -> None:
        self.realtime_calc_enabled = bool(self.view.realtime_calc_checkbox.isChecked())
        self._mark_dirty_current()
        self._auto_calc_if_ready()
        self._refresh()

    def on_toggle_include_zero(self) -> None:
        self.include_zero_distance = self.view.include_zero_checkbox.isChecked()
        self._mark_dirty_all()
        self._mark_all_sessions_summary_stale()
        self._mark_analysis_dirty()
        self._auto_calc_if_ready()
        self._refresh()

    def on_toggle_show_links(self) -> None:
        self.show_nearest_links = self.view.show_nearest_checkbox.isChecked()
        self._mark_overlay_dirty()
        self._refresh()

    def on_shortcut_toggle_show_current_mask(self) -> None:
        if not hasattr(self.view, "show_current_checkbox"):
            return
        self.view.show_current_checkbox.setChecked(not self.view.show_current_checkbox.isChecked())

    def on_toggle_show_current(self) -> None:
        self.show_current_mask = self.view.show_current_checkbox.isChecked()
        self._mark_overlay_dirty()
        self._refresh()

    def on_toggle_show_set(self) -> None:
        self.show_set_masks = self.view.show_set_checkbox.isChecked()
        self._mark_overlay_dirty()
        self._refresh()

    def on_toggle_show_prompts(self) -> None:
        self.show_prompts = self.view.show_prompts_checkbox.isChecked()
        self._mark_overlay_dirty()
        self._refresh()

    def on_toggle_show_gt_overlay(self) -> None:
        self.show_gt_overlay = self.view.show_gt_checkbox.isChecked()
        self._mark_overlay_dirty()
        self._refresh()

    def on_toggle_show_bbox(self) -> None:
        self.show_bbox_overlay = self.view.show_bbox_checkbox.isChecked()
        self._mark_overlay_dirty()
        self._refresh()

    def on_toggle_show_feret(self) -> None:
        checked = self.view.show_axes_checkbox.isChecked()
        self.show_axes_overlay = bool(checked)
        self.show_feret_parallelogram_overlay = bool(checked)
        self._mark_overlay_dirty()
        self._refresh()

    def on_toggle_show_axes(self) -> None:
        # Backward-compatible path: unified Feret toggle controls both.
        self.on_toggle_show_feret()

    def on_toggle_show_feret_parallelogram(self) -> None:
        # Backward-compatible path: unified Feret toggle controls both.
        self.on_toggle_show_feret()

    def on_toggle_show_ellipse(self) -> None:
        self.show_ellipse_overlay = self.view.show_ellipse_checkbox.isChecked()
        self._mark_overlay_dirty()
        self._refresh()

    def on_nearest_hist_metric_changed(self, *_args) -> None:
        next_metric = self._normalize_nearest_hist_metric(self.view.get_nearest_hist_metric())
        if next_metric == self.nearest_hist_metric:
            return
        self.nearest_hist_metric = next_metric
        self._mark_analysis_dirty()
        self._refresh()

    def on_size_hist_metric_changed(self, *_args) -> None:
        next_metric = self._normalize_size_hist_metric(self.view.get_size_hist_metric())
        if next_metric == self.size_hist_metric:
            return
        self.size_hist_metric = next_metric
        self._mark_analysis_dirty()
        self._refresh()

    def on_area_hist_metric_changed(self, *_args) -> None:
        next_metric = self._normalize_area_hist_metric(self.view.get_area_hist_metric())
        if next_metric == self.area_hist_metric:
            return
        self.area_hist_metric = next_metric
        self._mark_analysis_dirty()
        self._refresh()

    def on_aspect_hist_metric_changed(self, *_args) -> None:
        next_metric = self._normalize_aspect_hist_metric(self.view.get_aspect_hist_metric())
        if next_metric == self.aspect_hist_metric:
            return
        self.aspect_hist_metric = next_metric
        self._mark_analysis_dirty()
        self._refresh()

    def on_main_graph_metric_changed(self, *_args) -> None:
        next_metric = self._normalize_main_graph_metric(self.view.get_main_graph_metric())
        if next_metric == self.main_graph_metric:
            return
        self.main_graph_metric = next_metric
        self._mark_analysis_dirty()
        self._refresh()

    def on_distribution_metric_changed(self, *_args) -> None:
        metric = self._normalize_distribution_metric(self.view.get_distribution_metric())
        if metric == "none":
            self.distribution_metric = "none"
            self.distribution_edges = []
            self._distribution_slider_min_internal = None
            self._distribution_slider_max_internal = None
            self.view.set_distribution_slider_state(0.0, 1.0, [], enabled=False)
            self.view.set_distribution_status("")
            self._mark_overlay_dirty()
            self._mark_analysis_dirty()
            self._refresh()
            return
        prev_metric = self._normalize_distribution_metric(self.distribution_metric)
        if (
            prev_metric != "none"
            and prev_metric != metric
            and self._distribution_metric_is_length(prev_metric) != self._distribution_metric_is_length(metric)
        ):
            self.distribution_edges = []
            self.view.set_distribution_edges_text("")
        self.distribution_metric = metric
        self._sync_distribution_slider_ui()
        self._mark_overlay_dirty()
        self._mark_analysis_dirty()
        self._refresh()

    def on_distribution_bins_changed(self, value: int) -> None:
        self._apply_distribution_bin_count(int(value))

    def on_distribution_bins_increase(self) -> None:
        cur = int(self.view.get_distribution_bins_count())
        self._apply_distribution_bin_count(cur + 1)

    def on_distribution_bins_decrease(self) -> None:
        cur = int(self.view.get_distribution_bins_count())
        self._apply_distribution_bin_count(cur - 1)

    def on_apply_distribution_bins(self) -> None:
        metric = self._normalize_distribution_metric(self.view.get_distribution_metric())
        if metric == "none":
            self.distribution_metric = "none"
            self.distribution_edges = []
            self.view.set_distribution_status("")
            self._mark_overlay_dirty()
            self._mark_analysis_dirty()
            self._refresh()
            return
        edges, err = self._parse_distribution_edges_from_text(self.view.get_distribution_edges_text(), metric)
        if err is not None:
            self.view.set_distribution_status(err, error=True)
            return
        self.distribution_metric = metric
        self.distribution_edges = [float(v) for v in edges]
        self.view.set_distribution_bins_count(max(2, len(self.distribution_edges) - 1))
        self._sync_distribution_slider_ui()
        self._mark_overlay_dirty()
        self._mark_analysis_dirty()
        self._refresh()

    def on_clear_distribution_bins(self) -> None:
        self.distribution_metric = "none"
        self.distribution_edges = []
        self._distribution_slider_min_internal = None
        self._distribution_slider_max_internal = None
        self.view.set_distribution_metric("none")
        self.view.set_distribution_bins_count(3)
        self.view.set_distribution_bins_enabled(False)
        self.view.set_distribution_edges_text("")
        self.view.set_distribution_slider_state(0.0, 1.0, [], enabled=False)
        self.view.set_distribution_status("")
        self._mark_overlay_dirty()
        self._mark_analysis_dirty()
        self._refresh()

    def on_distribution_slider_changed(self, values_display: Sequence[float]) -> None:
        metric = self._normalize_distribution_metric(self.distribution_metric)
        if metric == "none":
            return
        lo_i = self._distribution_slider_min_internal
        hi_i = self._distribution_slider_max_internal
        if lo_i is None or hi_i is None:
            return
        vals_internal: List[float] = []
        for v in values_display:
            iv = self._distribution_value_from_display(metric, float(v))
            if iv is None:
                continue
            vals_internal.append(float(iv))
        vals_internal = sorted(vals_internal)
        span = max(float(hi_i) - float(lo_i), 1e-12)
        eps = span * 1e-6
        cleaned: List[float] = []
        for iv in vals_internal:
            clamped = min(max(float(iv), float(lo_i) + eps), float(hi_i) - eps)
            if cleaned and abs(clamped - cleaned[-1]) <= eps:
                continue
            cleaned.append(clamped)
        self.distribution_edges = [float(lo_i)] + cleaned + [float(hi_i)]
        self.view.set_distribution_bins_count(max(2, len(self.distribution_edges) - 1))
        self.view.set_distribution_edges_text(self._format_distribution_edges_for_ui(metric, self.distribution_edges))
        self.view.set_distribution_status(f"{metric.upper()} bins: {len(self.distribution_edges) - 1}", error=False)

    def on_distribution_slider_released(self) -> None:
        if self._normalize_distribution_metric(self.distribution_metric) == "none":
            return
        self._mark_overlay_dirty()
        self._mark_analysis_dirty()
        self._refresh()

    def _refresh_filter_ui(self) -> None:
        self.view.set_filter_adjustments(self.filter_brightness, self.filter_contrast, self.filter_gamma)
        self.view.set_filter_fft_mode(self.filter_fft_mode)
        self.spatial_filter_chain = self._normalize_spatial_filter_chain(self.spatial_filter_chain)
        self.frequency_filter_chain = self._normalize_frequency_filter_chain(self.frequency_filter_chain)
        self.filter_chain = self._combined_filter_chain()

        if self.spatial_filter_chain:
            self.spatial_filter_selected_row = int(
                np.clip(int(self.spatial_filter_selected_row), 0, len(self.spatial_filter_chain) - 1)
            )
        else:
            self.spatial_filter_selected_row = -1
        if self.frequency_filter_chain:
            self.frequency_filter_selected_row = int(
                np.clip(int(self.frequency_filter_selected_row), 0, len(self.frequency_filter_chain) - 1)
            )
        else:
            self.frequency_filter_selected_row = -1

        self.view.set_spatial_filter_chain_rows(
            [self._filter_step_text(step, idx) for idx, step in enumerate(self.spatial_filter_chain)],
            selected_row=self.spatial_filter_selected_row,
        )
        self.view.set_frequency_filter_chain_rows(
            [self._filter_step_text(step, idx) for idx, step in enumerate(self.frequency_filter_chain)],
            selected_row=self.frequency_filter_selected_row,
        )

        active_step: Optional[Dict[str, Any]] = None
        if self.filter_selected_domain == "frequency":
            if 0 <= self.frequency_filter_selected_row < len(self.frequency_filter_chain):
                active_step = self.frequency_filter_chain[self.frequency_filter_selected_row]
            elif 0 <= self.spatial_filter_selected_row < len(self.spatial_filter_chain):
                self.filter_selected_domain = "spatial"
                active_step = self.spatial_filter_chain[self.spatial_filter_selected_row]
        else:
            if 0 <= self.spatial_filter_selected_row < len(self.spatial_filter_chain):
                active_step = self.spatial_filter_chain[self.spatial_filter_selected_row]
            elif 0 <= self.frequency_filter_selected_row < len(self.frequency_filter_chain):
                self.filter_selected_domain = "frequency"
                active_step = self.frequency_filter_chain[self.frequency_filter_selected_row]
        if active_step is None:
            self.view.clear_filter_editor()
        else:
            self.view.set_filter_editor_from_step(active_step)

    def _on_filter_chain_changed(self, refresh_preview: bool = True, mark_all: bool = False) -> None:
        self.spatial_filter_chain = self._normalize_spatial_filter_chain(self.spatial_filter_chain)
        self.frequency_filter_chain = self._normalize_frequency_filter_chain(self.frequency_filter_chain)
        self.filter_chain = self._combined_filter_chain()
        self._recompute_filtered_for_current()
        if self.filter_input_source == "filtered":
            self._refresh_predictor_image()
            if refresh_preview:
                if self.mode == "polygon" and len(self.polygon_points) >= 3:
                    if mark_all:
                        self._mark_dirty_all()
                    else:
                        self._mark_dirty_current()
                    self._update_polygon_preview()
                    return
                if self.prompt_points or self.prompt_box is not None:
                    if mark_all:
                        self._mark_dirty_all()
                    else:
                        self._mark_dirty_current()
                    self._update_prompt_preview()
                    return
        self._mark_overlay_dirty()
        if mark_all:
            self._mark_dirty_all()
        else:
            self._mark_dirty_current()
        self._refresh()

    def on_filter_source_changed(self, source: str) -> None:
        token = str(source or "").strip().lower()
        self.filter_input_source = "original" if token == "original" else "filtered"
        self.view.set_filter_input_source(self.filter_input_source)
        self._refresh_predictor_image()
        if self.mode == "polygon" and len(self.polygon_points) >= 3:
            self._mark_dirty_current()
            self._update_polygon_preview()
            return
        if self.prompt_points or self.prompt_box is not None:
            self._mark_dirty_current()
            self._update_prompt_preview()
            return
        self._mark_overlay_dirty()
        self._mark_dirty_current()
        self._refresh()

    def on_filter_set_adjustment(self, key: str, value: Any) -> None:
        adjustments = self._current_filter_adjustments()
        token = str(key or "").strip().lower()
        if token == "brightness_ui":
            adjustments["brightness"] = self._brightness_ui_to_internal(value)
        elif token == "contrast_ui":
            adjustments["contrast"] = self._contrast_ui_to_internal(value)
        elif token == "gamma_ui":
            adjustments["gamma"] = self._gamma_ui_to_internal(value)
        else:
            adjustments[token] = value
        norm = self._normalize_filter_adjustments(adjustments)
        self.filter_brightness = int(norm["brightness"])
        self.filter_contrast = float(norm["contrast"])
        self.filter_gamma = float(norm["gamma"])
        self._refresh_filter_ui()
        self._on_filter_chain_changed(refresh_preview=True)

    def on_filter_reset_adjustments(self) -> None:
        self.filter_brightness = 0
        self.filter_contrast = 1.0
        self.filter_gamma = 1.0
        self._refresh_filter_ui()
        self._on_filter_chain_changed(refresh_preview=True)

    def on_filter_select_row(self, domain: str, row: int) -> None:
        self._clear_sym_notch_drag_state()
        dom = "frequency" if str(domain or "").strip().lower() == "frequency" else "spatial"
        if dom == "spatial":
            if not self.spatial_filter_chain:
                self.spatial_filter_selected_row = -1
            else:
                if int(row) < 0:
                    self.spatial_filter_selected_row = -1
                else:
                    self.spatial_filter_selected_row = int(np.clip(int(row), 0, len(self.spatial_filter_chain) - 1))
                    self.filter_selected_domain = "spatial"
        else:
            if not self.frequency_filter_chain:
                self.frequency_filter_selected_row = -1
            else:
                if int(row) < 0:
                    self.frequency_filter_selected_row = -1
                else:
                    self.frequency_filter_selected_row = int(np.clip(int(row), 0, len(self.frequency_filter_chain) - 1))
                    self.filter_selected_domain = "frequency"
        if not self.spatial_filter_chain and not self.frequency_filter_chain:
            self.view.clear_filter_editor()
            return
        self._refresh_filter_ui()

    def on_filter_enter_fft_mode(self) -> None:
        self._clear_sym_notch_drag_state()
        if self.filter_fft_mode:
            return
        self.filter_fft_mode = True
        self.view.set_filter_fft_mode(True)
        self._mark_overlay_dirty()
        self._sync_workspace_ui()
        self._refresh()

    def on_filter_exit_fft_mode(self) -> None:
        self._clear_sym_notch_drag_state()
        if not self.filter_fft_mode:
            return
        self.filter_fft_mode = False
        self.view.set_filter_fft_mode(False)
        self._mark_overlay_dirty()
        self._sync_workspace_ui()
        self._refresh()

    def on_filter_add_step(self, kind: str) -> None:
        step = self._normalize_filter_step({"kind": kind, "params": {}})
        is_frequency = self._normalize_filter_kind(kind) in {"lowpass", "highpass", "bandpass", "sym_notch"}
        if is_frequency:
            idx = int(self.frequency_filter_selected_row)
            if 0 <= idx < len(self.frequency_filter_chain):
                insert_at = idx + 1
                self.frequency_filter_chain.insert(insert_at, step)
                self.frequency_filter_selected_row = insert_at
            else:
                self.frequency_filter_chain.append(step)
                self.frequency_filter_selected_row = len(self.frequency_filter_chain) - 1
            self.filter_selected_domain = "frequency"
        else:
            idx = int(self.spatial_filter_selected_row)
            if 0 <= idx < len(self.spatial_filter_chain):
                insert_at = idx + 1
                self.spatial_filter_chain.insert(insert_at, step)
                self.spatial_filter_selected_row = insert_at
            else:
                self.spatial_filter_chain.append(step)
                self.spatial_filter_selected_row = len(self.spatial_filter_chain) - 1
            self.filter_selected_domain = "spatial"
        self._refresh_filter_ui()
        self._on_filter_chain_changed(refresh_preview=True)

    def on_filter_remove_step(self, domain: Optional[str] = None) -> None:
        dom = str(domain or self.filter_selected_domain or "spatial").strip().lower()
        dom = "frequency" if dom == "frequency" else "spatial"
        chain = self.frequency_filter_chain if dom == "frequency" else self.spatial_filter_chain
        if not chain:
            return
        idx = int(self.frequency_filter_selected_row if dom == "frequency" else self.spatial_filter_selected_row)
        if idx < 0 or idx >= len(chain):
            idx = len(chain) - 1
        chain.pop(idx)
        if chain:
            idx = min(idx, len(chain) - 1)
        else:
            idx = -1
        if dom == "frequency":
            self.frequency_filter_selected_row = idx
        else:
            self.spatial_filter_selected_row = idx
        self.filter_selected_domain = dom
        self._refresh_filter_ui()
        self._on_filter_chain_changed(refresh_preview=True)

    def on_filter_move_up(self) -> None:
        dom = "frequency" if self.filter_selected_domain == "frequency" else "spatial"
        chain = self.frequency_filter_chain if dom == "frequency" else self.spatial_filter_chain
        idx = int(self.frequency_filter_selected_row if dom == "frequency" else self.spatial_filter_selected_row)
        if idx <= 0 or idx >= len(chain):
            return
        chain[idx - 1], chain[idx] = chain[idx], chain[idx - 1]
        if dom == "frequency":
            self.frequency_filter_selected_row = idx - 1
        else:
            self.spatial_filter_selected_row = idx - 1
        self._refresh_filter_ui()
        self._on_filter_chain_changed(refresh_preview=True)

    def on_filter_move_down(self) -> None:
        dom = "frequency" if self.filter_selected_domain == "frequency" else "spatial"
        chain = self.frequency_filter_chain if dom == "frequency" else self.spatial_filter_chain
        idx = int(self.frequency_filter_selected_row if dom == "frequency" else self.spatial_filter_selected_row)
        if idx < 0 or idx >= len(chain) - 1:
            return
        chain[idx + 1], chain[idx] = chain[idx], chain[idx + 1]
        if dom == "frequency":
            self.frequency_filter_selected_row = idx + 1
        else:
            self.spatial_filter_selected_row = idx + 1
        self._refresh_filter_ui()
        self._on_filter_chain_changed(refresh_preview=True)

    def on_filter_rows_moved(self, domain: str, start_row: int, end_row: int, destination_row: int) -> None:
        dom = "frequency" if str(domain or "").strip().lower() == "frequency" else "spatial"
        chain = self.frequency_filter_chain if dom == "frequency" else self.spatial_filter_chain
        if start_row < 0 or end_row < 0 or destination_row < 0:
            return
        if start_row >= len(chain) or end_row >= len(chain):
            return
        if start_row != end_row:
            return
        src = int(start_row)
        dst = int(destination_row)
        step = chain.pop(src)
        if dst > src:
            dst -= 1
        dst = int(np.clip(dst, 0, len(chain)))
        chain.insert(dst, step)
        if dom == "frequency":
            self.frequency_filter_selected_row = dst
        else:
            self.spatial_filter_selected_row = dst
        self.filter_selected_domain = dom
        self._refresh_filter_ui()
        self._on_filter_chain_changed(refresh_preview=True)

    def on_filter_set_enabled(self, enabled: bool) -> None:
        # Legacy no-op: filter enable/disable toggle was removed (filters are always active).
        _ = enabled
        return

    def on_filter_set_param(self, key: str, value: Any) -> None:
        dom = "frequency" if self.filter_selected_domain == "frequency" else "spatial"
        chain = self.frequency_filter_chain if dom == "frequency" else self.spatial_filter_chain
        idx = int(self.frequency_filter_selected_row if dom == "frequency" else self.spatial_filter_selected_row)
        if idx < 0 or idx >= len(chain):
            return
        step = dict(chain[idx])
        params = dict(step.get("params") or {})
        params[str(key)] = value
        step["params"] = params
        chain[idx] = self._normalize_filter_step(step)
        self._refresh_filter_ui()
        self._on_filter_chain_changed(refresh_preview=True)

    def on_filter_apply_current_to_all(self) -> None:
        spatial_chain = self._normalize_spatial_filter_chain(self.spatial_filter_chain)
        frequency_chain = self._normalize_frequency_filter_chain(self.frequency_filter_chain)
        norm_adj = self._normalize_filter_adjustments(
            {
                "brightness": self.filter_brightness,
                "contrast": self.filter_contrast,
                "gamma": self.filter_gamma,
            }
        )
        self.filter_brightness = int(norm_adj["brightness"])
        self.filter_contrast = float(norm_adj["contrast"])
        self.filter_gamma = float(norm_adj["gamma"])
        for state in self.image_sessions:
            state.spatial_filter_chain = copy.deepcopy(spatial_chain)
            state.frequency_filter_chain = copy.deepcopy(frequency_chain)
            state.filter_chain = [*spatial_chain, *frequency_chain]
            state.filter_brightness = int(norm_adj["brightness"])
            state.filter_contrast = float(norm_adj["contrast"])
            state.filter_gamma = float(norm_adj["gamma"])
            self._recompute_filtered_image_for_state(state)
        self.spatial_filter_chain = copy.deepcopy(spatial_chain)
        self.frequency_filter_chain = copy.deepcopy(frequency_chain)
        self.filter_chain = self._combined_filter_chain()
        cur_state = self.image_sessions[self.current_image_idx]
        self.filtered_image_bgr = (
            cur_state.filtered_image_bgr if isinstance(cur_state.filtered_image_bgr, np.ndarray) else cur_state.image_bgr
        )
        if self.filter_input_source == "filtered":
            self._refresh_predictor_image()
            if self.mode == "polygon" and len(self.polygon_points) >= 3:
                self._mark_dirty_all()
                self._update_polygon_preview()
                return
            if self.prompt_points or self.prompt_box is not None:
                self._mark_dirty_all()
                self._update_prompt_preview()
                return
        self._mark_overlay_dirty()
        self._mark_dirty_all()
        self._refresh()

    def on_filter_reset_current(self) -> None:
        self.spatial_filter_chain = self._default_filter_chain()
        self.frequency_filter_chain = []
        self.filter_chain = []
        self.spatial_filter_selected_row = -1
        self.frequency_filter_selected_row = -1
        self.filter_selected_domain = "spatial"
        self._refresh_filter_ui()
        self._on_filter_chain_changed(refresh_preview=True)

    def on_review_sort_changed(self, *_args) -> None:
        self.review_sort_key = self.view.get_review_sort_key()
        self.review_sort_desc = self.view.get_review_sort_desc()
        self._review_cache_key = None
        self._refresh()

    def on_toggle_original_mode(self) -> None:
        if not self._can_edit_in_preview():
            return
        self.use_original_mode = not self.use_original_mode
        if self.current:
            raw = self.current.raw
            cleaned = clean_mask(raw, min_area=20, largest_only=True)
            cleaned = binary_fill_holes(cleaned > 0).astype(np.uint8)
            candidate = cleaned
            if not self.use_original_mode and self.set_masks:
                trimmed = self._apply_no_overlap(cleaned)
                if trimmed.sum() > 0:
                    candidate = trimmed
            if self.use_original_mode:
                candidate = raw
            self.current = MaskEntry(
                candidate,
                raw,
                self.current.score,
                prompt_data=copy.deepcopy(self.current.prompt_data),
            )
            self.current_is_raw_display = self.use_original_mode
            self._clear_analysis_state()
        self._mark_overlay_dirty()
        self._refresh()

    def on_change_mode(self, mode: str) -> None:
        if not self._can_edit_in_preview():
            return
        next_mode = (mode or "sam").lower()
        if next_mode not in ("sam", "lora", "polygon"):
            next_mode = "sam"
        if next_mode == "lora" and not self.lora_mode_available:
            next_mode = "sam"
        self.mode = next_mode
        self.polygon_mode = self.mode == "polygon"
        self.view.set_mode(self.mode)
        self.view.set_action_mode(self.mode)
        self.prompt_points.clear()
        self.prompt_box = None
        self.drag_prompt_box = None
        self.polygon_points.clear()
        self.current = None
        self.current_is_raw_display = False
        self._clear_selection()
        self._mark_overlay_dirty()
        self._sync_workspace_ui()
        self._refresh()

    def on_lora_browse_checkpoint(self) -> None:
        current_text = self.view.get_lora_checkpoint_text().strip()
        start_dir = ""
        if current_text:
            p = Path(current_text).expanduser()
            start_dir = str(p.parent if p.suffix else p)
        elif self.lora_checkpoint_path is not None:
            start_dir = str(self.lora_checkpoint_path.parent)
        else:
            start_dir = str(Path.cwd())
        file_path, _ = QFileDialog.getOpenFileName(
            self.view,
            "Select LoRA checkpoint",
            start_dir,
            "LoRA checkpoint (*.pt *.pth *.bin *.safetensors *.ckpt);;All files (*)",
        )
        if not file_path:
            return
        self.view.set_lora_checkpoint_text(file_path)
        self.on_lora_apply_checkpoint()

    def on_lora_apply_checkpoint(self) -> None:
        path_text = self.view.get_lora_checkpoint_text().strip()
        if not path_text:
            if self.lora_checkpoint_path is None and not self.lora_mode_available:
                self.view.set_lora_runtime_status("", None)
                return
            self.lora_checkpoint_path = None
            self.lora_mode_available = False
            if self.mode == "lora":
                self.mode = "sam"
                self.polygon_mode = False
                self.view.set_mode("sam")
                self.view.set_action_mode("sam")
            self.view.set_lora_mode_enabled(False)
            self.view.set_lora_runtime_status("", None)
            self._sync_workspace_ui()
            self._refresh()
            return

        ckpt = Path(path_text).expanduser()
        if not ckpt.exists() or not ckpt.is_file():
            QMessageBox.warning(self.view, "LoRA Error", f"Checkpoint not found:\n{ckpt}")
            self.view.set_lora_runtime_status("", False)
            return
        try:
            same_as_loaded = (
                self.lora_checkpoint_path is not None
                and self.lora_checkpoint_path.exists()
                and ckpt.resolve() == self.lora_checkpoint_path.resolve()
                and self.lora_mode_available
            )
        except Exception:
            same_as_loaded = False
        if same_as_loaded:
            self.view.set_lora_runtime_status(self.lora_checkpoint_path.name, True)
            return

        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            predictor = self._build_predictor(ckpt)
        except Exception as exc:
            QMessageBox.warning(self.view, "LoRA Error", f"Failed to load LoRA checkpoint:\n{exc}")
            self.view.set_lora_runtime_status("", False)
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        self.predictor = predictor
        self._refresh_predictor_image()
        self.lora_checkpoint_path = ckpt
        self.lora_mode_available = True
        self.view.set_lora_checkpoint_text(str(ckpt))
        self.view.set_lora_mode_enabled(True)
        self.view.set_lora_runtime_status(ckpt.name, True)
        self._sync_workspace_ui()
        if self.prompt_points or self.prompt_box is not None:
            self._update_prompt_preview()
            return
        self._refresh()

    def on_clear_pending_prompts(self) -> None:
        if not self._can_edit_in_preview():
            return
        had_pending = bool(
            self.prompt_points
            or self.prompt_box is not None
            or self.drag_prompt_box is not None
            or self.polygon_points
            or self.scale_bar_points
            or self.scale_bar_px_length is not None
            or self.current is not None
        )
        if not had_pending:
            return
        self.prompt_points.clear()
        self.prompt_box = None
        self.drag_prompt_box = None
        self.polygon_points.clear()
        self.scale_bar_points = []
        self.scale_bar_px_length = None
        self.view.set_scale_calibration_pixels(None)
        self.current = None
        self.current_is_raw_display = False
        self._last_undoable_action = ""
        self._mark_overlay_dirty()
        self._refresh()

    def on_toggle_polygon_mode(self, enabled: bool) -> None:
        self.on_change_mode("polygon" if enabled else "sam")

    def on_remove_selected(self) -> None:
        if not self._can_edit_in_preview():
            return
        indices = sorted(self._valid_selected_indices(), reverse=True)
        if not indices and self.selected_idx is not None and 0 <= self.selected_idx < len(self.set_masks):
            indices = [int(self.selected_idx)]
        if not indices:
            return
        self._push_undo_snapshot()
        for idx in indices:
            if 0 <= idx < len(self.set_masks):
                self.set_masks.pop(idx)
        self._clear_selection()
        self._mark_dirty_current()
        self._mark_overlay_dirty()
        self._auto_calc_if_ready()
        self._refresh()

    def on_resize(self) -> None:
        if QtCore.QThread.currentThread() is not self.view.thread():
            QtCore.QMetaObject.invokeMethod(self, "_flush_resize_refresh", QtCore.Qt.QueuedConnection)
            return
        if self._resize_refresh_pending:
            return
        self._resize_refresh_pending = True
        QtCore.QTimer.singleShot(0, self._flush_resize_refresh)

    @QtCore.Slot()
    def _flush_resize_refresh(self) -> None:
        if self._resize_refresh_pending is False:
            self._resize_refresh_pending = True
        try:
            self._refresh()
        finally:
            self._resize_refresh_pending = False

    def on_app_close(self) -> None:
        if self._train_process is not None and self._train_process.state() != QtCore.QProcess.NotRunning:
            self._train_process.terminate()
            self._train_process.waitForFinished(1200)
            if self._train_process.state() != QtCore.QProcess.NotRunning:
                self._train_process.kill()
        try:
            self._persist_current_session_state()
            self._write_recovery_snapshot()
        except Exception:
            pass

    # Rendering & UI sync ------------------------------------------------------
    def _mask_info_text(self) -> str:
        rec = None
        if self.selected_idx is not None and 0 <= self.selected_idx < len(self.set_masks):
            rec = self.set_masks[self.selected_idx]
        elif self.current:
            rec = self.current
        elif self.set_masks:
            rec = self.set_masks[-1]
        if rec is None:
            if self.polygon_points:
                return f"Polygon vertices: {len(self.polygon_points)} (Set to commit)"
            if self.prompt_points:
                if self.prompt_box is not None:
                    return f"Prompt points: {len(self.prompt_points)} + box (Set to commit)"
                return f"Prompt points: {len(self.prompt_points)} (Set to commit)"
            if self.prompt_box is not None:
                return "Prompt box (Set to commit)"
            return "No mask yet"
        mask = rec.mask.astype(bool)
        coords = np.argwhere(mask)
        score_text = self._score_text(rec.score)
        if coords.size == 0:
            return f"Empty mask (score={score_text})"
        metrics = compute_mask_shape_metrics(rec.mask.astype(np.uint8), self.scale_nm_per_px)
        cx, cy = metrics.get("centroid_px", (None, None))
        area_px = float(metrics.get("area_px", 0.0) or 0.0)
        area_nm2 = float(metrics.get("area_nm2", 0.0) or 0.0)
        area_vesd_nm2 = metrics.get("area_vesd_nm2")
        ecd_nm = float(metrics.get("ecd_nm", 0.0) or 0.0)
        vesd_nm = metrics.get("vesd_nm")
        major_nm = metrics.get("major_axis_nm")
        minor_nm = metrics.get("minor_axis_nm")
        shape_ratio = metrics.get("shape_ratio")
        yy = coords[:, 0]
        xx = coords[:, 1]
        y0, y1 = int(yy.min()), int(yy.max())
        x0, x1 = int(xx.min()), int(xx.max())
        h = y1 - y0 + 1
        w = x1 - x0 + 1
        shape_text = "-"
        if major_nm is not None and minor_nm is not None and shape_ratio is not None:
            shape_text = f"L/S={major_nm:.1f}/{minor_nm:.1f}nm, shape={shape_ratio:.3f}"
        area_vesd_text = "-" if area_vesd_nm2 is None else f"{float(area_vesd_nm2):.0f} nm^2"
        return (
            f"score={score_text} | area={area_px:.0f}px ({area_nm2:.0f} nm^2) | "
            f"area(VESD)={area_vesd_text} | "
            f"ECD={ecd_nm:.1f} nm | VESD={'-' if vesd_nm is None else f'{float(vesd_nm):.1f}'} nm | "
            f"{shape_text} | centroid=({cx:.1f}, {cy:.1f}) px | bbox={w}x{h}px @({x0},{y0})"
        )

    def _build_overlay(self, draw_pairs: bool = True) -> np.ndarray:
        cache_key = (int(self.current_image_idx), int(self._overlay_revision), int(bool(draw_pairs)))
        if self._overlay_cache_key == cache_key and self._overlay_cache is not None:
            return self._overlay_cache
        base_image = self.image_bgr
        if self.workspace_tab == self.WORKSPACE_FILTERS:
            if self.filter_input_source == "filtered" and isinstance(self.filtered_image_bgr, np.ndarray):
                base_image = self.filtered_image_bgr
            else:
                base_image = self.image_bgr
            if self.filter_fft_mode:
                base_image = self._fft_visualize_bgr(base_image)
                out = np.clip(base_image, 0, 255).astype(np.uint8)
                self._draw_active_sym_notch_overlay(out)
                self._overlay_cache_key = cache_key
                self._overlay_cache = out
                return out
        overlay = base_image.copy().astype(np.float32)
        default_alpha = 0.25
        cluster_alpha = 0.35
        distribution_alpha = 0.36
        current_alpha = 0.4
        has_clusters = (
            self.cluster_threshold_nm > 0
            and self.cluster_labels
            and len(self.cluster_labels) == len(self.set_masks)
            and self.cluster_stats.get("count", 0) > 0
        )
        palette = self._cluster_palette(self.cluster_stats.get("count", 0)) if has_clusters else {}
        distribution_assignments, distribution_palette = self._distribution_overlay_assignments()
        distribution_active = (
            self._normalize_distribution_metric(self.distribution_metric) != "none"
            and len(self.distribution_edges) >= 2
            and bool(distribution_palette)
        )
        selected_many = self._valid_selected_indices()
        if self.selected_idx is not None and 0 <= self.selected_idx < len(self.set_masks):
            selected_many.add(int(self.selected_idx))

        def _is_mask_visible(mask_idx: int) -> bool:
            if mask_idx in selected_many:
                return bool(self.show_current_mask)
            return bool(self.show_set_masks)

        for idx, rec in enumerate(self.set_masks):
            if not _is_mask_visible(idx):
                continue
            mask_bool = rec.mask.astype(bool)
            if self.selected_idx is not None and idx == self.selected_idx and self.use_original_mode:
                mask_bool = rec.raw.astype(bool)
            dist_idx = distribution_assignments.get(int(idx)) if distribution_active else None
            if distribution_active:
                if dist_idx is not None and 0 <= int(dist_idx) < len(distribution_palette):
                    dcol = distribution_palette[int(dist_idx)]
                    base = np.array(dcol, dtype=np.float32)
                    overlay[mask_bool] = overlay[mask_bool] * (1.0 - distribution_alpha) + base * distribution_alpha
                else:
                    overlay[mask_bool] = (
                        overlay[mask_bool] * (1.0 - default_alpha)
                        + np.array([0, 200, 0], dtype=np.float32) * default_alpha
                    )
            elif has_clusters:
                cid = self.cluster_labels[idx]
                col = palette.get(cid, (0, 200, 0))
                base = np.array(col, dtype=np.float32)
                overlay[mask_bool] = overlay[mask_bool] * (1.0 - cluster_alpha) + base * cluster_alpha
            else:
                overlay[mask_bool] = (
                    overlay[mask_bool] * (1.0 - default_alpha)
                    + np.array([0, 200, 0], dtype=np.float32) * default_alpha
                )
        if self.current and self.show_current_mask:
            overlay[self.current.mask.astype(bool)] = overlay[self.current.mask.astype(bool)] * (1.0 - current_alpha) + np.array(
                [255, 255, 0], dtype=np.float32
            ) * current_alpha
        if (
            self.show_bbox_overlay
            or self.overlay_use_centroid
            or self.show_axes_overlay
            or self.show_feret_parallelogram_overlay
            or self.show_ellipse_overlay
        ):
            for idx, rec in enumerate(self.set_masks):
                if not _is_mask_visible(idx):
                    continue
                metrics = compute_mask_shape_metrics(rec.mask.astype(np.uint8), self.scale_nm_per_px)
                bx, by, bw, bh = metrics.get("bbox_xywh_px", (None, None, None, None))
                if self.show_bbox_overlay and None not in (bx, by, bw, bh):
                    p0 = (int(bx), int(by))
                    p1 = (int(bx + bw - 1), int(by + bh - 1))
                    if self.selected_idx is not None and idx == int(self.selected_idx):
                        color = (0, 255, 255)
                        thickness = 3
                    else:
                        color = (255, 120, 0)
                        thickness = 2
                    cv2.rectangle(overlay, p0, p1, color, thickness, cv2.LINE_AA)
                if self.overlay_use_centroid:
                    cx, cy = metrics.get("centroid_px", (None, None))
                    if cx is not None and cy is not None:
                        point = (int(round(float(cx))), int(round(float(cy))))
                        if self.selected_idx is not None and idx == int(self.selected_idx):
                            color = (0, 255, 255)
                            radius = 4
                        else:
                            color = (255, 0, 255)
                            radius = 3
                        cv2.circle(overlay, point, radius, color, -1, cv2.LINE_AA)
                segs = None
                if self.show_axes_overlay or self.show_feret_parallelogram_overlay:
                    segs = self._mask_major_minor_segments(rec.mask.astype(np.uint8))
                if segs is not None and (self.show_axes_overlay or self.show_feret_parallelogram_overlay):
                    ma0, ma1, mi0, mi1 = segs
                    pa0 = (int(round(float(ma0[0]))), int(round(float(ma0[1]))))
                    pa1 = (int(round(float(ma1[0]))), int(round(float(ma1[1]))))
                    pi0 = (int(round(float(mi0[0]))), int(round(float(mi0[1]))))
                    pi1 = (int(round(float(mi1[0]))), int(round(float(mi1[1]))))
                    is_selected = self.selected_idx is not None and idx == int(self.selected_idx)
                    if self.show_axes_overlay:
                        if is_selected:
                            major_color = (0, 255, 255)
                            minor_color = (255, 255, 0)
                            thickness = 3
                        else:
                            major_color = (0, 220, 255)
                            minor_color = (255, 160, 0)
                            thickness = 2
                        cv2.line(overlay, pa0, pa1, major_color, thickness, cv2.LINE_AA)
                        cv2.line(overlay, pi0, pi1, minor_color, thickness, cv2.LINE_AA)
                    if self.show_feret_parallelogram_overlay:
                        cx, cy = metrics.get("centroid_px", (None, None))
                        if cx is not None and cy is not None:
                            para_pts = self._feret_parallelogram_from_segments(segs, (float(cx), float(cy)))
                            if para_pts is not None:
                                para_i = np.round(para_pts).astype(np.int32).reshape(-1, 1, 2)
                                para_color = (255, 220, 0) if is_selected else (200, 60, 255)
                                para_thickness = 3 if is_selected else 2
                                cv2.polylines(
                                    overlay,
                                    [para_i],
                                    isClosed=True,
                                    color=para_color,
                                    thickness=para_thickness,
                                    lineType=cv2.LINE_AA,
                                )
                if self.show_ellipse_overlay:
                    ell = mask_ellipse_fit_params_px(rec.mask.astype(np.uint8))
                    if ell is not None:
                        (ecx, ecy), major_len, minor_len, angle_deg = ell
                        center = (int(round(float(ecx))), int(round(float(ecy))))
                        axes = (
                            max(1, int(round(float(major_len) * 0.5))),
                            max(1, int(round(float(minor_len) * 0.5))),
                        )
                        theta = np.deg2rad(float(angle_deg))
                        major_vec = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
                        minor_vec = np.array([-float(major_vec[1]), float(major_vec[0])], dtype=np.float32)
                        major_half = float(major_len) * 0.5
                        minor_half = float(minor_len) * 0.5
                        c = np.array([float(ecx), float(ecy)], dtype=np.float32)
                        ma0 = c - major_vec * major_half
                        ma1 = c + major_vec * major_half
                        mi0 = c - minor_vec * minor_half
                        mi1 = c + minor_vec * minor_half
                        pa0 = (int(round(float(ma0[0]))), int(round(float(ma0[1]))))
                        pa1 = (int(round(float(ma1[0]))), int(round(float(ma1[1]))))
                        pi0 = (int(round(float(mi0[0]))), int(round(float(mi0[1]))))
                        pi1 = (int(round(float(mi1[0]))), int(round(float(mi1[1]))))
                        if self.selected_idx is not None and idx == int(self.selected_idx):
                            ellipse_color = (255, 80, 200)
                            major_color = (255, 40, 255)
                            minor_color = (255, 200, 255)
                            thickness = 3
                        else:
                            ellipse_color = (220, 80, 180)
                            major_color = (255, 0, 200)
                            minor_color = (255, 180, 220)
                            thickness = 2
                        cv2.ellipse(
                            overlay,
                            center,
                            axes,
                            float(angle_deg),
                            0.0,
                            360.0,
                            ellipse_color,
                            thickness,
                            cv2.LINE_AA,
                        )
                        cv2.line(overlay, pa0, pa1, major_color, thickness, cv2.LINE_AA)
                        cv2.line(overlay, pi0, pi1, minor_color, thickness, cv2.LINE_AA)
        if self.show_gt_overlay:
            gt_instances = self._get_eval_gt_instances(
                self.current_image_idx,
                allow_shared_dir=True,
                silent=True,
            )
            if gt_instances:
                gt_line = boundary_from_instances(gt_instances, self.image_bgr.shape[:2], dilate=1)
                # Keep GT ring color distinct from selected-mask highlight (yellow).
                overlay[gt_line] = np.array([255, 0, 255], dtype=np.float32)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        # highlight selected masks with contour (primary: yellow, additional: orange)
        if self.show_current_mask and selected_many:
            primary_idx = int(self.selected_idx) if self.selected_idx is not None else None
            for idx in sorted(selected_many):
                if idx == primary_idx:
                    continue
                msel = self.set_masks[idx].mask
                cnts, _ = cv2.findContours(msel.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, cnts, -1, (0, 165, 255), 2, cv2.LINE_AA)
            if primary_idx is not None and 0 <= primary_idx < len(self.set_masks):
                msel = self.set_masks[primary_idx].mask
                if self.use_original_mode:
                    msel = self.set_masks[primary_idx].raw
                cnts, _ = cv2.findContours(msel.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, cnts, -1, (0, 255, 255), 2, cv2.LINE_AA)

        # prompt markers (SAM points / polygon vertices) can be hidden by UI toggle.
        if self.show_prompts:
            if self.prompt_box is not None:
                x0b, y0b, x1b, y1b = self.prompt_box
                x0i = int(round(min(x0b, x1b)))
                y0i = int(round(min(y0b, y1b)))
                x1i = int(round(max(x0b, x1b)))
                y1i = int(round(max(y0b, y1b)))
                cv2.rectangle(overlay, (x0i, y0i), (x1i, y1i), (255, 200, 0), 2, cv2.LINE_AA)
            if self.drag_prompt_box is not None:
                x0d, y0d, x1d, y1d = self.drag_prompt_box
                dx0 = int(round(min(x0d, x1d)))
                dy0 = int(round(min(y0d, y1d)))
                dx1 = int(round(max(x0d, x1d)))
                dy1 = int(round(max(y0d, y1d)))
                cv2.rectangle(overlay, (dx0, dy0), (dx1, dy1), (255, 255, 0), 1, cv2.LINE_AA)
            for px, py, lbl in self.prompt_points:
                color = (0, 255, 0) if lbl == 1 else (0, 0, 255)
                cv2.circle(overlay, (int(px), int(py)), 4, color, -1, cv2.LINE_AA)
            if self.polygon_points:
                poly_pts = np.array(
                    [[int(round(px)), int(round(py))] for px, py in self.polygon_points],
                    dtype=np.int32,
                ).reshape(-1, 1, 2)
                if len(self.polygon_points) >= 2:
                    is_closed = len(self.polygon_points) >= 3
                    cv2.polylines(overlay, [poly_pts], isClosed=is_closed, color=(255, 200, 0), thickness=2, lineType=cv2.LINE_AA)
                for px, py in self.polygon_points:
                    cv2.circle(overlay, (int(round(px)), int(round(py))), 3, (255, 255, 0), -1, cv2.LINE_AA)

        if self.scale_calibration_mode or self.scale_bar_points:
            pts = [(int(round(px)), int(round(py))) for px, py in self.scale_bar_points]
            if pts:
                cv2.circle(overlay, pts[0], 4, (0, 220, 255), -1, cv2.LINE_AA)
            if len(pts) >= 2:
                cv2.circle(overlay, pts[1], 4, (0, 220, 255), -1, cv2.LINE_AA)
                cv2.line(overlay, pts[0], pts[1], (0, 220, 255), 2, cv2.LINE_AA)
                px_len = self.scale_bar_px_length
                if px_len is None:
                    px_len = float(np.hypot(float(pts[1][0] - pts[0][0]), float(pts[1][1] - pts[0][1])))
                mx = int(round((pts[0][0] + pts[1][0]) * 0.5))
                my = int(round((pts[0][1] + pts[1][1]) * 0.5))
                text = f"{px_len:.1f}px"
                cv2.putText(overlay, text, (mx + 6, my - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(overlay, text, (mx + 6, my - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # collect edge overlaps (by mask index) per rank
        edge_counts = {0: Counter(), 1: Counter()}
        draw_key = "nearest_centroid" if self.overlay_use_centroid else "nearest"
        for i, res in enumerate(self.pair_results):
            near = res.get(draw_key, [])
            for k, item in enumerate(near):
                if k > 1:
                    break
                j = item.get("index")
                if j is None:
                    continue
                j_idx = int(j)
                if not (0 <= j_idx < len(self.set_masks)):
                    continue
                if not (_is_mask_visible(i) and _is_mask_visible(j_idx)):
                    continue
                edge = tuple(sorted((i, j_idx)))
                edge_counts[k][edge] += 1

        if draw_pairs and self.pair_results and self.show_nearest_links and (self.show_set_masks or self.show_current_mask):
            for idx, res in enumerate(self.pair_results):
                key = "nearest_centroid" if self.overlay_use_centroid else "nearest"
                for k, item in enumerate(res.get(key, [])):
                    j_raw = item.get("index", -1)
                    try:
                        j_idx = int(j_raw)
                    except Exception:
                        continue
                    if not (0 <= j_idx < len(self.set_masks)):
                        continue
                    if not (_is_mask_visible(idx) and _is_mask_visible(j_idx)):
                        continue
                    if self.overlay_use_centroid:
                        pa = (int(round(item["centroid_a"][0])), int(round(item["centroid_a"][1])))
                        pb = (int(round(item["centroid_b"][0])), int(round(item["centroid_b"][1])))
                    else:
                        pa = (int(round(item["contour_point_a"][1])), int(round(item["contour_point_a"][0])))
                        pb = (int(round(item["contour_point_b"][1])), int(round(item["contour_point_b"][0])))
                    if k == 0:
                        start = FIRST_COLOR
                        end = (255, 80, 0)
                    else:
                        start = SECOND_COLOR
                        end = (0, 60, 180)
                    edge = tuple(sorted((idx, j_idx)))
                    same_rank_multi = edge_counts.get(k, {}).get(edge, 0) > 1
                    cross_rank_overlap = False
                    if k == 0:
                        cross_rank_overlap = edge in edge_counts.get(1, {})
                    elif k == 1:
                        cross_rank_overlap = edge in edge_counts.get(0, {})
                    if same_rank_multi and not cross_rank_overlap:
                        # same-rank overlap → draw solid color
                        draw_gradient_line(overlay, pa, pb, start, start)
                    else:
                        draw_gradient_line(overlay, pa, pb, start, end)
        self._overlay_cache_key = cache_key
        self._overlay_cache = overlay
        return overlay

    def _render_graphs_composite(self, analysis_payload: Optional[Dict[str, Any]] = None) -> Optional[QtGui.QPixmap]:
        payload = analysis_payload if isinstance(analysis_payload, dict) else self._analysis_scope_payload()
        scope = str(payload.get("scope", "current"))
        mask_count = int(payload.get("mask_count", len(self.set_masks)))
        summary_data = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
        size_data = payload.get("size_summary", {}) if isinstance(payload.get("size_summary"), dict) else {}
        selected_markers = self._selected_hist_markers(payload)
        dist_style = self._distribution_hist_style()
        nearest_hist_mode = self._normalize_nearest_hist_metric(self.nearest_hist_metric)
        size_hist_mode = self._normalize_size_hist_metric(self.size_hist_metric)
        area_hist_mode = self._normalize_area_hist_metric(self.area_hist_metric)
        aspect_hist_mode = self._normalize_aspect_hist_metric(self.aspect_hist_metric)
        selected_primary = int(self.selected_idx) if self.selected_idx is not None else -1
        selected_key = (
            selected_primary,
            None if selected_markers.get("nearest1") is None else round(float(selected_markers.get("nearest1")), 6),
            None if selected_markers.get("nearest2") is None else round(float(selected_markers.get("nearest2")), 6),
            None if selected_markers.get("ecd") is None else round(float(selected_markers.get("ecd")), 6),
            None if selected_markers.get("vesd") is None else round(float(selected_markers.get("vesd")), 6),
            None if selected_markers.get("area") is None else round(float(selected_markers.get("area")), 6),
            None if selected_markers.get("area_bbox") is None else round(float(selected_markers.get("area_bbox")), 6),
            None if selected_markers.get("area_vesd") is None else round(float(selected_markers.get("area_vesd")), 6),
            None if selected_markers.get("aspect") is None else round(float(selected_markers.get("aspect")), 6),
            None if selected_markers.get("aspect_ellipse") is None else round(float(selected_markers.get("aspect_ellipse")), 6),
            nearest_hist_mode,
            aspect_hist_mode,
        )
        w = self.view.graphs_label.width() if self.view.graphs_label.width() > 0 else 640
        fixed_graph_h = int(getattr(self.view, "_graphs_fixed_height", 560))
        total_h = self.view.graphs_label.height() if self.view.graphs_label.height() > 0 else fixed_graph_h
        cache_key = (
            scope,
            int(self.current_image_idx),
            int(self._analysis_revision),
            int(mask_count),
            selected_key,
            int(w),
            int(total_h),
        )
        if self._graphs_cache_key == cache_key and self._graphs_cache_pixmap is not None:
            return self._graphs_cache_pixmap
        total_h = max(fixed_graph_h, int(total_h))
        h_top = max(1, int(round(total_h * 0.34)))
        h_mid = max(1, int(round(total_h * 0.33)))
        h_bot = max(1, int(total_h - h_top - h_mid))
        gap = 10
        col_w = max(200, (int(w) - gap) // 2)
        col_w2 = max(200, int(w) - gap - col_w)
        bg_color = (248, 250, 252)

        img_top = pixmap_to_rgb(
            self._render_histogram_pixmap(
                summary_data=summary_data,
                mask_count=mask_count,
                selected_nearest1=selected_markers.get("nearest1"),
                selected_nearest2=selected_markers.get("nearest2"),
                display_mode=nearest_hist_mode,
                width=int(w),
                height=h_top,
            )
        )
        img_ecd = pixmap_to_rgb(
            self._render_ecd_histogram_pixmap(
                size_summary_data=size_data,
                mask_count=mask_count,
                selected_ecd=selected_markers.get("ecd"),
                selected_vesd=selected_markers.get("vesd"),
                display_mode=size_hist_mode,
                distribution_metric=dist_style.get("metric", "none"),
                distribution_edges=dist_style.get("edges", []),
                distribution_colors=dist_style.get("colors", []),
                width=col_w,
                height=h_mid,
            )
        )
        img_area = pixmap_to_rgb(
            self._render_area_histogram_pixmap(
                size_summary_data=size_data,
                mask_count=mask_count,
                selected_area=selected_markers.get("area"),
                selected_bbox_area=selected_markers.get("area_bbox"),
                selected_area_vesd=selected_markers.get("area_vesd"),
                display_mode=area_hist_mode,
                width=col_w2,
                height=h_mid,
            )
        )
        img_fractal = pixmap_to_rgb(
            self._render_fractal_curve_pixmap(
                size_summary_data=size_data,
                width=col_w,
                height=h_bot,
            )
        )
        img_aspect = pixmap_to_rgb(
            self._render_aspect_histogram_pixmap(
                size_summary_data=size_data,
                mask_count=mask_count,
                selected_aspect=selected_markers.get("aspect"),
                selected_aspect_ellipse=selected_markers.get("aspect_ellipse"),
                display_mode=aspect_hist_mode,
                distribution_metric=dist_style.get("metric", "none"),
                distribution_edges=dist_style.get("edges", []),
                distribution_colors=dist_style.get("colors", []),
                width=col_w2,
                height=h_bot,
            )
        )

        if img_top is None and img_ecd is None and img_area is None and img_fractal is None and img_aspect is None:
            return None

        top_row = fit_rgb_to_cell(img_top, int(w), h_top, bg_color)
        gap_mid = np.full((h_mid, gap, 3), bg_color, dtype=np.uint8)
        mid_left = fit_rgb_to_cell(img_ecd, col_w, h_mid, bg_color)
        mid_right = fit_rgb_to_cell(img_area, col_w2, h_mid, bg_color)
        mid_row = np.concatenate([mid_left, gap_mid, mid_right], axis=1)
        gap_bot = np.full((h_bot, gap, 3), bg_color, dtype=np.uint8)
        bot_left = fit_rgb_to_cell(img_fractal, col_w, h_bot, bg_color)
        bot_right = fit_rgb_to_cell(img_aspect, col_w2, h_bot, bg_color)
        bot_row = np.concatenate([bot_left, gap_bot, bot_right], axis=1)
        composite = np.vstack([top_row, mid_row, bot_row])
        pix = rgb_to_qpixmap(composite)
        self._graphs_cache_key = cache_key
        self._graphs_cache_pixmap = pix
        return pix

    def _render_main_graph_pixmap(
        self,
        *,
        metric: str,
        summary_data: Dict[str, Any],
        size_data: Dict[str, Any],
        mask_count: int,
        selected_markers: Dict[str, Optional[float]],
        dist_style: Dict[str, Any],
        width: int,
        height: int,
    ) -> Optional[QtGui.QPixmap]:
        mode = self._normalize_main_graph_metric(metric)
        if mode in {"nearest1", "nearest2"}:
            return self._render_histogram_pixmap(
                summary_data=summary_data,
                mask_count=mask_count,
                selected_nearest1=selected_markers.get("nearest1"),
                selected_nearest2=selected_markers.get("nearest2"),
                display_mode=mode,
                width=width,
                height=height,
            )
        if mode in {"ecd", "vesd"}:
            return self._render_ecd_histogram_pixmap(
                size_summary_data=size_data,
                mask_count=mask_count,
                selected_ecd=selected_markers.get("ecd"),
                selected_vesd=selected_markers.get("vesd"),
                display_mode=mode,
                distribution_metric=dist_style.get("metric", "none"),
                distribution_edges=dist_style.get("edges", []),
                distribution_colors=dist_style.get("colors", []),
                width=width,
                height=height,
            )
        if mode in {"area", "bbox", "area_vesd"}:
            area_mode = "vesd" if mode == "area_vesd" else mode
            return self._render_area_histogram_pixmap(
                size_summary_data=size_data,
                mask_count=mask_count,
                selected_area=selected_markers.get("area"),
                selected_bbox_area=selected_markers.get("area_bbox"),
                selected_area_vesd=selected_markers.get("area_vesd"),
                display_mode=area_mode,
                width=width,
                height=height,
            )
        if mode in {"aspect_feret", "aspect_ellipse"}:
            aspect_mode = "ellipse" if mode == "aspect_ellipse" else "feret"
            return self._render_aspect_histogram_pixmap(
                size_summary_data=size_data,
                mask_count=mask_count,
                selected_aspect=selected_markers.get("aspect"),
                selected_aspect_ellipse=selected_markers.get("aspect_ellipse"),
                display_mode=aspect_mode,
                distribution_metric=dist_style.get("metric", "none"),
                distribution_edges=dist_style.get("edges", []),
                distribution_colors=dist_style.get("colors", []),
                width=width,
                height=height,
            )
        return self._render_fractal_curve_pixmap(
            size_summary_data=size_data,
            width=width,
            height=height,
        )

    def _render_graph_panels(self, analysis_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Optional[QtGui.QPixmap]]:
        payload = analysis_payload if isinstance(analysis_payload, dict) else self._analysis_scope_payload()
        scope = str(payload.get("scope", "current"))
        mask_count = int(payload.get("mask_count", len(self.set_masks)))
        summary_data = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
        size_data = payload.get("size_summary", {}) if isinstance(payload.get("size_summary"), dict) else {}
        selected_markers = self._selected_hist_markers(payload)
        dist_style = self._distribution_hist_style()
        nearest_hist_mode = self._normalize_nearest_hist_metric(self.nearest_hist_metric)
        size_hist_mode = self._normalize_size_hist_metric(self.size_hist_metric)
        area_hist_mode = self._normalize_area_hist_metric(self.area_hist_metric)
        aspect_hist_mode = self._normalize_aspect_hist_metric(self.aspect_hist_metric)
        main_graph_mode = self._normalize_main_graph_metric(self.main_graph_metric)
        selected_primary = int(self.selected_idx) if self.selected_idx is not None else -1
        main_w = self.view.graphs_main_label.width() if self.view.graphs_main_label.width() > 0 else 640
        main_h = self.view.graphs_main_label.height() if self.view.graphs_main_label.height() > 0 else 300
        top_w = self.view.graphs_top_label.width() if self.view.graphs_top_label.width() > 0 else 640
        top_h = self.view.graphs_top_label.height() if self.view.graphs_top_label.height() > 0 else int(self._graphs_fixed_height * 0.34)
        size_w = self.view.graphs_size_label.width() if self.view.graphs_size_label.width() > 0 else 320
        size_h = self.view.graphs_size_label.height() if self.view.graphs_size_label.height() > 0 else int(self._graphs_fixed_height * 0.33)
        area_w = self.view.graphs_area_label.width() if self.view.graphs_area_label.width() > 0 else 320
        area_h = self.view.graphs_area_label.height() if self.view.graphs_area_label.height() > 0 else int(self._graphs_fixed_height * 0.33)
        fractal_w = self.view.graphs_fractal_label.width() if self.view.graphs_fractal_label.width() > 0 else 320
        fractal_h = self.view.graphs_fractal_label.height() if self.view.graphs_fractal_label.height() > 0 else int(self._graphs_fixed_height * 0.33)
        aspect_w = self.view.graphs_aspect_label.width() if self.view.graphs_aspect_label.width() > 0 else 320
        aspect_h = self.view.graphs_aspect_label.height() if self.view.graphs_aspect_label.height() > 0 else int(self._graphs_fixed_height * 0.33)
        cache_key = (
            scope,
            int(self.current_image_idx),
            int(self._analysis_revision),
            int(mask_count),
            selected_primary,
            None if selected_markers.get("nearest1") is None else round(float(selected_markers.get("nearest1")), 6),
            None if selected_markers.get("nearest2") is None else round(float(selected_markers.get("nearest2")), 6),
            None if selected_markers.get("ecd") is None else round(float(selected_markers.get("ecd")), 6),
            None if selected_markers.get("vesd") is None else round(float(selected_markers.get("vesd")), 6),
            None if selected_markers.get("area") is None else round(float(selected_markers.get("area")), 6),
            None if selected_markers.get("area_bbox") is None else round(float(selected_markers.get("area_bbox")), 6),
            None if selected_markers.get("area_vesd") is None else round(float(selected_markers.get("area_vesd")), 6),
            None if selected_markers.get("aspect") is None else round(float(selected_markers.get("aspect")), 6),
            None if selected_markers.get("aspect_ellipse") is None else round(float(selected_markers.get("aspect_ellipse")), 6),
            nearest_hist_mode,
            size_hist_mode,
            area_hist_mode,
            aspect_hist_mode,
            main_graph_mode,
            str(self.display_length_unit),
            tuple(dist_style.get("edges", [])),
            str(dist_style.get("metric", "none")),
            int(main_w),
            int(main_h),
            int(top_w),
            int(top_h),
            int(size_w),
            int(size_h),
            int(area_w),
            int(area_h),
            int(fractal_w),
            int(fractal_h),
            int(aspect_w),
            int(aspect_h),
        )
        if self._graph_panels_cache_key == cache_key and self._graph_panels_cache is not None:
            return self._graph_panels_cache

        main_pix = self._render_main_graph_pixmap(
            metric=main_graph_mode,
            summary_data=summary_data,
            size_data=size_data,
            mask_count=mask_count,
            selected_markers=selected_markers,
            dist_style=dist_style,
            width=main_w,
            height=main_h,
        )
        top_pix = self._render_histogram_pixmap(
            summary_data=summary_data,
            mask_count=mask_count,
            selected_nearest1=selected_markers.get("nearest1"),
            selected_nearest2=selected_markers.get("nearest2"),
            display_mode=nearest_hist_mode,
            width=top_w,
            height=top_h,
        )
        size_pix = self._render_ecd_histogram_pixmap(
            size_summary_data=size_data,
            mask_count=mask_count,
            selected_ecd=selected_markers.get("ecd"),
            selected_vesd=selected_markers.get("vesd"),
            display_mode=size_hist_mode,
            distribution_metric=dist_style.get("metric", "none"),
            distribution_edges=dist_style.get("edges", []),
            distribution_colors=dist_style.get("colors", []),
            width=size_w,
            height=size_h,
        )
        area_pix = self._render_area_histogram_pixmap(
            size_summary_data=size_data,
            mask_count=mask_count,
            selected_area=selected_markers.get("area"),
            selected_bbox_area=selected_markers.get("area_bbox"),
            selected_area_vesd=selected_markers.get("area_vesd"),
            display_mode=area_hist_mode,
            width=area_w,
            height=area_h,
        )
        fractal_pix = self._render_fractal_curve_pixmap(
            size_summary_data=size_data,
            width=fractal_w,
            height=fractal_h,
        )
        aspect_pix = self._render_aspect_histogram_pixmap(
            size_summary_data=size_data,
            mask_count=mask_count,
            selected_aspect=selected_markers.get("aspect"),
            selected_aspect_ellipse=selected_markers.get("aspect_ellipse"),
            display_mode=aspect_hist_mode,
            distribution_metric=dist_style.get("metric", "none"),
            distribution_edges=dist_style.get("edges", []),
            distribution_colors=dist_style.get("colors", []),
            width=aspect_w,
            height=aspect_h,
        )
        panels = {
            "main": main_pix,
            "top": top_pix,
            "size": size_pix,
            "area": area_pix,
            "fractal": fractal_pix,
            "aspect": aspect_pix,
        }
        self._graph_panels_cache_key = cache_key
        self._graph_panels_cache = panels
        return panels

    def _render_histogram_pixmap(
        self,
        summary_data: Optional[Dict[str, Any]] = None,
        mask_count: Optional[int] = None,
        selected_nearest1: Optional[float] = None,
        selected_nearest2: Optional[float] = None,
        display_mode: str = "nearest1",
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Optional[QtGui.QPixmap]:
        effective_mask_count = int(mask_count) if mask_count is not None else len(self.set_masks)
        if effective_mask_count < 2:
            return None
        summary = summary_data if isinstance(summary_data, dict) else (self.summary if isinstance(self.summary, dict) else {})
        hist_first = self._convert_length_series_from_nm(summary.get("hist_first", []) if summary else [])
        hist_second = self._convert_length_series_from_nm(summary.get("hist_second", []) if summary else [])
        hist_c1 = self._convert_length_series_from_nm(summary.get("hist_centroid1", []) if summary else [])
        hist_c2 = self._convert_length_series_from_nm(summary.get("hist_centroid2", []) if summary else [])
        if not hist_first and not hist_second:
            return None
        target_w = width or 640
        target_h = height or 320
        img = render_nearest_hist_rgb(
            hist_first,
            hist_second,
            hist_c1,
            hist_c2,
            display_mode=self._normalize_nearest_hist_metric(display_mode),
            selected_first=self._convert_optional_length_from_nm(selected_nearest1),
            selected_second=self._convert_optional_length_from_nm(selected_nearest2),
            width=target_w,
            height=target_h,
        )
        return rgb_to_qpixmap(img)

    def _render_ecd_histogram_pixmap(
        self,
        size_summary_data: Optional[Dict[str, Any]] = None,
        mask_count: Optional[int] = None,
        selected_ecd: Optional[float] = None,
        selected_vesd: Optional[float] = None,
        display_mode: str = "ecd",
        distribution_metric: str = "none",
        distribution_edges: Sequence[float] | None = None,
        distribution_colors: Sequence[str] | None = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Optional[QtGui.QPixmap]:
        effective_mask_count = int(mask_count) if mask_count is not None else len(self.set_masks)
        if effective_mask_count < 1:
            return None
        size_summary = (
            size_summary_data
            if isinstance(size_summary_data, dict)
            else (self.size_summary if isinstance(self.size_summary, dict) else {})
        )
        hist_size = self._convert_length_series_from_nm(size_summary.get("hist_ecd", []) if size_summary else [])
        hist_vesd = self._convert_length_series_from_nm(size_summary.get("hist_vesd", []) if size_summary else [])
        if not hist_size and not hist_vesd:
            return None
        target_w = width or 320
        target_h = height or 220
        metric = self._normalize_distribution_metric(distribution_metric)
        dist_edges = list(distribution_edges or [])
        dist_colors = list(distribution_colors or [])
        if metric in {"ecd", "vesd"}:
            dist_edges = self._convert_length_series_from_nm(dist_edges)
        ecd_edges = dist_edges if metric == "ecd" else None
        ecd_colors = dist_colors if metric == "ecd" else None
        vesd_edges = dist_edges if metric == "vesd" else None
        vesd_colors = dist_colors if metric == "vesd" else None
        img = render_ecd_hist_rgb(
            hist_size,
            hist_vesd=hist_vesd,
            selected_ecd=self._convert_optional_length_from_nm(selected_ecd),
            selected_vesd=self._convert_optional_length_from_nm(selected_vesd),
            display_mode=self._normalize_size_hist_metric(display_mode),
            ecd_interval_edges=ecd_edges,
            ecd_interval_colors=ecd_colors,
            vesd_interval_edges=vesd_edges,
            vesd_interval_colors=vesd_colors,
            width=target_w,
            height=target_h,
        )
        return rgb_to_qpixmap(img)

    def _render_area_histogram_pixmap(
        self,
        size_summary_data: Optional[Dict[str, Any]] = None,
        mask_count: Optional[int] = None,
        selected_area: Optional[float] = None,
        selected_bbox_area: Optional[float] = None,
        selected_area_vesd: Optional[float] = None,
        display_mode: str = "area",
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Optional[QtGui.QPixmap]:
        effective_mask_count = int(mask_count) if mask_count is not None else len(self.set_masks)
        if effective_mask_count < 1:
            return None
        size_summary = (
            size_summary_data
            if isinstance(size_summary_data, dict)
            else (self.size_summary if isinstance(self.size_summary, dict) else {})
        )
        metric = self._normalize_area_hist_metric(display_mode)
        if metric == "bbox":
            hist_area = self._convert_area_series_from_nm2(size_summary.get("hist_bbox_area", []) if size_summary else [])
            selected_value = self._convert_optional_area_from_nm2(selected_bbox_area)
            title = "Area: BBox"
            label = "Area(BBox)"
        elif metric == "vesd":
            hist_area = self._convert_area_series_from_nm2(size_summary.get("hist_area_vesd", []) if size_summary else [])
            selected_value = self._convert_optional_area_from_nm2(selected_area_vesd)
            title = "Area: VESD"
            label = "Area(VESD)"
        else:
            hist_area = self._convert_area_series_from_nm2(size_summary.get("hist_area", []) if size_summary else [])
            selected_value = self._convert_optional_area_from_nm2(selected_area)
            title = "Area: px"
            label = "Area(px)"
        if not hist_area:
            return None
        target_w = width or 320
        target_h = height or 220
        img = render_area_hist_rgb(
            hist_area,
            selected_area=selected_value,
            title=title,
            label=label,
            width=target_w,
            height=target_h,
        )
        return rgb_to_qpixmap(img)

    def _render_aspect_histogram_pixmap(
        self,
        size_summary_data: Optional[Dict[str, Any]] = None,
        mask_count: Optional[int] = None,
        selected_aspect: Optional[float] = None,
        selected_aspect_ellipse: Optional[float] = None,
        display_mode: str = "feret",
        distribution_metric: str = "none",
        distribution_edges: Sequence[float] | None = None,
        distribution_colors: Sequence[str] | None = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Optional[QtGui.QPixmap]:
        effective_mask_count = int(mask_count) if mask_count is not None else len(self.set_masks)
        if effective_mask_count < 1:
            return None
        size_summary = (
            size_summary_data
            if isinstance(size_summary_data, dict)
            else (self.size_summary if isinstance(self.size_summary, dict) else {})
        )
        mode = self._normalize_aspect_hist_metric(display_mode)
        hist_aspect: List[float] = []
        selected_value: Optional[float] = None
        title = "Aspect Ratio: Feret"
        if mode == "ellipse":
            hist_major = size_summary.get("hist_ellipse_major_axis", []) if size_summary else []
            hist_minor = size_summary.get("hist_ellipse_minor_axis", []) if size_summary else []
            if isinstance(hist_major, list) and isinstance(hist_minor, list):
                for ma, mi in zip(hist_major, hist_minor):
                    try:
                        ma_f = float(ma)
                        mi_f = float(mi)
                    except Exception:
                        continue
                    if not np.isfinite(ma_f) or not np.isfinite(mi_f) or mi_f <= 0.0:
                        continue
                    hist_aspect.append(float(ma_f / mi_f))
            selected_value = selected_aspect_ellipse
            title = "Aspect Ratio: Ellipse"
        else:
            raw = size_summary.get("hist_aspect_ratio", []) if size_summary else []
            if isinstance(raw, list):
                hist_aspect = [float(v) for v in raw if v is not None and np.isfinite(float(v))]
            selected_value = selected_aspect
        if not hist_aspect:
            return None
        target_w = width or 320
        target_h = height or 220
        metric = self._normalize_distribution_metric(distribution_metric)
        interval_edges = list(distribution_edges or []) if metric == "aspect" else None
        interval_colors = list(distribution_colors or []) if metric == "aspect" else None
        img = render_aspect_hist_rgb(
            hist_aspect,
            selected_aspect=selected_value,
            title=title,
            interval_edges=interval_edges,
            interval_colors=interval_colors,
            width=target_w,
            height=target_h,
        )
        return rgb_to_qpixmap(img)

    def _render_fractal_curve_pixmap(
        self,
        size_summary_data: Optional[Dict[str, Any]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Optional[QtGui.QPixmap]:
        size_summary = (
            size_summary_data
            if isinstance(size_summary_data, dict)
            else (self.size_summary if isinstance(self.size_summary, dict) else {})
        )
        if not size_summary:
            return None
        multi_series_raw = size_summary.get("fractal_per_image")
        if isinstance(multi_series_raw, list):
            multi_series: List[Dict[str, Any]] = []
            for item in multi_series_raw:
                if not isinstance(item, dict):
                    continue
                eps = item.get("log_eps")
                cnt = item.get("log_counts")
                if not isinstance(eps, list) or not isinstance(cnt, list) or len(eps) != len(cnt) or len(eps) < 2:
                    continue
                multi_series.append(item)
            if multi_series:
                target_w = width or 640
                target_h = height or 200
                img = render_fractal_loglog_multi(multi_series, width=target_w, height=target_h)
                return rgb_to_qpixmap(img)
        gcurve = size_summary.get("fractal_global", {})
        g_eps = gcurve.get("log_eps", [])
        g_counts = gcurve.get("log_counts", [])
        if len(g_eps) < 2 or len(g_counts) < 2:
            return None
        target_w = width or 640
        target_h = height or 200
        img = render_fractal_loglog_curve(g_eps, g_counts, gcurve.get("slope"), width=target_w, height=target_h)
        return rgb_to_qpixmap(img)

    def _update_review_table(self) -> None:
        review_key = (
            int(self.current_image_idx),
            int(self._overlay_revision),
            int(self.selected_idx if self.selected_idx is not None else -1),
            str(self.review_sort_key),
            int(self.review_sort_desc),
            str(self.display_length_unit),
        )
        if self._review_cache_key != review_key:
            records: List[Dict[str, Any]] = []
            for mask_idx, rec in enumerate(self.set_masks):
                m = compute_mask_shape_metrics(rec.mask.astype(np.uint8), self.scale_nm_per_px)
                area_nm2 = float(m.get("area_nm2", 0.0) or 0.0)
                bbox_area_nm2 = m.get("bbox_area_nm2")
                area_vesd_nm2 = m.get("area_vesd_nm2")
                ecd_nm = float(m.get("ecd_nm", 0.0) or 0.0)
                major_nm = m.get("major_axis_nm")
                minor_nm = m.get("minor_axis_nm")
                ellipse_major_nm = m.get("ellipse_major_axis_nm")
                ellipse_minor_nm = m.get("ellipse_minor_axis_nm")
                vesd_nm = m.get("vesd_nm")
                aspect_feret = m.get("aspect_ratio")
                if (
                    ellipse_major_nm is not None
                    and ellipse_minor_nm is not None
                    and float(ellipse_minor_nm) > 0.0
                ):
                    aspect_ellipse = float(ellipse_major_nm) / float(ellipse_minor_nm)
                else:
                    aspect_ellipse = None
                ecd_disp = self._convert_length_from_nm(ecd_nm, self.display_length_unit)
                area_disp = self._convert_area_from_nm2(area_nm2, self.display_length_unit)
                bbox_area_disp = self._convert_area_from_nm2(float(bbox_area_nm2), self.display_length_unit) if bbox_area_nm2 is not None else None
                area_vesd_disp = self._convert_area_from_nm2(float(area_vesd_nm2), self.display_length_unit) if area_vesd_nm2 is not None else None
                major_disp = self._convert_length_from_nm(float(major_nm), self.display_length_unit) if major_nm is not None else None
                minor_disp = self._convert_length_from_nm(float(minor_nm), self.display_length_unit) if minor_nm is not None else None
                ellipse_major_disp = (
                    self._convert_length_from_nm(float(ellipse_major_nm), self.display_length_unit)
                    if ellipse_major_nm is not None
                    else None
                )
                ellipse_minor_disp = (
                    self._convert_length_from_nm(float(ellipse_minor_nm), self.display_length_unit)
                    if ellipse_minor_nm is not None
                    else None
                )
                vesd_disp = self._convert_length_from_nm(float(vesd_nm), self.display_length_unit) if vesd_nm is not None else None
                bbox_x_text = "-"
                bbox_y_text = "-"
                bbox_cx_text = "-"
                bbox_cy_text = "-"
                bx, by, bw, bh = m.get("bbox_xywh_px", (None, None, None, None))
                if None not in (bx, by, bw, bh):
                    x0 = int(round(float(bx)))
                    y0 = int(round(float(by)))
                    bw_i = max(1, int(round(float(bw))))
                    bh_i = max(1, int(round(float(bh))))
                    bbox_x_text = str(x0)
                    bbox_y_text = str(y0)
                    bbox_cx_text = f"{x0 + (bw_i - 1) * 0.5:.1f}"
                    bbox_cy_text = f"{y0 + (bh_i - 1) * 0.5:.1f}"
                cent_x_text = "-"
                cent_y_text = "-"
                cx, cy = m.get("centroid_px", (None, None))
                if cx is not None and cy is not None:
                    cent_x_text = f"{float(cx):.1f}"
                    cent_y_text = f"{float(cy):.1f}"
                score_num = self._score_value(rec.score)
                records.append(
                    {
                        "mask_idx": int(mask_idx),
                        "sort": {
                            "index": float(mask_idx),
                            "ecd": ecd_nm,
                            "area": area_nm2,
                            "bbox_area": float(bbox_area_nm2) if bbox_area_nm2 is not None else None,
                            "area_vesd": float(area_vesd_nm2) if area_vesd_nm2 is not None else None,
                            "major": float(major_nm) if major_nm is not None else None,
                            "minor": float(minor_nm) if minor_nm is not None else None,
                            "ellipse_major": float(ellipse_major_nm) if ellipse_major_nm is not None else None,
                            "ellipse_minor": float(ellipse_minor_nm) if ellipse_minor_nm is not None else None,
                            "vesd": float(vesd_nm) if vesd_nm is not None else None,
                            "aspect": float(aspect_feret) if aspect_feret is not None else None,
                            "aspect_feret": float(aspect_feret) if aspect_feret is not None else None,
                            "aspect_ellipse": float(aspect_ellipse) if aspect_ellipse is not None else None,
                            "score": float(score_num) if score_num is not None else None,
                        },
                        "row": [
                            "-" if ecd_disp is None else f"{float(ecd_disp):.3f}",
                            "-" if vesd_disp is None else f"{float(vesd_disp):.3f}",
                            "-" if area_disp is None else f"{float(area_disp):.3f}",
                            "-" if bbox_area_disp is None else f"{float(bbox_area_disp):.3f}",
                            "-" if area_vesd_disp is None else f"{float(area_vesd_disp):.3f}",
                            "-" if aspect_feret is None else f"{float(aspect_feret):.2f}",
                            "-" if aspect_ellipse is None else f"{float(aspect_ellipse):.2f}",
                            "-" if major_disp is None else f"{float(major_disp):.3f}",
                            "-" if minor_disp is None else f"{float(minor_disp):.3f}",
                            "-" if ellipse_major_disp is None else f"{float(ellipse_major_disp):.3f}",
                            "-" if ellipse_minor_disp is None else f"{float(ellipse_minor_disp):.3f}",
                            cent_x_text,
                            cent_y_text,
                            bbox_x_text,
                            bbox_y_text,
                            bbox_cx_text,
                            bbox_cy_text,
                            self._score_text(rec.score),
                        ],
                    }
                )

            sort_key = str(self.review_sort_key or "index")
            if sort_key not in {
                "index",
                "ecd",
                "area",
                "bbox_area",
                "area_vesd",
                "major",
                "minor",
                "ellipse_major",
                "ellipse_minor",
                "vesd",
                "aspect",
                "aspect_feret",
                "aspect_ellipse",
                "score",
            }:
                sort_key = "index"

            def _sort_value(rec_item: Dict[str, Any]) -> float:
                val = rec_item["sort"].get(sort_key)
                if val is None:
                    return float("-inf") if self.review_sort_desc else float("inf")
                return float(val)

            records.sort(key=_sort_value, reverse=bool(self.review_sort_desc))
            self._review_row_to_mask_idx = [int(item["mask_idx"]) for item in records]
            self._review_cache_rows = [item["row"] for item in records]
            self._review_cache_summary = f"Masks={len(self.set_masks)}"
            self._review_cache_key = review_key
        self.view.set_review_table_unit(self.display_length_unit)
        self.view.set_review_rows(self._review_cache_rows, summary_text=self._review_cache_summary)
        row_by_mask = {mask_idx: row_idx for row_idx, mask_idx in enumerate(self._review_row_to_mask_idx)}
        selected_rows = [row_by_mask[idx] for idx in sorted(self._valid_selected_indices()) if idx in row_by_mask]
        if not selected_rows and self.selected_idx is not None and 0 <= self.selected_idx < len(self.set_masks):
            row = row_by_mask.get(int(self.selected_idx))
            if row is not None:
                selected_rows = [int(row)]
        self.view.set_review_selected_rows(selected_rows)

    @staticmethod
    def _fmt_stat_value(v: Optional[float], digits: int = 2) -> str:
        if v is None:
            return "-"
        return f"{v:.{digits}f}"

    def _update_stats_table(self, analyze_view_active: bool, analysis_payload: Dict[str, Any]) -> None:
        mask_count = int(analysis_payload.get("mask_count", len(self.set_masks)))
        scope_text = "All images" if str(analysis_payload.get("scope", "current")) == "all" else "Current image"
        summary_data = analysis_payload.get("summary", {}) if isinstance(analysis_payload.get("summary"), dict) else {}
        size_data = analysis_payload.get("size_summary", {}) if isinstance(analysis_payload.get("size_summary"), dict) else {}
        self.view.set_stats_summary_text(f"{scope_text} · n={mask_count}")
        if analyze_view_active and (summary_data or size_data):
            len_unit = self._normalize_length_unit(self.display_length_unit)
            columns = [
                ("N1", "summary", "nearest1", "N1", "length"),
                ("N2", "summary", "nearest2", "N2", "length"),
                ("Cent1", "summary", "centroid1", "Cent1", "length"),
                ("Cent2", "summary", "centroid2", "Cent2", "length"),
                ("ECD", "size", "ecd", "ECD", "length"),
                ("VESD", "size", "vesd", "VESD", "length"),
                ("Area", "size", "area", "Area(px)", "area"),
                ("BBoxArea", "size", "bbox_area", "BBox Area", "area"),
                ("AreaV", "size", "area_vesd", "Area(VESD)", "area"),
                ("FeretMajor", "size", "major_axis", "Feret Maj", "length"),
                ("FeretMinor", "size", "minor_axis", "Feret Min", "length"),
                ("AreaFeretRect", "size", "feret_rect_area", "Area(FeretRect)", "area"),
                ("EllipseMajor", "size", "ellipse_major_axis", "Ellipse Maj", "length"),
                ("EllipseMinor", "size", "ellipse_minor_axis", "Ellipse Min", "length"),
                ("Aspect", "size", "aspect_ratio", "Aspect", "none"),
                ("Shape", "size", "shape_ratio", "Shape", "none"),
            ]
            rows_cfg = [
                ("mean", "mean"),
                ("median", "median"),
                ("std", "std"),
                ("cv(%)", "cv_pct"),
                ("min", "min"),
                ("max", "max"),
            ]
            rows = [label for label, _ in rows_cfg]
            vals: List[List[str]] = []
            for _row_label, key in rows_cfg:
                data_map: Dict[str, Optional[float]] = {}
                for col_id, src, src_key, _header, unit_kind in columns:
                    if src == "summary":
                        raw_v = summary_data.get(src_key, {}).get(key) if summary_data else None
                    elif src == "size":
                        raw_v = size_data.get(src_key, {}).get(key) if size_data else None
                    elif src == "scalar":
                        raw_v = size_data.get(src_key) if (size_data and key == "mean") else None
                    else:
                        raw_v = None
                    if raw_v is None:
                        data_map[col_id] = None
                    elif unit_kind == "length":
                        data_map[col_id] = self._convert_length_from_nm(float(raw_v), len_unit)
                    elif unit_kind == "area":
                        data_map[col_id] = self._convert_area_from_nm2(float(raw_v), len_unit)
                    else:
                        data_map[col_id] = float(raw_v)
                vals.append([self._fmt_stat_value(data_map.get(col_id), 2) for col_id, _, _, _, _ in columns])
            headers = [header for _, _, _, header, _ in columns]
            if self._stats_cache_revision != self._analysis_revision:
                self.view.set_stats_table(headers, rows, vals)
                self._stats_cache_revision = self._analysis_revision
        elif analyze_view_active and self._stats_cache_revision != self._analysis_revision:
            self.view.set_stats_table([], [], [])
            self._stats_cache_revision = self._analysis_revision

    def _refresh(self) -> None:
        if QtCore.QThread.currentThread() is not self.view.thread():
            self.on_resize()
            return
        analysis_payload = self._analysis_scope_payload()
        self._sync_cluster_controls()
        self._sync_workspace_ui()
        self._sync_distribution_slider_ui(analysis_payload=analysis_payload)
        self._update_save_state_ui()
        self.view.set_calc_pending(bool(analysis_payload.get("pending", False)))
        self.view.set_workspace_tab(self.workspace_tab)
        img_h, img_w = self.image_bgr.shape[:2]
        self.view.adapt_layout_for_image(img_w, img_h)
        train_preview_active = self.workspace_tab == self.WORKSPACE_TRAIN
        track_preview_active = self.workspace_tab == self.WORKSPACE_TRACK
        if train_preview_active:
            self._ensure_train_preview_overlay()
            if self._train_preview_overlay is not None:
                overlay_full = self._train_preview_overlay
            else:
                overlay_full = self._build_workspace_placeholder(
                    "Train",
                    "Set Train image/mask dirs to import workspace-specific data.",
                )
            overlay = overlay_full
            x0, y0 = 0, 0
        elif track_preview_active:
            self._ensure_track_preview_image()
            if self._track_preview_image is not None:
                overlay_full = self._track_preview_image
            else:
                overlay_full = self._build_workspace_placeholder(
                    "Track",
                    "Import Track image to start a separate workspace preview.",
                )
            overlay = overlay_full
            x0, y0 = 0, 0
        else:
            overlay_full = self._build_overlay(draw_pairs=True)
            x0, y0, vw, vh = self._compute_view_rect()
            overlay = overlay_full[y0 : y0 + vh, x0 : x0 + vw]
            if overlay.size == 0:
                overlay = overlay_full
                y0, x0 = 0, 0
        self._view_x0 = float(x0)
        self._view_y0 = float(y0)
        self._view_h = float(overlay.shape[0])
        self._view_w = float(overlay.shape[1])
        pixmap = bgr_to_qpixmap(overlay)
        # On first refresh (before show), QLabel can be uninitialized and would keep
        # full image size as hint. Clamp to label/min size to avoid oversized buffers.
        content_rect = self.view.image_label.contentsRect()
        label_w = content_rect.width()
        label_h = content_rect.height()
        if label_w <= 1 or label_h <= 1:
            min_sz = self.view.image_label.minimumSize()
            label_w = max(1, min_sz.width())
            label_h = max(1, min_sz.height())
        pixmap = pixmap.scaled(label_w, label_h, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.view.image_label.setPixmap(pixmap)
        disp_w = pixmap.width() if not pixmap.isNull() else 0
        disp_h = pixmap.height() if not pixmap.isNull() else 0
        if disp_w > 0:
            # Keep content width aligned to visible image width, but compensate
            # QLabel frame/border so repeated refresh does not shrink the image.
            frame_w = max(0, int(self.view.image_label.width()) - int(content_rect.width()))
            self.view.set_left_content_width(int(disp_w) + int(frame_w))
        options_side = max(0, (int(label_w) - int(disp_w)) // 2) if disp_w > 0 else 0
        self.view.set_image_header_side_padding(options_side)
        self.view.set_options_image_side_padding(options_side)
        self._disp_size = (disp_w, disp_h)
        if disp_w > 0 and disp_h > 0:
            off_x = int(content_rect.x()) + max(0, (int(content_rect.width()) - disp_w) // 2)
            off_y = int(content_rect.y()) + max(0, (int(content_rect.height()) - disp_h) // 2)
        else:
            off_x, off_y = 0, 0
        self._disp_origin = (off_x, off_y)
        if disp_w > 0 and disp_h > 0:
            self._scale_x = self._view_w / disp_w
            self._scale_y = self._view_h / disp_h
        mode_text = "Polygon mode: Left=vertex, Right=close"
        if self.mode == "sam":
            mode_text = "SAM mode: Left=+, Right=-, drag=box (mask right-click: multi-select), Set to commit"
        elif self.mode == "lora":
            mode_text = "LoRA mode: Left=+, Right=-, drag=box (mask right-click: multi-select), Set to commit"
        info_lines = [
            f"Set masks: {len(self.set_masks)}",
            f"Zoom: {self.zoom_factor:.2f}x",
            f"Scale: {self.scale_nm_per_px:.3f} nm/px",
            f"Max dist: {'unlimited' if self.max_distance_nm is None else f'{self.max_distance_nm:.0f} nm'}",
            mode_text,
        ]
        if train_preview_active:
            info_lines.insert(0, self._train_preview_note)
        elif track_preview_active:
            info_lines.insert(0, self._track_preview_note)
        if self.summary_stale:
            info_lines.append("Stats stale: press Calc")
        if self.current:
            info_lines.append(f"Current score: {self._score_text(self.current.score)}")
        if self.summary:
            n1 = self.summary.get("nearest1", {})
            n2 = self.summary.get("nearest2", {})
            info_lines.append(f"N1 mean={n1.get('mean')}, std={n1.get('std')}")
            info_lines.append(f"N2 mean={n2.get('mean')}, std={n2.get('std')}")
        if self.cluster_stats.get("count"):
            info_lines.append(f"Clusters={self.cluster_stats.get('count')}")
        info_lines.append(f"Include0={self.include_zero_distance}")
        self.view.set_info_text(" | ".join(info_lines))
        if train_preview_active:
            self.view.set_mask_info_text(self._train_preview_note)
        elif track_preview_active:
            self.view.set_mask_info_text(self._track_preview_note)
        else:
            self.view.set_mask_info_text(self._mask_info_text())
        if self.workspace_tab == self.WORKSPACE_TRAIN:
            self.view.output_info_label.setText(f"Train output dir: {self.view.train_output_dir_edit.text().strip() or '-'}")
        elif self.workspace_tab == self.WORKSPACE_ANALYZE and self.view.is_evaluate_tab_active():
            roi_text = self.view.eval_gt_roi_edit.text().strip() if hasattr(self.view, "eval_gt_roi_edit") else ""
            self.view.output_info_label.setText(f"Eval GT ROI: {roi_text or '-'}")
        else:
            self.view.output_info_label.setText(f"Output dir: {self.output_dir}")
        self._update_image_status_ui()

        analyze_view_active = self.workspace_tab == self.WORKSPACE_ANALYZE and self.view.is_analyze_tab_active()
        if analyze_view_active:
            panels = self._render_graph_panels(analysis_payload=analysis_payload)
            self.view.set_graph_panel_pixmaps(
                main=panels.get("main"),
                top=panels.get("top"),
                size=panels.get("size"),
                area=panels.get("area"),
                fractal=panels.get("fractal"),
                aspect=panels.get("aspect"),
            )
        self._update_review_table()
        self._update_stats_table(analyze_view_active, analysis_payload)

    # Launch -------------------------------------------------------------------
    def launch(self) -> None:
        self.view.show()
        self._refresh()

    # Manual calc --------------------------------------------------------------
    def on_calc_current(self) -> None:
        self.analysis_scope = "current"
        self.on_calc()

    def on_calc_all(self) -> None:
        self.analysis_scope = "all"
        self.on_calc()

    def on_calc(self) -> None:
        if self.workspace_tab != self.WORKSPACE_ANALYZE:
            self.workspace_tab = self.WORKSPACE_ANALYZE
            self.view.set_workspace_tab(self.WORKSPACE_ANALYZE)
        self.view.set_preview_analyze_tab("analyze")

        scope = "all" if self.analysis_scope == "all" else "current"

        if scope == "all":
            if not any(len(state.set_masks) > 0 for state in self.image_sessions):
                return
            prev_idx = int(self.current_image_idx)
            self._persist_current_session_state()
            self.view.set_calc_running(True, "Calculating all images...")
            status_text = f"Calc done ({self._now_text()})"
            try:
                for state in self.image_sessions:
                    self._run_calc_for_state(state)
                self._apply_image_session(prev_idx)
                if len(self.set_masks) >= 2:
                    self._compute_clusters()
                else:
                    self.cluster_labels = []
                    self.cluster_stats = {}
                self.summary_stale = False
                self._persist_current_session_state()
            except Exception as exc:
                status_text = "Calc failed"
                QMessageBox.warning(self.view, "Calc Error", f"Calc failed:\n{exc}")
            finally:
                self.view.set_calc_running(False, status_text)
            self._mark_analysis_dirty()
            self._refresh()
            return

        if len(self.set_masks) == 0:
            return
        if len(self.set_masks) < 2:
            self._reset_calc_outputs()
            self._refresh()
            return
        self.summary_stale = True
        self._request_calc_async()

def launch_app(
    image_paths: List[Path],
    output_dir: Path,
    hf_model_id: str = "facebook/sam-vit-base",
    lora_checkpoint: Optional[Path] = None,
    lora_rank: int = 16,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.1,
    mask_threshold: float = 0.5,
    scale_nm_per_px: float = DEFAULT_SCALE,
    max_distance_nm: Optional[float] = None,
    fractal_slides: int = 0,
    init_mask_id_path: Optional[Path] = None,
    init_mask_dir: Optional[Path] = None,
) -> None:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    controller = MaskDistanceController(
        image_paths=image_paths,
        init_mask_id_path=init_mask_id_path,
        init_mask_dir=init_mask_dir,
        output_dir=output_dir,
        hf_model_id=hf_model_id,
        lora_checkpoint=lora_checkpoint,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        mask_threshold=mask_threshold,
        scale_nm_per_px=scale_nm_per_px,
        max_distance_nm=max_distance_nm,
        fractal_slides=fractal_slides,
    )
    controller.launch()
    app.exec()
