from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from microseg.compute import MaskEntry
from microseg.config import DEFAULT_SCALE


@dataclass
class ImageSessionState:
    image_path: Path
    image_bgr: np.ndarray
    output_dir: Path
    scale_nm_per_px: float = DEFAULT_SCALE
    set_masks: List[MaskEntry] = field(default_factory=list)
    current: Optional[MaskEntry] = None
    current_is_raw_display: bool = False
    pair_results: List[Dict] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)
    size_summary: Dict = field(default_factory=dict)
    summary_stale: bool = False
    selected_idx: Optional[int] = None
    selected_indices: set[int] = field(default_factory=set)
    prompt_points: List[Tuple[float, float, int]] = field(default_factory=list)
    prompt_box: Optional[Tuple[float, float, float, float]] = None
    polygon_points: List[Tuple[float, float]] = field(default_factory=list)
    cluster_labels: List[int] = field(default_factory=list)
    cluster_stats: Dict = field(default_factory=dict)
    undo_stack: List[Dict[str, Any]] = field(default_factory=list)
    view_center_x: float = 0.0
    view_center_y: float = 0.0
    zoom_factor: float = 1.0
    dirty: bool = False
    filter_brightness: int = 0
    filter_contrast: float = 1.0
    filter_gamma: float = 1.0
    spatial_filter_chain: List[Dict[str, Any]] = field(default_factory=list)
    frequency_filter_chain: List[Dict[str, Any]] = field(default_factory=list)
    spatial_filter_selected_row: int = -1
    frequency_filter_selected_row: int = -1
    filter_selected_domain: str = "spatial"
    filter_chain: List[Dict[str, Any]] = field(default_factory=list)
    filtered_image_bgr: Optional[np.ndarray] = None
    filter_fft_mode: bool = False
