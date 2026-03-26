from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

from PySide6 import QtCore, QtGui, QtWidgets
from microseg.config import MAGNIFICATION_PRESETS


class ClickableLabel(QtWidgets.QLabel):
    clicked = QtCore.Signal(int, int)
    right_clicked = QtCore.Signal(int, int)
    box_dragging = QtCore.Signal(int, int, int, int)
    box_dragged = QtCore.Signal(int, int, int, int)
    wheel_scrolled = QtCore.Signal(int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._press_pos: Optional[QtCore.QPoint] = None
        self._press_button: Optional[QtCore.Qt.MouseButton] = None
        self._dragging = False
        self._drag_threshold_px = 4

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        self._press_pos = event.position().toPoint()
        self._press_button = event.button()
        self._dragging = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if self._press_button == QtCore.Qt.LeftButton and self._press_pos is not None:
            cur = event.position().toPoint()
            if (cur - self._press_pos).manhattanLength() >= self._drag_threshold_px:
                self._dragging = True
            if self._dragging:
                self.box_dragging.emit(self._press_pos.x(), self._press_pos.y(), cur.x(), cur.y())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        release_pt = event.position().toPoint()
        button = event.button()
        press_pt = self._press_pos
        press_button = self._press_button
        dragging = self._dragging
        self._press_pos = None
        self._press_button = None
        self._dragging = False

        if button == QtCore.Qt.LeftButton and press_button == QtCore.Qt.LeftButton:
            if dragging and press_pt is not None:
                self.box_dragged.emit(press_pt.x(), press_pt.y(), release_pt.x(), release_pt.y())
            else:
                self.clicked.emit(release_pt.x(), release_pt.y())
        elif button == QtCore.Qt.RightButton and press_button == QtCore.Qt.RightButton:
            self.right_clicked.emit(release_pt.x(), release_pt.y())
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:  # type: ignore[override]
        delta = int(event.angleDelta().y())
        if delta != 0:
            pt = event.position().toPoint()
            self.wheel_scrolled.emit(delta, pt.x(), pt.y())
            event.accept()
            return
        super().wheelEvent(event)


class ScalePresetRuler(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._ticks: list[tuple[float, str]] = []
        self.setMinimumHeight(28)
        self.setMaximumHeight(28)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    def set_ticks(self, ticks: Sequence[tuple[float, str]]) -> None:
        normalized: list[tuple[float, str]] = []
        for pos, label in ticks:
            try:
                p = float(pos)
            except Exception:
                continue
            p = max(0.0, min(1.0, p))
            normalized.append((p, str(label)))
        self._ticks = normalized
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        super().paintEvent(event)
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        r = self.rect().adjusted(8, 1, -8, -1)
        if r.width() <= 4:
            p.end()
            return
        base_y = 8
        p.setPen(QtGui.QPen(QtGui.QColor("#c7d4e7"), 1))
        p.drawLine(r.left(), base_y, r.right(), base_y)
        tick_pen = QtGui.QPen(QtGui.QColor("#8ea4c6"), 1)
        text_pen = QtGui.QPen(QtGui.QColor("#607086"), 1)
        fm = p.fontMetrics()
        for pos, label in self._ticks:
            x = int(round(r.left() + pos * r.width()))
            p.setPen(tick_pen)
            p.drawLine(x, base_y - 3, x, base_y + 4)
            p.setPen(text_pen)
            tw = fm.horizontalAdvance(label)
            p.drawText(x - tw // 2, 24, label)
        p.end()


class MultiHandleSlider(QtWidgets.QWidget):
    valuesChanged = QtCore.Signal(list)
    editingFinished = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._minimum = 0.0
        self._maximum = 1.0
        self._values: list[float] = []
        self._active_idx: Optional[int] = None
        self._dragging = False
        self.setMinimumHeight(44)
        self.setMaximumHeight(44)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.setCursor(QtCore.Qt.PointingHandCursor)

    @staticmethod
    def _format_handle_value(value: float) -> str:
        try:
            v = float(value)
        except Exception:
            return "-"
        if not math.isfinite(v):
            return "-"
        av = abs(v)
        if av >= 100.0:
            return f"{v:.0f}"
        if av >= 1.0:
            return f"{v:.1f}"
        return f"{v:.3f}"

    def setRange(self, minimum: float, maximum: float) -> None:
        try:
            lo = float(minimum)
            hi = float(maximum)
        except Exception:
            lo, hi = 0.0, 1.0
        if not math.isfinite(lo):
            lo = 0.0
        if not math.isfinite(hi):
            hi = lo + 1.0
        if hi <= lo:
            hi = lo + max(abs(lo) * 1e-3, 1e-6)
        self._minimum = lo
        self._maximum = hi
        self._values = self._clamp_values(self._values)
        self.update()

    def setValues(self, values: Sequence[float]) -> None:
        self._values = self._clamp_values(values)
        self.update()

    def values(self) -> list[float]:
        return [float(v) for v in self._values]

    def _track_geometry(self) -> tuple[int, int, int]:
        r = self.rect().adjusted(10, 0, -10, 0)
        cy = int(r.center().y())
        return int(r.left()), int(r.right()), cy

    def _x_from_value(self, value: float) -> int:
        left, right, _ = self._track_geometry()
        span = max(self._maximum - self._minimum, 1e-12)
        t = (float(value) - self._minimum) / span
        t = max(0.0, min(1.0, t))
        return int(round(left + t * float(max(1, right - left))))

    def _value_from_x(self, x: int) -> float:
        left, right, _ = self._track_geometry()
        if right <= left:
            return float(self._minimum)
        t = (float(x) - float(left)) / float(right - left)
        t = max(0.0, min(1.0, t))
        return float(self._minimum + t * (self._maximum - self._minimum))

    def _clamp_values(self, values: Sequence[float]) -> list[float]:
        out: list[float] = []
        span = max(self._maximum - self._minimum, 1e-12)
        eps = span * 1e-6
        for v in values:
            try:
                fv = float(v)
            except Exception:
                continue
            if not math.isfinite(fv):
                continue
            fv = min(max(fv, self._minimum + eps), self._maximum - eps)
            out.append(fv)
        out.sort()
        uniq: list[float] = []
        for fv in out:
            if not uniq:
                uniq.append(fv)
                continue
            if abs(fv - uniq[-1]) <= eps:
                continue
            uniq.append(fv)
        return uniq

    def _pick_handle_index(self, x: int, y: int) -> Optional[int]:
        if not self._values:
            return None
        _left, _right, cy = self._track_geometry()
        if abs(int(y) - int(cy)) > 12:
            return None
        positions = [self._x_from_value(v) for v in self._values]
        if not positions:
            return None
        dists = [abs(int(px) - int(x)) for px in positions]
        idx = int(min(range(len(dists)), key=lambda i: dists[i]))
        return idx

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if not self.isEnabled() or event.button() != QtCore.Qt.LeftButton:
            super().mousePressEvent(event)
            return
        pos = event.position().toPoint()
        idx = self._pick_handle_index(pos.x(), pos.y())
        if idx is None:
            super().mousePressEvent(event)
            return
        self._active_idx = idx
        self._dragging = True
        self._update_active_from_x(pos.x(), emit_change=True)
        event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if not self.isEnabled() or not self._dragging or self._active_idx is None:
            super().mouseMoveEvent(event)
            return
        pos = event.position().toPoint()
        self._update_active_from_x(pos.x(), emit_change=True)
        event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        had_drag = bool(self._dragging)
        self._dragging = False
        self._active_idx = None
        if had_drag:
            self.editingFinished.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _update_active_from_x(self, x: int, emit_change: bool) -> None:
        if self._active_idx is None or self._active_idx < 0 or self._active_idx >= len(self._values):
            return
        span = max(self._maximum - self._minimum, 1e-12)
        eps = span * 1e-6
        new_v = self._value_from_x(int(x))
        lo = self._minimum + eps
        hi = self._maximum - eps
        if self._active_idx > 0:
            lo = max(lo, self._values[self._active_idx - 1] + eps)
        if self._active_idx < len(self._values) - 1:
            hi = min(hi, self._values[self._active_idx + 1] - eps)
        if hi < lo:
            hi = lo
        new_v = min(max(new_v, lo), hi)
        if abs(new_v - self._values[self._active_idx]) <= eps:
            return
        self._values[self._active_idx] = new_v
        self.update()
        if emit_change:
            self.valuesChanged.emit(self.values())

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        super().paintEvent(event)
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        left, right, cy = self._track_geometry()
        if right <= left:
            p.end()
            return
        enabled = self.isEnabled()
        line_color = QtGui.QColor("#c8d4e5" if enabled else "#d6dee8")
        p.setPen(QtGui.QPen(line_color, 3))
        p.drawLine(left, cy, right, cy)

        if self._values:
            seg_color = QtGui.QColor("#7ea6df" if enabled else "#b7c4d8")
            p.setPen(QtGui.QPen(seg_color, 4))
            points = [left] + [self._x_from_value(v) for v in self._values] + [right]
            for i in range(len(points) - 1):
                p.drawLine(int(points[i]), cy, int(points[i + 1]), cy)

        for i, v in enumerate(self._values):
            x = self._x_from_value(v)
            radius = 6
            if self._active_idx is not None and i == self._active_idx:
                fill = QtGui.QColor("#1e4ed8")
                edge = QtGui.QColor("#153a9d")
            else:
                fill = QtGui.QColor("#ffffff" if enabled else "#f4f7fb")
                edge = QtGui.QColor("#4f79b8" if enabled else "#9eb1cc")
            p.setPen(QtGui.QPen(edge, 1.5))
            p.setBrush(QtGui.QBrush(fill))
            p.drawEllipse(QtCore.QPoint(int(x), int(cy)), radius, radius)

        # Draw value labels above each handle.
        fm = p.fontMetrics()
        for i, v in enumerate(self._values):
            x = self._x_from_value(v)
            text = self._format_handle_value(v)
            tw = fm.horizontalAdvance(text)
            th = fm.height()
            pad_x = 6
            box_h = th + 2
            box_w = tw + pad_x * 2
            bx = int(x - box_w // 2)
            by = max(1, int(cy - 16 - box_h))

            if self._active_idx is not None and i == self._active_idx:
                fill = QtGui.QColor("#1e4ed8")
                edge = QtGui.QColor("#153a9d")
                fg = QtGui.QColor("#ffffff")
            else:
                fill = QtGui.QColor("#ffffff" if enabled else "#f7f9fc")
                edge = QtGui.QColor("#b8c7dc" if enabled else "#d1dbe9")
                fg = QtGui.QColor("#334155" if enabled else "#94a3b8")

            rect = QtCore.QRectF(float(bx), float(by), float(box_w), float(box_h))
            p.setPen(QtGui.QPen(edge, 1))
            p.setBrush(QtGui.QBrush(fill))
            p.drawRoundedRect(rect, 6.0, 6.0)
            p.setPen(QtGui.QPen(fg, 1))
            p.drawText(rect, QtCore.Qt.AlignCenter, text)
        p.end()


class MaskDistanceView(QtWidgets.QMainWindow):
    def __init__(self, controller: "MaskDistanceController", parent=None):
        super().__init__(parent)
        self.controller = controller
        self._table_margin_px = 8
        self._body_layout_profile = ""
        self._last_image_size: tuple[int, int] = (0, 0)
        self._image_name_full = "-"
        self._shortcuts: list[QtGui.QShortcut] = []
        self._calc_running = False
        self._calc_pending = False
        self._graphs_fixed_height = 560
        self._train_graph_source_pixmap: Optional[QtGui.QPixmap] = None
        self._eval_plot_scope = "current"
        self._eval_plot_pixmaps: dict[str, Optional[QtGui.QPixmap]] = {"current": None, "all": None}
        self._eval_plot_placeholders: dict[str, str] = {
            "current": "Run Compare to view current-image metrics",
            "all": "Run Compare to view all-images metrics",
        }
        self._options_image_side_px = 0
        self._options_soft_margin_px = 12
        self._header_image_side_px = 0
        self._app_event_filter_installed = False
        self._setup_ui()
        # Top menubar is intentionally disabled to avoid duplicate controls.
        self.menuBar().setVisible(False)
        self._setup_shortcuts()
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
            self._app_event_filter_installed = True

    # UI ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        self.setWindowTitle("MicroSeg")
        self.resize(1720, 1020)
        self.setMinimumSize(960, 640)
        self.setStyleSheet(
            """
            QMainWindow { background: #f2f5fb; color: #111827; }
            QFrame#Card {
                background: #ffffff;
                border: 1px solid #d6deec;
                border-radius: 12px;
            }
            QFrame#SubCard {
                background: #f8fbff;
                border: 1px solid #dce6f3;
                border-radius: 10px;
            }
            QLabel#Title {
                font-size: 18px;
                font-weight: 700;
                color: #10253d;
            }
            QLabel#PanelTitle {
                font-size: 13px;
                font-weight: 600;
                color: #233a56;
            }
            QLabel#Subtle {
                color: #5b6777;
                font-size: 12px;
            }
            QPushButton {
                background: #ffffff;
                border: 1px solid #c6d0df;
                border-radius: 8px;
                padding: 6px 12px;
            }
            QPushButton:hover { background: #ecf2fb; border-color: #8aa5cf; }
            QPushButton:pressed { background: #dce7f7; }
            QPushButton:checked {
                background: #1e4ed8;
                color: #ffffff;
                border-color: #1e4ed8;
            }
            QPushButton#WorkspacePill {
                border-radius: 14px;
                padding: 6px 12px;
                background: #eef3fb;
                border: 1px solid #d6deec;
                color: #29405e;
            }
            QPushButton#WorkspacePill:hover {
                background: #e6eefb;
                border-color: #9eb3d4;
            }
            QPushButton#WorkspacePill:checked {
                background: #1e4ed8;
                color: #ffffff;
                border-color: #1e4ed8;
                font-weight: 600;
            }
            QCheckBox { spacing: 6px; }
            QLineEdit {
                border: 1px solid #c8d4e5;
                border-radius: 7px;
                padding: 4px 6px;
                background: #ffffff;
            }
            QGroupBox {
                font-weight: 600;
                color: #1f344f;
                border: 1px solid #d8e2f1;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                color: #2c4462;
            }
            QTableWidget {
                border: 1px solid #d6deec;
                border-radius: 8px;
                gridline-color: #e6edf7;
                background: #ffffff;
                alternate-background-color: #f6f9ff;
                selection-background-color: #dbeafe;
                selection-color: #0f172a;
            }
            QTableWidget::item { padding: 4px; }
            QHeaderView::section {
                background: #eef3fb;
                color: #1f344f;
                border: none;
                border-right: 1px solid #d8e2f1;
                border-bottom: 1px solid #d8e2f1;
                padding: 6px 4px;
                font-weight: 600;
            }
            QTableCornerButton::section {
                background: #eef3fb;
                color: #1f344f;
                border: none;
                border-right: 1px solid #d8e2f1;
                border-bottom: 1px solid #d8e2f1;
                font-weight: 600;
            }
            QTabWidget::pane {
                border: 1px solid #d6deec;
                border-radius: 10px;
                background: #ffffff;
                margin-top: 6px;
            }
            QTabBar::tab {
                background: #eef3fb;
                border: 1px solid #d6deec;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: 6px 12px;
                color: #29405e;
                margin-right: 4px;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                color: #10253d;
                font-weight: 600;
            }
            QProgressBar {
                border: 1px solid #cbd6e6;
                border-radius: 7px;
                background: #f7f9fd;
                text-align: center;
                height: 16px;
            }
            QProgressBar::chunk {
                background: #3b82f6;
                border-radius: 6px;
            }
            QToolButton#WorkspaceNav {
                border: 1px solid transparent;
                background: transparent;
                border-radius: 10px;
                min-width: 40px;
                min-height: 40px;
                padding: 4px;
            }
            QToolButton#WorkspaceNav:hover {
                border-color: #d6deec;
                background: #eef3fb;
            }
            QToolButton#WorkspaceNav:checked {
                border-color: #1e4ed8;
                background: #dbeafe;
            }
            QToolButton#SectionToggle {
                border: none;
                background: transparent;
                color: #1f344f;
                font-weight: 600;
                text-align: left;
                padding: 2px 4px;
            }
            QToolButton#SectionToggle:hover {
                background: #eef3fb;
                border-radius: 6px;
            }
            QWidget#RightPane,
            QWidget#WorkspaceContainer,
            QStackedWidget#WorkspaceStack,
            QWidget#PreviewWorkspace,
            QWidget#InferBody {
                background: #f2f5fb;
            }
            QScrollArea#InferScroll {
                border: none;
                background: #f2f5fb;
            }
            QScrollArea#InferScroll QWidget#qt_scrollarea_viewport {
                background: #f2f5fb;
            }
            QScrollArea#InferScroll QScrollBar:vertical {
                background: #f2f5fb;
                width: 10px;
                margin: 0px;
                border: none;
            }
            QScrollArea#InferScroll QScrollBar::handle:vertical {
                background: #cfd9e7;
                min-height: 24px;
                border-radius: 5px;
            }
            QScrollArea#InferScroll QScrollBar::handle:vertical:hover {
                background: #b7c6db;
            }
            QScrollArea#InferScroll QScrollBar::add-line:vertical,
            QScrollArea#InferScroll QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollArea#InferScroll QScrollBar::add-page:vertical,
            QScrollArea#InferScroll QScrollBar::sub-page:vertical {
                background: #f2f5fb;
            }
            """
        )

        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        self.workspace_sidebar_width = 64

        # Placeholder for controller visibility sync. Real controls live in Analyze panel.
        self.mode_header_card = QtWidgets.QFrame()
        self.mode_header_card.setVisible(False)

        # Keep action button objects for controller state sync and shortcuts/menu integration,
        # but remove the dedicated Actions panel from the main header.
        self._hidden_action_host = QtWidgets.QWidget(central)
        self._hidden_action_host.setVisible(False)

        self.btn_set = QtWidgets.QPushButton("Set", self._hidden_action_host)
        self.btn_calc = QtWidgets.QPushButton("Calc")
        self.btn_remove = QtWidgets.QPushButton("Remove", self._hidden_action_host)
        self.btn_undo = QtWidgets.QPushButton("Undo", self._hidden_action_host)
        self.btn_reset = QtWidgets.QPushButton("Reset", self._hidden_action_host)
        self.btn_save = QtWidgets.QPushButton("Save", self._hidden_action_host)
        self.btn_load_session = QtWidgets.QPushButton("Load", self._hidden_action_host)
        self.btn_save_all = QtWidgets.QPushButton("Save All", self._hidden_action_host)
        for b, cb in [
            (self.btn_set, self.controller.on_set),
            (self.btn_calc, self.controller.on_calc),
            (self.btn_remove, self.controller.on_remove_selected),
            (self.btn_undo, self.controller.on_undo),
            (self.btn_reset, self.controller.on_reset),
            (self.btn_save, self.controller.on_save),
            (self.btn_load_session, self.controller.on_load_session),
            (self.btn_save_all, self.controller.on_save_all),
        ]:
            b.clicked.connect(cb)
            b.setMinimumWidth(72)
        self.btn_save.setFixedWidth(74)
        self.btn_save_all.setFixedWidth(74)
        self.btn_load_session.setFixedWidth(74)
        # Buttons are re-used in Preview/Analyze panels below.

        self.mode_sam_btn = QtWidgets.QPushButton("SAM")
        self.mode_sam_btn.setCheckable(True)
        self.mode_lora_btn = QtWidgets.QPushButton("LoRA")
        self.mode_lora_btn.setCheckable(True)
        self.mode_polygon_btn = QtWidgets.QPushButton("Polygon")
        self.mode_polygon_btn.setCheckable(True)
        self.scale_calib_toggle_btn = QtWidgets.QPushButton("Measure")
        self.scale_calib_toggle_btn.setCheckable(True)
        self.scale_calib_toggle_btn.toggled.connect(self.controller.on_toggle_scale_calibration_mode)
        self.mode_group = QtWidgets.QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_group.addButton(self.mode_sam_btn)
        self.mode_group.addButton(self.mode_lora_btn)
        self.mode_group.addButton(self.mode_polygon_btn)
        self.mode_sam_btn.setChecked(True)
        self.mode_sam_btn.clicked.connect(self._on_mode_sam)
        self.mode_lora_btn.clicked.connect(self._on_mode_lora)
        self.mode_polygon_btn.clicked.connect(self._on_mode_polygon)
        self.save_state_label = QtWidgets.QLabel("Saved")
        self._save_state_full_text = "Saved"
        self.save_state_label.setObjectName("Subtle")
        self.save_state_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.save_state_label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        self.save_state_label.setMinimumWidth(120)
        self.save_state_label.setMaximumWidth(190)

        self.body_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        # Keep window size stable when switching tabs with heavier sub-panels.
        self.body_split.setChildrenCollapsible(False)
        self.body_split.setHandleWidth(8)
        self.body_split.splitterMoved.connect(self._sync_top_cards_with_body_split)

        self.left_card = QtWidgets.QFrame()
        self.left_card.setObjectName("Card")
        left_layout = QtWidgets.QVBoxLayout(self.left_card)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(8)

        self.image_title_strip = QtWidgets.QWidget()
        image_title_row = QtWidgets.QHBoxLayout(self.image_title_strip)
        image_title_row.setContentsMargins(2, 0, 2, 0)
        image_title_row.setSpacing(6)
        image_title = QtWidgets.QLabel("Image")
        image_title.setObjectName("PanelTitle")
        self.image_title_label = image_title
        image_title_row.addWidget(image_title)
        image_title_row.addStretch(1)
        image_title_row.addWidget(self.save_state_label, 0, QtCore.Qt.AlignRight)
        image_title_row.addWidget(self.btn_load_session, 0, QtCore.Qt.AlignRight)
        image_title_row.addWidget(self.btn_save, 0, QtCore.Qt.AlignRight)
        image_title_row.addWidget(self.btn_save_all, 0, QtCore.Qt.AlignRight)
        left_layout.addWidget(self.image_title_strip, 0, QtCore.Qt.AlignHCenter)

        self.image_toolbar_card = QtWidgets.QWidget()
        image_toolbar_layout = QtWidgets.QVBoxLayout(self.image_toolbar_card)
        image_toolbar_layout.setContentsMargins(2, 0, 2, 0)
        image_toolbar_layout.setSpacing(4)
        mode_lora_row = QtWidgets.QHBoxLayout()
        mode_lora_row.setContentsMargins(0, 0, 0, 0)
        mode_lora_row.setSpacing(6)
        mode_label = QtWidgets.QLabel("Mode")
        mode_label.setObjectName("PanelTitle")
        mode_lora_row.addWidget(mode_label)
        mode_lora_row.addSpacing(2)
        mode_lora_row.addWidget(self.mode_sam_btn)
        mode_lora_row.addWidget(self.mode_lora_btn)
        mode_lora_row.addWidget(self.mode_polygon_btn)
        mode_lora_row.addWidget(self.scale_calib_toggle_btn)
        mode_lora_row.addStretch(1)
        mode_lora_row.addSpacing(8)
        lora_label = QtWidgets.QLabel("LoRA")
        lora_label.setObjectName("PanelTitle")
        mode_lora_row.addWidget(lora_label)
        self.lora_path_edit = QtWidgets.QLineEdit()
        self.lora_path_edit.setPlaceholderText("checkpoint (.pt/.pth/.bin/.safetensors)")
        self.lora_path_edit.setMinimumWidth(140)
        self.lora_path_edit.setMaximumWidth(420)
        self.lora_path_edit.returnPressed.connect(self.controller.on_lora_apply_checkpoint)
        self.lora_path_edit.editingFinished.connect(self.controller.on_lora_apply_checkpoint)
        mode_lora_row.addWidget(self.lora_path_edit)
        self.lora_browse_btn = QtWidgets.QPushButton("Browse")
        self.lora_browse_btn.setFixedWidth(74)
        self.lora_browse_btn.clicked.connect(self.controller.on_lora_browse_checkpoint)
        mode_lora_row.addWidget(self.lora_browse_btn)
        self.lora_state_label = QtWidgets.QLabel("●")
        self.lora_state_label.setVisible(False)
        image_toolbar_layout.addLayout(mode_lora_row)
        # Backward-compatible alias consumed by controller visibility sync.
        self.preview_control_card = self.image_toolbar_card
        left_layout.addWidget(self.image_toolbar_card, 0, QtCore.Qt.AlignHCenter)

        self.image_header_strip = QtWidgets.QWidget()
        image_top_row = QtWidgets.QHBoxLayout(self.image_header_strip)
        image_top_row.setContentsMargins(0, 0, 0, 0)
        image_top_row.setSpacing(8)
        self.image_index_label = QtWidgets.QLabel("1/1")
        self.image_index_label.setObjectName("Subtle")
        self.image_index_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        image_top_row.addWidget(self.image_index_label, 0, QtCore.Qt.AlignLeft)
        image_top_row.addStretch()

        self.image_nav_center = QtWidgets.QWidget()
        center_layout = QtWidgets.QHBoxLayout(self.image_nav_center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(8)
        self.btn_prev_image = QtWidgets.QPushButton("◀")
        self.btn_prev_image.setFixedWidth(34)
        self.btn_prev_image.clicked.connect(self.controller.on_prev_image)
        self.btn_next_image = QtWidgets.QPushButton("▶")
        self.btn_next_image.setFixedWidth(34)
        self.btn_next_image.clicked.connect(self.controller.on_next_image)
        self.image_name_label = QtWidgets.QLabel("-")
        self.image_name_label.setObjectName("Subtle")
        self.image_name_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_name_label.setFixedWidth(300)
        center_layout.addWidget(self.btn_prev_image)
        center_layout.addWidget(self.image_name_label)
        center_layout.addWidget(self.btn_next_image)
        image_top_row.addWidget(self.image_nav_center, 0, QtCore.Qt.AlignHCenter)
        image_top_row.addStretch()
        self.image_masks_label = QtWidgets.QLabel("Masks=0")
        self.image_masks_label.setObjectName("Subtle")
        self.image_masks_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        image_top_row.addWidget(self.image_masks_label, 0, QtCore.Qt.AlignRight)
        left_layout.addWidget(self.image_header_strip, 0, QtCore.Qt.AlignHCenter)

        self.image_label = ClickableLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.image_label.setMinimumSize(320, 320)
        self.image_label.clicked.connect(self.controller.on_click_image)
        self.image_label.right_clicked.connect(self.controller.on_right_click_image)
        self.image_label.box_dragging.connect(self.controller.on_box_dragging_image)
        self.image_label.box_dragged.connect(self.controller.on_box_drag_image)
        self.image_label.wheel_scrolled.connect(self.controller.on_wheel_image)
        self.image_label.setStyleSheet("QLabel { background: #edf3fb; border-radius: 0px; border: 1px solid #cfdae9; }")
        left_layout.addWidget(self.image_label, stretch=9, alignment=QtCore.Qt.AlignHCenter)

        self.image_actions_card = QtWidgets.QWidget()
        image_actions_layout = QtWidgets.QHBoxLayout(self.image_actions_card)
        image_actions_layout.setContentsMargins(2, 0, 2, 0)
        image_actions_layout.setSpacing(6)
        image_actions_layout.addSpacing(1)
        image_actions_layout.addWidget(self.btn_set)
        image_actions_layout.addWidget(self.btn_undo)
        image_actions_layout.addWidget(self.btn_remove)
        image_actions_layout.addStretch(1)
        left_layout.addWidget(self.image_actions_card, stretch=0, alignment=QtCore.Qt.AlignHCenter)

        display_strip = QtWidgets.QWidget()
        display_strip_layout = QtWidgets.QVBoxLayout(display_strip)
        display_strip_layout.setContentsMargins(2, 0, 2, 0)
        display_strip_layout.setSpacing(3)
        display_title = QtWidgets.QLabel("Options")
        display_title.setObjectName("PanelTitle")
        display_strip_layout.addWidget(display_title)
        display_grid = QtWidgets.QGridLayout()
        display_grid.setContentsMargins(0, 0, 0, 0)
        display_grid.setHorizontalSpacing(8)
        display_grid.setVerticalSpacing(3)
        display_grid.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.display_grid = display_grid
        self.show_current_checkbox = QtWidgets.QCheckBox("Current")
        self.show_current_checkbox.setChecked(True)
        self.show_current_checkbox.stateChanged.connect(self.controller.on_toggle_show_current)
        self.show_set_checkbox = QtWidgets.QCheckBox("All masks")
        self.show_set_checkbox.setChecked(True)
        self.show_set_checkbox.stateChanged.connect(self.controller.on_toggle_show_set)
        self.show_prompts_checkbox = QtWidgets.QCheckBox("Prompts")
        self.show_prompts_checkbox.setChecked(True)
        self.show_prompts_checkbox.stateChanged.connect(self.controller.on_toggle_show_prompts)
        self.show_nearest_checkbox = QtWidgets.QCheckBox("Nearest")
        self.show_nearest_checkbox.setChecked(True)
        self.show_nearest_checkbox.stateChanged.connect(self.controller.on_toggle_show_links)
        self.show_gt_checkbox = QtWidgets.QCheckBox("GT")
        self.show_gt_checkbox.setChecked(False)
        self.show_gt_checkbox.stateChanged.connect(self.controller.on_toggle_show_gt_overlay)
        self.show_bbox_checkbox = QtWidgets.QCheckBox("BBox")
        self.show_bbox_checkbox.setChecked(False)
        self.show_bbox_checkbox.stateChanged.connect(self.controller.on_toggle_show_bbox)
        self.show_feret_checkbox = QtWidgets.QCheckBox("Feret")
        self.show_feret_checkbox.setChecked(False)
        self.show_feret_checkbox.stateChanged.connect(self.controller.on_toggle_show_feret)
        # Backward-compatible aliases used in controller sync/restore paths.
        self.show_axes_checkbox = self.show_feret_checkbox
        self.show_feret_para_checkbox = self.show_feret_checkbox
        self.show_ellipse_checkbox = QtWidgets.QCheckBox("Ellipse")
        self.show_ellipse_checkbox.setChecked(False)
        self.show_ellipse_checkbox.stateChanged.connect(self.controller.on_toggle_show_ellipse)
        self.overlay_centroid_checkbox = QtWidgets.QCheckBox("Centroid")
        self.overlay_centroid_checkbox.setChecked(False)
        self.overlay_centroid_checkbox.stateChanged.connect(self.controller.on_toggle_overlay_centroid)
        display_grid.addWidget(self.show_prompts_checkbox, 0, 0)
        display_grid.addWidget(self.show_current_checkbox, 0, 1)
        display_grid.addWidget(self.show_set_checkbox, 0, 2)
        display_grid.addWidget(self.show_nearest_checkbox, 0, 3)
        display_grid.addWidget(self.overlay_centroid_checkbox, 0, 4)
        display_grid.addWidget(self.show_gt_checkbox, 1, 0)
        display_grid.addWidget(self.show_bbox_checkbox, 1, 1)
        display_grid.addWidget(self.show_feret_checkbox, 1, 2)
        display_grid.addWidget(self.show_ellipse_checkbox, 1, 3)
        self._display_option_rows: tuple[tuple[QtWidgets.QCheckBox, ...], ...] = (
            (
                self.show_prompts_checkbox,
                self.show_current_checkbox,
                self.show_set_checkbox,
                self.show_nearest_checkbox,
                self.overlay_centroid_checkbox,
            ),
            (
                self.show_gt_checkbox,
                self.show_bbox_checkbox,
                self.show_feret_checkbox,
                self.show_ellipse_checkbox,
            ),
        )
        display_grid.setColumnStretch(0, 0)
        display_grid.setColumnStretch(1, 0)
        display_grid.setColumnStretch(2, 0)
        display_grid.setColumnStretch(3, 0)
        display_grid.setColumnStretch(4, 0)
        display_strip_layout.addLayout(display_grid)
        self.display_strip = display_strip
        self._apply_options_side_padding()
        left_layout.addWidget(display_strip, stretch=0, alignment=QtCore.Qt.AlignHCenter)

        scale_strip = QtWidgets.QWidget()
        scale_strip_layout = QtWidgets.QVBoxLayout(scale_strip)
        scale_strip_layout.setContentsMargins(2, 0, 2, 0)
        scale_strip_layout.setSpacing(5)
        scale_title = QtWidgets.QLabel("Scale")
        scale_title.setObjectName("PanelTitle")
        scale_strip_layout.addWidget(scale_title)

        scale_row = QtWidgets.QHBoxLayout()
        scale_row.setSpacing(6)
        scale_row.addWidget(QtWidgets.QLabel("Length:"))
        self.scale_bar_length_edit = QtWidgets.QLineEdit()
        self.scale_bar_length_edit.setFixedWidth(90)
        self.scale_bar_length_edit.setPlaceholderText("e.g. 5")
        self.scale_bar_length_edit.returnPressed.connect(self.controller.on_scale_fields_edited)
        self.scale_bar_length_edit.editingFinished.connect(self.controller.on_scale_fields_edited)
        scale_row.addWidget(self.scale_bar_length_edit)
        self.display_unit_combo = QtWidgets.QComboBox()
        self.display_unit_combo.addItems(["nm", "um", "mm"])
        self.display_unit_combo.setFixedWidth(72)
        self.display_unit_combo.currentIndexChanged.connect(self.controller.on_display_unit_changed)
        scale_row.addWidget(self.display_unit_combo)
        scale_row.addWidget(QtWidgets.QLabel("px:"))
        self.scale_px_edit = QtWidgets.QLineEdit()
        self.scale_px_edit.setFixedWidth(86)
        self.scale_px_edit.setPlaceholderText("e.g. 177")
        self.scale_px_edit.returnPressed.connect(self.controller.on_scale_fields_edited)
        self.scale_px_edit.editingFinished.connect(self.controller.on_scale_fields_edited)
        scale_row.addWidget(self.scale_px_edit)
        self.scale_slider_wrap = QtWidgets.QWidget()
        self.scale_slider_wrap.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        scale_slider_wrap_layout = QtWidgets.QVBoxLayout(self.scale_slider_wrap)
        scale_slider_wrap_layout.setContentsMargins(0, 0, 0, 0)
        scale_slider_wrap_layout.setSpacing(1)
        self.scale_preset_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.scale_preset_slider.setRange(0, 1000)
        self.scale_preset_slider.setSingleStep(1)
        self.scale_preset_slider.setPageStep(20)
        self.scale_preset_slider.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.scale_preset_slider.setMinimumWidth(220)
        self.scale_preset_slider.valueChanged.connect(self.controller.on_scale_preset_slider_changed)
        scale_slider_wrap_layout.addWidget(self.scale_preset_slider)
        self.scale_preset_ruler = ScalePresetRuler()
        if MAGNIFICATION_PRESETS:
            self.scale_preset_ruler.set_ticks([(0.0, MAGNIFICATION_PRESETS[0]), (1.0, MAGNIFICATION_PRESETS[-1])])
        else:
            self.scale_preset_ruler.set_ticks([])
        # Labels on the ruler can overlap on compact widths; rely on right-side hint text.
        self.scale_preset_ruler.setVisible(False)
        scale_slider_wrap_layout.addWidget(self.scale_preset_ruler)
        scale_row.addWidget(self.scale_slider_wrap, stretch=1)
        self.scale_preset_hint_label = QtWidgets.QLabel("")
        self.scale_preset_hint_label.setObjectName("Subtle")
        self.scale_preset_hint_label.setMinimumWidth(96)
        self.scale_preset_hint_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        scale_row.addWidget(self.scale_preset_hint_label)
        scale_row.addStretch()
        scale_strip_layout.addLayout(scale_row)

        cluster_row = QtWidgets.QHBoxLayout()
        cluster_row.setSpacing(6)
        self.cluster_label = QtWidgets.QLabel("Cluster th (N1, nm):")
        cluster_row.addWidget(self.cluster_label)
        self.cluster_edit = QtWidgets.QLineEdit()
        self.cluster_edit.setFixedWidth(80)
        cluster_row.addWidget(self.cluster_edit)
        cluster_apply = QtWidgets.QPushButton("Apply")
        cluster_apply.clicked.connect(self.controller.on_apply_cluster_threshold)
        cluster_row.addWidget(cluster_apply)
        cluster_row.addSpacing(8)
        self.cluster_count_label = QtWidgets.QLabel("count:-")
        self.cluster_count_label.setObjectName("Subtle")
        self.cluster_mean_label = QtWidgets.QLabel("mean:-")
        self.cluster_mean_label.setObjectName("Subtle")
        self.cluster_min_label = QtWidgets.QLabel("min:-")
        self.cluster_min_label.setObjectName("Subtle")
        self.cluster_max_label = QtWidgets.QLabel("max:-")
        self.cluster_max_label.setObjectName("Subtle")
        cluster_row.addWidget(self.cluster_count_label)
        cluster_row.addWidget(self.cluster_mean_label)
        cluster_row.addWidget(self.cluster_min_label)
        cluster_row.addWidget(self.cluster_max_label)
        cluster_row.addStretch()
        scale_strip_layout.addLayout(cluster_row)
        self.scale_strip = scale_strip
        left_layout.addWidget(scale_strip, stretch=0, alignment=QtCore.Qt.AlignHCenter)

        self.mask_info_label = QtWidgets.QLabel("")
        self.mask_info_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.mask_info_label.setWordWrap(True)
        self.mask_info_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        self.mask_info_label.setStyleSheet("QLabel { padding: 7px; background: #f7f9fc; border: 1px solid #dbe3ef; border-radius: 8px; color: #334155; }")
        self.mask_info_label.setVisible(False)
        left_layout.addWidget(self.mask_info_label, stretch=0)

        self.body_split.addWidget(self.left_card)

        right_pane = QtWidgets.QWidget()
        right_pane.setObjectName("RightPane")
        right_pane_layout = QtWidgets.QVBoxLayout(right_pane)
        right_pane_layout.setContentsMargins(0, 0, 0, 0)
        right_pane_layout.setSpacing(8)

        workspace_container = QtWidgets.QWidget()
        workspace_container.setObjectName("WorkspaceContainer")
        workspace_container_layout = QtWidgets.QVBoxLayout(workspace_container)
        workspace_container_layout.setContentsMargins(0, 0, 0, 0)
        workspace_container_layout.setSpacing(0)
        self.workspace_menu_actions: dict[int, QtGui.QAction] = {}
        self.workspace_buttons: dict[int, QtWidgets.QToolButton] = {}
        self.workspace_button_group = QtWidgets.QButtonGroup(self)
        self.workspace_button_group.setExclusive(True)

        self.workspace_stack = QtWidgets.QStackedWidget()
        self.workspace_stack.setObjectName("WorkspaceStack")

        preview_workspace = QtWidgets.QWidget()
        preview_workspace.setObjectName("PreviewWorkspace")
        preview_workspace_layout = QtWidgets.QVBoxLayout(preview_workspace)
        preview_workspace_layout.setContentsMargins(0, 0, 0, 0)
        preview_workspace_layout.setSpacing(8)

        self._infer_active_panel = "analyze"
        self._infer_sections: dict[str, tuple[QtWidgets.QToolButton, QtWidgets.QWidget, QtWidgets.QWidget]] = {}

        # Keep help text object for mode-switch sync, but hide it from the panel.
        self.annotate_help_label = QtWidgets.QLabel(
            "SAM/LoRA: left=positive, right=negative. Polygon: left=vertex, right=close. Use Set to commit."
        )
        self.annotate_help_label.setWordWrap(True)
        self.annotate_help_label.setObjectName("Subtle")
        self.annotate_help_label.setVisible(False)

        mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        mono.setPointSize(max(8, mono.pointSize()))

        self.review_summary_label = QtWidgets.QLabel("No masks")
        self.review_summary_label.setObjectName("Subtle")
        self.review_summary_label.setWordWrap(True)
        self.review_table = QtWidgets.QTableWidget(0, 18)
        self.review_table.setHorizontalHeaderLabels(
            [
                "ECD",
                "VESD",
                "Area(px)",
                "Area(BBox)",
                "Area(VESD)",
                "Aspect(Feret)",
                "Aspect(Ellipse)",
                "Feret Maj",
                "Feret Min",
                "Ellipse Maj",
                "Ellipse Min",
                "Cent X",
                "Cent Y",
                "BBox X",
                "BBox Y",
                "BBox Cx",
                "BBox Cy",
                "Score",
            ]
        )
        review_header = self.review_table.horizontalHeader()
        review_header.setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        review_header.setStretchLastSection(False)
        review_header.setMinimumSectionSize(56)
        review_header.setDefaultAlignment(QtCore.Qt.AlignCenter)
        self.review_table.verticalHeader().setVisible(True)
        self.review_table.verticalHeader().setDefaultSectionSize(30)
        self.review_table.verticalHeader().setDefaultAlignment(QtCore.Qt.AlignCenter)
        self._review_visible_rows = 10
        self.review_table.setAlternatingRowColors(False)
        self.review_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.review_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.review_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.review_table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.review_table.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.review_table.setMinimumWidth(0)
        self.review_table.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)
        self.review_table.setFont(mono)
        self.review_table.setWordWrap(False)
        self.review_table.setTextElideMode(QtCore.Qt.ElideNone)
        self._apply_review_table_height()
        self.review_table.cellClicked.connect(self.controller.on_review_row_clicked)
        review_card = QtWidgets.QWidget()
        review_card_layout = QtWidgets.QVBoxLayout(review_card)
        review_card_layout.setContentsMargins(10, 10, 10, 10)
        review_card_layout.setSpacing(8)
        sort_row = QtWidgets.QHBoxLayout()
        sort_row.setContentsMargins(0, 0, 0, 0)
        sort_row.setSpacing(6)
        sort_row.addWidget(QtWidgets.QLabel("Sort"))
        self.review_sort_key_combo = QtWidgets.QComboBox()
        self.review_sort_key_combo.addItem("Index", "index")
        self.review_sort_key_combo.addItem("ECD", "ecd")
        self.review_sort_key_combo.addItem("VESD", "vesd")
        self.review_sort_key_combo.addItem("Area(px)", "area")
        self.review_sort_key_combo.addItem("Area(BBox)", "bbox_area")
        self.review_sort_key_combo.addItem("Area(VESD)", "area_vesd")
        self.review_sort_key_combo.addItem("Aspect(Feret)", "aspect_feret")
        self.review_sort_key_combo.addItem("Aspect(Ellipse)", "aspect_ellipse")
        self.review_sort_key_combo.addItem("Feret Maj", "major")
        self.review_sort_key_combo.addItem("Feret Min", "minor")
        self.review_sort_key_combo.addItem("Ellipse Maj", "ellipse_major")
        self.review_sort_key_combo.addItem("Ellipse Min", "ellipse_minor")
        self.review_sort_key_combo.addItem("Score", "score")
        self.review_sort_key_combo.currentIndexChanged.connect(self.controller.on_review_sort_changed)
        sort_row.addWidget(self.review_sort_key_combo)
        self.review_sort_order_combo = QtWidgets.QComboBox()
        self.review_sort_order_combo.addItem("Asc", "asc")
        self.review_sort_order_combo.addItem("Desc", "desc")
        self.review_sort_order_combo.currentIndexChanged.connect(self.controller.on_review_sort_changed)
        sort_row.addWidget(self.review_sort_order_combo)
        sort_row.addStretch(1)
        review_card_layout.addLayout(sort_row)
        self.review_summary_label.setVisible(False)
        review_card_layout.addWidget(self.review_summary_label)
        review_card_layout.addWidget(self.review_table, stretch=1)

        self.calc_status_label = QtWidgets.QLabel("")
        self.calc_status_label.setObjectName("Subtle")
        self.calc_progress = QtWidgets.QProgressBar()
        self.calc_progress.setRange(0, 1)
        self.calc_progress.setValue(0)
        self.calc_progress.setVisible(False)
        self.calc_progress.setMinimumWidth(120)
        self.btn_calc_current = QtWidgets.QPushButton("Calc Current")
        self.btn_calc_current.clicked.connect(self.controller.on_calc_current)
        self.btn_calc_all = QtWidgets.QPushButton("Calc All")
        self.btn_calc_all.clicked.connect(self.controller.on_calc_all)

        analyze_card = QtWidgets.QWidget()
        analyze_layout = QtWidgets.QVBoxLayout(analyze_card)
        analyze_layout.setContentsMargins(10, 10, 10, 10)
        analyze_layout.setSpacing(8)
        self.stats_table = QtWidgets.QTableWidget(6, 16)
        self.stats_table.setHorizontalHeaderLabels(
            [
                "N1",
                "N2",
                "Cent1",
                "Cent2",
                "ECD",
                "VESD",
                "Area(px)",
                "BBox Area",
                "Area(VESD)",
                "Feret Maj",
                "Feret Min",
                "Area(FeretRect)",
                "Ellipse Maj",
                "Ellipse Min",
                "Aspect",
                "Shape",
            ]
        )
        self.stats_table.setVerticalHeaderLabels(["mean", "median", "std", "cv(%)", "min", "max"])
        self.stats_table.setAlternatingRowColors(False)
        self.stats_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        self.stats_table.horizontalHeader().setStretchLastSection(False)
        self.stats_table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
        self.stats_table.verticalHeader().setStretchLastSection(False)
        self.stats_table.verticalHeader().setDefaultSectionSize(30)
        self.stats_table.verticalHeader().setMinimumWidth(66)
        self.stats_table.verticalHeader().setDefaultAlignment(QtCore.Qt.AlignCenter)
        self.stats_table.setCornerButtonEnabled(True)
        self.stats_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.stats_table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.stats_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.stats_table.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.stats_table.setMinimumWidth(0)
        self.stats_table.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        self.stats_table.setFont(mono)
        self._fit_table_height()
        self._adjust_stats_table_columns()

        self.stats_summary_label = QtWidgets.QLabel("Masks=0")
        self.stats_summary_label.setObjectName("Subtle")
        self.stats_summary_label.setVisible(False)
        analyze_layout.addWidget(self.stats_summary_label)
        analyze_layout.addWidget(self.stats_table, stretch=1)
        self._stats_corner_text = "n=0"
        self._stats_corner_tooltip = ""
        QtCore.QTimer.singleShot(0, self._apply_stats_corner_text)

        graph_card_style = "QLabel { background: #f8fafc; border: 1px dashed #cfd9e7; border-radius: 8px; color: #4b5563; }"
        self.graphs_main_label = QtWidgets.QLabel("Graphs will appear after Calc")
        self.graphs_main_label.setAlignment(QtCore.Qt.AlignCenter)
        self.graphs_main_label.setScaledContents(False)
        self.graphs_main_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        self.graphs_main_label.setMinimumHeight(300)
        self.graphs_main_label.setMaximumHeight(300)
        self.graphs_main_label.setStyleSheet(graph_card_style)
        analyze_layout.addWidget(self.graphs_main_label, stretch=0)

        distribution_row = QtWidgets.QHBoxLayout()
        distribution_row.setContentsMargins(0, 0, 0, 0)
        distribution_row.setSpacing(6)
        self.main_graph_metric_combo = QtWidgets.QComboBox()
        self.main_graph_metric_combo.addItem("Main: N1", "nearest1")
        self.main_graph_metric_combo.addItem("Main: N2", "nearest2")
        self.main_graph_metric_combo.addItem("Main: Diameter ECD", "ecd")
        self.main_graph_metric_combo.addItem("Main: Diameter VESD", "vesd")
        self.main_graph_metric_combo.addItem("Main: Area px", "area")
        self.main_graph_metric_combo.addItem("Main: Area BBox", "bbox")
        self.main_graph_metric_combo.addItem("Main: Area VESD", "area_vesd")
        self.main_graph_metric_combo.addItem("Main: Aspect Feret", "aspect_feret")
        self.main_graph_metric_combo.addItem("Main: Aspect Ellipse", "aspect_ellipse")
        self.main_graph_metric_combo.addItem("Main: Fractal", "fractal")
        self.main_graph_metric_combo.setMinimumWidth(170)
        self.main_graph_metric_combo.currentIndexChanged.connect(self.controller.on_main_graph_metric_changed)
        distribution_row.addWidget(self.main_graph_metric_combo)
        self.distribution_metric_combo = QtWidgets.QComboBox()
        self.distribution_metric_combo.addItem("Class: Off", "none")
        self.distribution_metric_combo.addItem("Class: VESD", "vesd")
        self.distribution_metric_combo.addItem("Class: ECD", "ecd")
        self.distribution_metric_combo.addItem("Class: Aspect", "aspect")
        self.distribution_metric_combo.setFixedWidth(132)
        self.distribution_metric_combo.currentIndexChanged.connect(self.controller.on_distribution_metric_changed)
        distribution_row.addWidget(self.distribution_metric_combo)
        self.distribution_bins_minus_btn = QtWidgets.QPushButton("-")
        self.distribution_bins_minus_btn.setFixedWidth(30)
        self.distribution_bins_minus_btn.clicked.connect(self.controller.on_distribution_bins_decrease)
        distribution_row.addWidget(self.distribution_bins_minus_btn)
        self.distribution_bins_spin = QtWidgets.QSpinBox()
        self.distribution_bins_spin.setRange(2, 20)
        self.distribution_bins_spin.setValue(3)
        self.distribution_bins_spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.distribution_bins_spin.setVisible(False)
        self.distribution_bins_spin.valueChanged.connect(self.controller.on_distribution_bins_changed)
        self.distribution_bins_plus_btn = QtWidgets.QPushButton("+")
        self.distribution_bins_plus_btn.setFixedWidth(30)
        self.distribution_bins_plus_btn.clicked.connect(self.controller.on_distribution_bins_increase)
        distribution_row.addWidget(self.distribution_bins_plus_btn)
        self.distribution_edges_edit = QtWidgets.QLineEdit()
        self.distribution_edges_edit.setPlaceholderText("edges: 0, 2, 4, 6")
        self.distribution_edges_edit.setMinimumWidth(160)
        self.distribution_edges_edit.setMaximumWidth(300)
        self.distribution_edges_edit.returnPressed.connect(self.controller.on_apply_distribution_bins)
        distribution_row.addWidget(self.distribution_edges_edit, stretch=1)
        self.distribution_apply_btn = QtWidgets.QPushButton("Apply")
        self.distribution_apply_btn.clicked.connect(self.controller.on_apply_distribution_bins)
        distribution_row.addWidget(self.distribution_apply_btn)
        self.distribution_clear_btn = QtWidgets.QPushButton("Clear")
        self.distribution_clear_btn.clicked.connect(self.controller.on_clear_distribution_bins)
        distribution_row.addWidget(self.distribution_clear_btn)
        self.distribution_status_label = QtWidgets.QLabel("")
        self.distribution_status_label.setObjectName("Subtle")
        self.distribution_status_label.setVisible(False)
        distribution_row.addWidget(self.distribution_status_label)
        distribution_row.addStretch(1)
        analyze_layout.addLayout(distribution_row)
        self.distribution_slider = MultiHandleSlider()
        self.distribution_slider.valuesChanged.connect(self.controller.on_distribution_slider_changed)
        self.distribution_slider.editingFinished.connect(self.controller.on_distribution_slider_released)
        analyze_layout.addWidget(self.distribution_slider)

        self.graphs_panel = QtWidgets.QWidget()
        self.graphs_panel.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.graphs_panel.setMinimumHeight(self._graphs_fixed_height)
        self.graphs_panel.setMaximumHeight(self._graphs_fixed_height)
        # Backward-compatible alias for legacy controller paths.
        self.graphs_label = self.graphs_panel
        graphs_panel_layout = QtWidgets.QVBoxLayout(self.graphs_panel)
        graphs_panel_layout.setContentsMargins(0, 0, 0, 0)
        graphs_panel_layout.setSpacing(8)

        top_graph_row = QtWidgets.QHBoxLayout()
        top_graph_row.setContentsMargins(0, 0, 0, 0)
        top_graph_row.setSpacing(10)

        nearest_graph_col = QtWidgets.QWidget()
        nearest_graph_col_layout = QtWidgets.QVBoxLayout(nearest_graph_col)
        nearest_graph_col_layout.setContentsMargins(0, 0, 0, 0)
        nearest_graph_col_layout.setSpacing(4)
        self.graphs_top_label = QtWidgets.QLabel("Graphs will appear after Calc")
        self.graphs_top_label.setAlignment(QtCore.Qt.AlignCenter)
        self.graphs_top_label.setScaledContents(False)
        self.graphs_top_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)
        self.graphs_top_label.setMinimumSize(0, 0)
        self.graphs_top_label.setStyleSheet(graph_card_style)
        nearest_graph_col_layout.addWidget(self.graphs_top_label, stretch=1)
        nearest_metric_layout = QtWidgets.QHBoxLayout()
        nearest_metric_layout.setContentsMargins(0, 0, 0, 0)
        nearest_metric_layout.setSpacing(6)
        nearest_metric_layout.addWidget(QtWidgets.QLabel("Nearest"))
        self.nearest_hist_metric_combo = QtWidgets.QComboBox()
        self.nearest_hist_metric_combo.addItem("N1", "nearest1")
        self.nearest_hist_metric_combo.addItem("N2", "nearest2")
        self.nearest_hist_metric_combo.setFixedWidth(86)
        self.nearest_hist_metric_combo.currentIndexChanged.connect(self.controller.on_nearest_hist_metric_changed)
        nearest_metric_layout.addWidget(self.nearest_hist_metric_combo)
        nearest_metric_layout.addStretch(1)
        nearest_graph_col_layout.addLayout(nearest_metric_layout)
        top_graph_row.addWidget(nearest_graph_col, stretch=1)

        self.graphs_fractal_label = QtWidgets.QLabel("")
        self.graphs_fractal_label.setAlignment(QtCore.Qt.AlignCenter)
        self.graphs_fractal_label.setScaledContents(False)
        self.graphs_fractal_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)
        self.graphs_fractal_label.setMinimumSize(0, 0)
        self.graphs_fractal_label.setStyleSheet(graph_card_style)
        top_graph_row.addWidget(self.graphs_fractal_label, stretch=1)
        graphs_panel_layout.addLayout(top_graph_row, stretch=33)

        mid_graph_row = QtWidgets.QHBoxLayout()
        mid_graph_row.setContentsMargins(0, 0, 0, 0)
        mid_graph_row.setSpacing(10)

        size_graph_col = QtWidgets.QWidget()
        size_graph_col_layout = QtWidgets.QVBoxLayout(size_graph_col)
        size_graph_col_layout.setContentsMargins(0, 0, 0, 0)
        size_graph_col_layout.setSpacing(4)
        self.graphs_size_label = QtWidgets.QLabel("")
        self.graphs_size_label.setAlignment(QtCore.Qt.AlignCenter)
        self.graphs_size_label.setScaledContents(False)
        self.graphs_size_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)
        self.graphs_size_label.setMinimumSize(0, 0)
        self.graphs_size_label.setStyleSheet(graph_card_style)
        size_graph_col_layout.addWidget(self.graphs_size_label, stretch=1)
        size_metric_layout = QtWidgets.QHBoxLayout()
        size_metric_layout.setContentsMargins(0, 0, 0, 0)
        size_metric_layout.setSpacing(6)
        size_metric_layout.addWidget(QtWidgets.QLabel("Size"))
        self.size_hist_metric_combo = QtWidgets.QComboBox()
        self.size_hist_metric_combo.addItem("ECD", "ecd")
        self.size_hist_metric_combo.addItem("VESD", "vesd")
        self.size_hist_metric_combo.setFixedWidth(92)
        self.size_hist_metric_combo.currentIndexChanged.connect(self.controller.on_size_hist_metric_changed)
        size_metric_layout.addWidget(self.size_hist_metric_combo)
        size_metric_layout.addStretch(1)
        size_graph_col_layout.addLayout(size_metric_layout)
        mid_graph_row.addWidget(size_graph_col, stretch=1)

        area_graph_col = QtWidgets.QWidget()
        area_graph_col_layout = QtWidgets.QVBoxLayout(area_graph_col)
        area_graph_col_layout.setContentsMargins(0, 0, 0, 0)
        area_graph_col_layout.setSpacing(4)
        self.graphs_area_label = QtWidgets.QLabel("")
        self.graphs_area_label.setAlignment(QtCore.Qt.AlignCenter)
        self.graphs_area_label.setScaledContents(False)
        self.graphs_area_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)
        self.graphs_area_label.setMinimumSize(0, 0)
        self.graphs_area_label.setStyleSheet(graph_card_style)
        area_graph_col_layout.addWidget(self.graphs_area_label, stretch=1)
        area_metric_layout = QtWidgets.QHBoxLayout()
        area_metric_layout.setContentsMargins(0, 0, 0, 0)
        area_metric_layout.setSpacing(6)
        area_metric_layout.addWidget(QtWidgets.QLabel("Area"))
        self.area_hist_metric_combo = QtWidgets.QComboBox()
        self.area_hist_metric_combo.addItem("px", "area")
        self.area_hist_metric_combo.addItem("BBox", "bbox")
        self.area_hist_metric_combo.addItem("VESD", "vesd")
        self.area_hist_metric_combo.setFixedWidth(86)
        self.area_hist_metric_combo.currentIndexChanged.connect(self.controller.on_area_hist_metric_changed)
        area_metric_layout.addWidget(self.area_hist_metric_combo)
        area_metric_layout.addStretch(1)
        area_graph_col_layout.addLayout(area_metric_layout)
        mid_graph_row.addWidget(area_graph_col, stretch=1)
        graphs_panel_layout.addLayout(mid_graph_row, stretch=33)

        bot_graph_row = QtWidgets.QHBoxLayout()
        bot_graph_row.setContentsMargins(0, 0, 0, 0)
        bot_graph_row.setSpacing(10)
        aspect_graph_col = QtWidgets.QWidget()
        aspect_graph_col_layout = QtWidgets.QVBoxLayout(aspect_graph_col)
        aspect_graph_col_layout.setContentsMargins(0, 0, 0, 0)
        aspect_graph_col_layout.setSpacing(4)
        self.graphs_aspect_label = QtWidgets.QLabel("")
        self.graphs_aspect_label.setAlignment(QtCore.Qt.AlignCenter)
        self.graphs_aspect_label.setScaledContents(False)
        self.graphs_aspect_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)
        self.graphs_aspect_label.setMinimumSize(0, 0)
        self.graphs_aspect_label.setStyleSheet(graph_card_style)
        aspect_graph_col_layout.addWidget(self.graphs_aspect_label, stretch=1)
        aspect_metric_layout = QtWidgets.QHBoxLayout()
        aspect_metric_layout.setContentsMargins(0, 0, 0, 0)
        aspect_metric_layout.setSpacing(6)
        aspect_metric_layout.addWidget(QtWidgets.QLabel("Aspect"))
        self.aspect_hist_metric_combo = QtWidgets.QComboBox()
        self.aspect_hist_metric_combo.addItem("Feret", "feret")
        self.aspect_hist_metric_combo.addItem("Ellipse", "ellipse")
        self.aspect_hist_metric_combo.setFixedWidth(92)
        self.aspect_hist_metric_combo.currentIndexChanged.connect(self.controller.on_aspect_hist_metric_changed)
        aspect_metric_layout.addWidget(self.aspect_hist_metric_combo)
        aspect_metric_layout.addStretch(1)
        aspect_graph_col_layout.addLayout(aspect_metric_layout)
        bot_graph_row.addWidget(aspect_graph_col, stretch=1)
        self.graphs_reserved_label = QtWidgets.QLabel("Reserved")
        self.graphs_reserved_label.setAlignment(QtCore.Qt.AlignCenter)
        self.graphs_reserved_label.setScaledContents(False)
        self.graphs_reserved_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.graphs_reserved_label.setStyleSheet(graph_card_style)
        bot_graph_row.addWidget(self.graphs_reserved_label, stretch=1)
        graphs_panel_layout.addLayout(bot_graph_row, stretch=33)

        analyze_layout.addWidget(self.graphs_panel, stretch=1)

        graph_options_row = QtWidgets.QHBoxLayout()
        graph_options_row.setContentsMargins(0, 0, 0, 0)
        graph_options_row.setSpacing(8)
        self.realtime_calc_checkbox = QtWidgets.QCheckBox("Realtime calc")
        self.realtime_calc_checkbox.setChecked(False)
        self.realtime_calc_checkbox.stateChanged.connect(self.controller.on_toggle_realtime_calc)
        self.include_zero_checkbox = QtWidgets.QCheckBox("Include distance=0")
        self.include_zero_checkbox.setChecked(False)
        self.include_zero_checkbox.stateChanged.connect(self.controller.on_toggle_include_zero)
        self.fractal_checkbox = QtWidgets.QCheckBox("Fractal slides x20")
        self.fractal_checkbox.setChecked(False)
        self.fractal_checkbox.stateChanged.connect(self.controller.on_toggle_fractal_slides)
        graph_options_row.addWidget(self.realtime_calc_checkbox)
        graph_options_row.addWidget(self.include_zero_checkbox)
        graph_options_row.addWidget(self.fractal_checkbox)
        graph_options_row.addStretch(1)
        analyze_layout.addLayout(graph_options_row)
        eval_card = self._build_evaluation_workspace()

        self.infer_scroll = QtWidgets.QScrollArea()
        self.infer_scroll.setObjectName("InferScroll")
        self.infer_scroll.setWidgetResizable(True)
        self.infer_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.infer_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        infer_body = QtWidgets.QWidget()
        infer_body.setObjectName("InferBody")
        infer_body_layout = QtWidgets.QVBoxLayout(infer_body)
        infer_body_layout.setContentsMargins(0, 0, 0, 0)
        infer_body_layout.setSpacing(8)
        infer_body_layout.addWidget(self._create_infer_section("preview", "Instances", review_card, expanded=True))
        infer_body_layout.addWidget(
            self._create_infer_section(
                "analyze",
                "Analysis",
                analyze_card,
                expanded=True,
                header_widgets=[
                    self.calc_status_label,
                    self.calc_progress,
                    self.btn_calc_current,
                    self.btn_calc_all,
                ],
            )
        )
        infer_body_layout.addWidget(
            self._create_infer_section(
                "evaluate",
                "Validation",
                eval_card,
                expanded=False,
                header_widgets=[self.eval_run_current_btn, self.eval_run_all_btn],
            )
        )
        infer_body_layout.addStretch(1)
        self.infer_scroll.setWidget(infer_body)
        preview_workspace_layout.addWidget(self.infer_scroll, stretch=1)

        self.workspace_stack.addWidget(preview_workspace)
        self.workspace_stack.addWidget(self._build_filters_workspace())
        self.workspace_stack.addWidget(self._build_train_workspace())
        self.workspace_stack.addWidget(self._build_track_workspace())

        workspace_container_layout.addWidget(self.workspace_stack, stretch=1)
        right_pane_layout.addWidget(workspace_container, stretch=1)
        right_pane.setMinimumWidth(0)
        right_pane.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)
        self.body_split.addWidget(right_pane)
        self.body_split.setStretchFactor(0, 5)
        self.body_split.setStretchFactor(1, 4)
        self._lock_body_splitter_handle()

        main_content = QtWidgets.QWidget()
        main_content_layout = QtWidgets.QHBoxLayout(main_content)
        main_content_layout.setContentsMargins(0, 0, 0, 0)
        main_content_layout.setSpacing(8)

        self.workspace_sidebar_card = QtWidgets.QWidget()
        sidebar_layout = QtWidgets.QVBoxLayout(self.workspace_sidebar_card)
        sidebar_layout.setContentsMargins(8, 10, 8, 10)
        sidebar_layout.setSpacing(8)

        def make_workspace_icon(kind: str) -> QtGui.QIcon:
            def _draw(color: QtGui.QColor) -> QtGui.QPixmap:
                size = 20
                pm = QtGui.QPixmap(size, size)
                pm.fill(QtCore.Qt.transparent)
                p = QtGui.QPainter(pm)
                p.setRenderHint(QtGui.QPainter.Antialiasing, True)
                pen = QtGui.QPen(color, 1.8, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
                p.setPen(pen)
                p.setBrush(QtCore.Qt.NoBrush)
                if kind == "infer":
                    p.drawRoundedRect(3, 3, 11, 11, 2, 2)
                    p.drawLine(13, 13, 17, 17)
                    p.drawEllipse(12, 12, 6, 6)
                elif kind == "train":
                    p.drawLine(3, 6, 17, 6)
                    p.drawLine(3, 14, 17, 14)
                    p.drawLine(6, 4, 6, 16)
                    p.drawLine(14, 4, 14, 16)
                elif kind == "filters":
                    p.drawRect(3, 4, 14, 12)
                    p.drawLine(6, 7, 14, 7)
                    p.drawLine(6, 10, 14, 10)
                    p.drawLine(6, 13, 14, 13)
                else:  # track
                    p.drawRoundedRect(3, 4, 14, 12, 2, 2)
                    p.drawLine(6, 8, 6, 12)
                    p.drawPolygon(
                        QtGui.QPolygon(
                            [
                                QtCore.QPoint(9, 8),
                                QtCore.QPoint(13, 10),
                                QtCore.QPoint(9, 12),
                            ]
                        )
                    )
                p.end()
                return pm

            icon = QtGui.QIcon()
            icon.addPixmap(_draw(QtGui.QColor("#111111")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            icon.addPixmap(_draw(QtGui.QColor("#ffffff")), QtGui.QIcon.Normal, QtGui.QIcon.On)
            return icon

        workspace_items = [
            ("Analyze", "infer"),
            ("Filters", "filters"),
            ("Train", "train"),
            ("Track", "track"),
        ]
        for idx, (name, icon_kind) in enumerate(workspace_items):
            btn = QtWidgets.QToolButton()
            btn.setObjectName("WorkspaceNav")
            btn.setCheckable(True)
            btn.setAutoExclusive(True)
            btn.setIcon(make_workspace_icon(icon_kind))
            btn.setIconSize(QtCore.QSize(20, 20))
            btn.setToolTip(name)
            btn.clicked.connect(lambda _=False, i=idx: self._on_workspace_tab_changed(i))
            sidebar_layout.addWidget(btn)
            self.workspace_buttons[idx] = btn
            self.workspace_button_group.addButton(btn)
        sidebar_layout.addStretch(1)
        self.workspace_sidebar_card.setFixedWidth(self.workspace_sidebar_width)

        main_content_layout.addWidget(self.workspace_sidebar_card, 0)
        sidebar_divider = QtWidgets.QFrame()
        sidebar_divider.setFrameShape(QtWidgets.QFrame.VLine)
        sidebar_divider.setFrameShadow(QtWidgets.QFrame.Plain)
        sidebar_divider.setLineWidth(1)
        sidebar_divider.setStyleSheet("QFrame { color: #d6deec; }")
        main_content_layout.addWidget(sidebar_divider, 0)
        main_content_layout.addWidget(self.body_split, 1)
        default_workspace = int(getattr(self.controller, "workspace_tab", 0))
        self._set_workspace_tab_ui(default_workspace)
        root.addWidget(main_content, stretch=1)

        # Bottom status rows are intentionally hidden to reduce visual noise.
        self.info_label = QtWidgets.QLabel("")
        self.info_label.setObjectName("Subtle")
        self.info_label.setVisible(False)
        self.output_info_label = QtWidgets.QLabel("")
        self.output_info_label.setObjectName("Subtle")
        self.output_info_label.setVisible(False)

        self.setCentralWidget(central)
        self._sync_image_header_balance()

    def _setup_shortcuts(self) -> None:
        # Global shortcuts for common actions and tab navigation.
        self._shortcuts.clear()
        bindings = [
            ("Ctrl+S", self.controller.on_save),
            ("S", self.controller.on_set),
            ("U", self.controller.on_undo),
            ("R", self.controller.on_remove_selected),
            ("C", self.controller.on_shortcut_toggle_show_current_mask),
            ("Esc", self.controller.on_clear_pending_prompts),
            ("A", self.controller.on_shortcut_open_analyze_and_calc),
            ("E", self.controller.on_shortcut_open_evaluate),
            ("[", self.controller.on_prev_image),
            ("]", self.controller.on_next_image),
        ]
        for key, cb in bindings:
            sc = QtGui.QShortcut(QtGui.QKeySequence(key), self, activated=cb)
            sc.setContext(QtCore.Qt.ApplicationShortcut)
            self._shortcuts.append(sc)
        q_sc = QtGui.QShortcut(QtGui.QKeySequence("Q"), self, activated=self.close)
        q_sc.setContext(QtCore.Qt.ApplicationShortcut)
        self._shortcuts.append(q_sc)

    def _create_infer_section(
        self,
        key: str,
        title: str,
        content: QtWidgets.QWidget,
        expanded: bool = True,
        header_widgets: Optional[Sequence[QtWidgets.QWidget]] = None,
    ) -> QtWidgets.QWidget:
        card = QtWidgets.QFrame()
        card.setObjectName("Card")
        card.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(8, 8, 8, 8)
        card_layout.setSpacing(6)

        header_row = QtWidgets.QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(6)
        header_btn = QtWidgets.QToolButton()
        header_btn.setObjectName("SectionToggle")
        header_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        header_btn.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
        header_btn.setCheckable(True)
        header_btn.setChecked(expanded)
        header_btn.setText(title)
        header_btn.toggled.connect(lambda checked=False, k=key: self._on_infer_section_toggled(k, bool(checked)))
        header_row.addWidget(header_btn)
        header_row.addStretch(1)
        for w in header_widgets or []:
            header_row.addWidget(w, 0, QtCore.Qt.AlignRight)

        content.setVisible(expanded)
        card_layout.addLayout(header_row)
        card_layout.addWidget(content, stretch=1)
        self._infer_sections[key] = (header_btn, card, content)
        return card

    def _on_infer_section_toggled(self, key: str, expanded: bool) -> None:
        section = self._infer_sections.get(key)
        if section is None:
            return
        header_btn, _card, content = section
        content.setVisible(bool(expanded))
        header_btn.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
        if expanded:
            self._infer_active_panel = "evaluate" if key == "evaluate" else "analyze"
        elif key == "evaluate" and self._infer_active_panel == "evaluate":
            self._infer_active_panel = "analyze"
        # Keep controller/UI sync behavior equivalent to previous tab-changed flow.
        self.controller.on_preview_analyze_tab_changed(0)

    def _focus_infer_section(self, key: str) -> None:
        section = self._infer_sections.get(key)
        if section is None:
            return
        header_btn, card, _content = section
        self._infer_active_panel = "evaluate" if key == "evaluate" else "analyze"
        if not header_btn.isChecked():
            header_btn.setChecked(True)
        else:
            self.controller.on_preview_analyze_tab_changed(0)
        if hasattr(self, "infer_scroll"):
            QtCore.QTimer.singleShot(0, lambda w=card: self.infer_scroll.ensureWidgetVisible(w, 0, 28))

    def _create_path_row(
        self,
        label: str,
        browse_cb,
    ) -> tuple[QtWidgets.QWidget, QtWidgets.QLineEdit]:
        row = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)
        row_layout.addWidget(QtWidgets.QLabel(label))
        edit = QtWidgets.QLineEdit()
        row_layout.addWidget(edit, stretch=1)
        btn = QtWidgets.QPushButton("Browse")
        btn.setFixedWidth(74)
        btn.clicked.connect(browse_cb)
        row_layout.addWidget(btn)
        return row, edit

    def _build_filters_workspace(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        page_layout = QtWidgets.QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(8)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        page_layout.addWidget(scroll, stretch=1)

        content = QtWidgets.QWidget()
        scroll.setWidget(content)
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(8)

        adjust_card = QtWidgets.QFrame()
        adjust_card.setObjectName("Card")
        adjust_layout = QtWidgets.QVBoxLayout(adjust_card)
        adjust_layout.setContentsMargins(10, 10, 10, 10)
        adjust_layout.setSpacing(8)

        adjust_header = QtWidgets.QHBoxLayout()
        adjust_header.setContentsMargins(0, 0, 0, 0)
        adjust_header.setSpacing(6)
        adjust_title = QtWidgets.QLabel("Image Adjust")
        adjust_title.setObjectName("PanelTitle")
        adjust_header.addWidget(adjust_title)
        adjust_header.addStretch(1)
        self.filter_adjust_reset_btn = QtWidgets.QPushButton("Reset")
        self.filter_adjust_reset_btn.clicked.connect(self.controller.on_filter_reset_adjustments)
        adjust_header.addWidget(self.filter_adjust_reset_btn)
        adjust_layout.addLayout(adjust_header)

        bright_row = QtWidgets.QHBoxLayout()
        bright_row.setContentsMargins(0, 0, 0, 0)
        bright_row.setSpacing(6)
        bright_row.addWidget(QtWidgets.QLabel("Brightness"))
        self.filter_brightness_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.filter_brightness_slider.setRange(0, 100)
        self.filter_brightness_slider.valueChanged.connect(
            lambda v: self.controller.on_filter_set_adjustment("brightness_ui", int(v))
        )
        bright_row.addWidget(self.filter_brightness_slider, stretch=1)
        self.filter_brightness_spin = QtWidgets.QSpinBox()
        self.filter_brightness_spin.setRange(0, 100)
        self.filter_brightness_spin.valueChanged.connect(
            lambda v: self.controller.on_filter_set_adjustment("brightness_ui", int(v))
        )
        bright_row.addWidget(self.filter_brightness_spin)
        adjust_layout.addLayout(bright_row)

        contrast_row = QtWidgets.QHBoxLayout()
        contrast_row.setContentsMargins(0, 0, 0, 0)
        contrast_row.setSpacing(6)
        contrast_row.addWidget(QtWidgets.QLabel("Contrast"))
        self.filter_contrast_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.filter_contrast_slider.setRange(0, 100)
        self.filter_contrast_slider.valueChanged.connect(
            lambda v: self.controller.on_filter_set_adjustment("contrast_ui", int(v))
        )
        contrast_row.addWidget(self.filter_contrast_slider, stretch=1)
        self.filter_contrast_spin = QtWidgets.QSpinBox()
        self.filter_contrast_spin.setRange(0, 100)
        self.filter_contrast_spin.valueChanged.connect(
            lambda v: self.controller.on_filter_set_adjustment("contrast_ui", int(v))
        )
        contrast_row.addWidget(self.filter_contrast_spin)
        adjust_layout.addLayout(contrast_row)

        gamma_row = QtWidgets.QHBoxLayout()
        gamma_row.setContentsMargins(0, 0, 0, 0)
        gamma_row.setSpacing(6)
        gamma_row.addWidget(QtWidgets.QLabel("Gamma"))
        self.filter_gamma_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.filter_gamma_slider.setRange(0, 100)
        self.filter_gamma_slider.valueChanged.connect(
            lambda v: self.controller.on_filter_set_adjustment("gamma_ui", int(v))
        )
        gamma_row.addWidget(self.filter_gamma_slider, stretch=1)
        self.filter_gamma_spin = QtWidgets.QSpinBox()
        self.filter_gamma_spin.setRange(0, 100)
        self.filter_gamma_spin.valueChanged.connect(
            lambda v: self.controller.on_filter_set_adjustment("gamma_ui", int(v))
        )
        gamma_row.addWidget(self.filter_gamma_spin)
        adjust_layout.addLayout(gamma_row)

        chain_card = QtWidgets.QFrame()
        chain_card.setObjectName("Card")
        chain_layout = QtWidgets.QVBoxLayout(chain_card)
        chain_layout.setContentsMargins(10, 10, 10, 10)
        chain_layout.setSpacing(8)

        header_row = QtWidgets.QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(6)
        title = QtWidgets.QLabel("Filter Chain")
        title.setObjectName("PanelTitle")
        header_row.addWidget(title)
        header_row.addStretch(1)
        self.filter_apply_all_btn = QtWidgets.QPushButton("Apply to All")
        self.filter_apply_all_btn.clicked.connect(self.controller.on_filter_apply_current_to_all)
        self.filter_reset_btn = QtWidgets.QPushButton("Reset")
        self.filter_reset_btn.clicked.connect(self.controller.on_filter_reset_current)
        header_row.addWidget(self.filter_apply_all_btn)
        header_row.addWidget(self.filter_reset_btn)
        chain_layout.addLayout(header_row)

        source_row = QtWidgets.QHBoxLayout()
        source_row.setContentsMargins(0, 0, 0, 0)
        source_row.setSpacing(6)
        source_row.addWidget(QtWidgets.QLabel("SAM Input"))
        self.filter_input_source_combo = QtWidgets.QComboBox()
        self.filter_input_source_combo.addItem("Filtered", "filtered")
        self.filter_input_source_combo.addItem("Original", "original")
        self.filter_input_source_combo.currentTextChanged.connect(
            lambda _text: self.controller.on_filter_source_changed(self.get_filter_input_source())
        )
        source_row.addWidget(self.filter_input_source_combo)
        source_row.addStretch(1)
        chain_layout.addLayout(source_row)

        self.filter_chain_list_style = """
            QListWidget#FilterChainList {
                background: #ffffff;
                border: 1px solid #d6deec;
                border-radius: 8px;
                outline: none;
            }
            QListWidget#FilterChainList::item {
                padding: 5px 8px;
                color: #1f344f;
                border: none;
            }
            QListWidget#FilterChainList::item:hover:!selected {
                background: #eef3fb;
            }
            QListWidget#FilterChainList::item:selected {
                background: #dbeafe;
                color: #0f172a;
            }
            """

        spatial_group = QtWidgets.QGroupBox("Spatial Chain")
        spatial_group_layout = QtWidgets.QVBoxLayout(spatial_group)
        spatial_group_layout.setContentsMargins(8, 8, 8, 8)
        spatial_group_layout.setSpacing(6)
        spatial_row = QtWidgets.QHBoxLayout()
        spatial_row.setContentsMargins(0, 0, 0, 0)
        spatial_row.setSpacing(6)
        self.filter_add_spatial_combo = QtWidgets.QComboBox()
        self.filter_add_spatial_combo.addItem("Gaussian", "gaussian")
        self.filter_add_spatial_combo.addItem("Median", "median")
        self.filter_add_spatial_combo.addItem("CLAHE", "clahe")
        self.filter_add_spatial_combo.addItem("Unsharp", "unsharp")
        spatial_row.addWidget(self.filter_add_spatial_combo, stretch=1)
        self.filter_add_spatial_btn = QtWidgets.QPushButton("Add")
        self.filter_add_spatial_btn.clicked.connect(
            lambda: self.controller.on_filter_add_step(
                str(self.filter_add_spatial_combo.currentData() or "gaussian")
            )
        )
        self.filter_remove_spatial_btn = QtWidgets.QPushButton("Remove")
        self.filter_remove_spatial_btn.clicked.connect(lambda: self.controller.on_filter_remove_step("spatial"))
        spatial_row.addWidget(self.filter_add_spatial_btn)
        spatial_row.addWidget(self.filter_remove_spatial_btn)
        spatial_group_layout.addLayout(spatial_row)
        self.spatial_filter_chain_list = QtWidgets.QListWidget()
        self.spatial_filter_chain_list.setObjectName("FilterChainList")
        self.spatial_filter_chain_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.spatial_filter_chain_list.setDragEnabled(True)
        self.spatial_filter_chain_list.setAcceptDrops(True)
        self.spatial_filter_chain_list.setDropIndicatorShown(True)
        self.spatial_filter_chain_list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.spatial_filter_chain_list.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.spatial_filter_chain_list.currentRowChanged.connect(
            lambda row: self.controller.on_filter_select_row("spatial", int(row))
        )
        self.spatial_filter_chain_list.model().rowsMoved.connect(
            lambda _sp, ss, se, _dp, dr: self._on_filter_rows_moved("spatial", int(ss), int(se), int(dr))
        )
        self.spatial_filter_chain_list.setMinimumHeight(140)
        self.spatial_filter_chain_list.setStyleSheet(self.filter_chain_list_style)
        spatial_group_layout.addWidget(self.spatial_filter_chain_list, stretch=1)

        frequency_group = QtWidgets.QGroupBox("Frequency Chain")
        frequency_group_layout = QtWidgets.QVBoxLayout(frequency_group)
        frequency_group_layout.setContentsMargins(8, 8, 8, 8)
        frequency_group_layout.setSpacing(6)
        freq_mode_row = QtWidgets.QHBoxLayout()
        freq_mode_row.setContentsMargins(0, 0, 0, 0)
        freq_mode_row.setSpacing(6)
        self.filter_fft_btn = QtWidgets.QPushButton("FFT")
        self.filter_fft_btn.clicked.connect(self.controller.on_filter_enter_fft_mode)
        self.filter_ifft_btn = QtWidgets.QPushButton("IFFT")
        self.filter_ifft_btn.clicked.connect(self.controller.on_filter_exit_fft_mode)
        self.filter_fft_mode_label = QtWidgets.QLabel("Spatial view")
        self.filter_fft_mode_label.setObjectName("Subtle")
        freq_mode_row.addWidget(self.filter_fft_btn)
        freq_mode_row.addWidget(self.filter_ifft_btn)
        freq_mode_row.addWidget(self.filter_fft_mode_label)
        freq_mode_row.addStretch(1)
        frequency_group_layout.addLayout(freq_mode_row)

        freq_row = QtWidgets.QHBoxLayout()
        freq_row.setContentsMargins(0, 0, 0, 0)
        freq_row.setSpacing(6)
        self.filter_add_frequency_combo = QtWidgets.QComboBox()
        self.filter_add_frequency_combo.addItem("Low-pass", "lowpass")
        self.filter_add_frequency_combo.addItem("High-pass", "highpass")
        self.filter_add_frequency_combo.addItem("Band-pass", "bandpass")
        self.filter_add_frequency_combo.addItem("Sym Notch", "sym_notch")
        freq_row.addWidget(self.filter_add_frequency_combo, stretch=1)
        self.filter_add_frequency_btn = QtWidgets.QPushButton("Add")
        self.filter_add_frequency_btn.clicked.connect(
            lambda: self.controller.on_filter_add_step(
                str(self.filter_add_frequency_combo.currentData() or "lowpass")
            )
        )
        self.filter_remove_frequency_btn = QtWidgets.QPushButton("Remove")
        self.filter_remove_frequency_btn.clicked.connect(lambda: self.controller.on_filter_remove_step("frequency"))
        freq_row.addWidget(self.filter_add_frequency_btn)
        freq_row.addWidget(self.filter_remove_frequency_btn)
        frequency_group_layout.addLayout(freq_row)

        self.frequency_filter_chain_list = QtWidgets.QListWidget()
        self.frequency_filter_chain_list.setObjectName("FilterChainList")
        self.frequency_filter_chain_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.frequency_filter_chain_list.setDragEnabled(True)
        self.frequency_filter_chain_list.setAcceptDrops(True)
        self.frequency_filter_chain_list.setDropIndicatorShown(True)
        self.frequency_filter_chain_list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.frequency_filter_chain_list.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.frequency_filter_chain_list.currentRowChanged.connect(
            lambda row: self.controller.on_filter_select_row("frequency", int(row))
        )
        self.frequency_filter_chain_list.model().rowsMoved.connect(
            lambda _sp, ss, se, _dp, dr: self._on_filter_rows_moved("frequency", int(ss), int(se), int(dr))
        )
        self.frequency_filter_chain_list.setMinimumHeight(140)
        self.frequency_filter_chain_list.setStyleSheet(
            """
            QListWidget#FilterChainList {
                background: #ffffff;
                border: 1px solid #d6deec;
                border-radius: 8px;
                outline: none;
            }
            QListWidget#FilterChainList::item {
                padding: 5px 8px;
                color: #1f344f;
                border: none;
            }
            QListWidget#FilterChainList::item:hover:!selected {
                background: #eef3fb;
            }
            QListWidget#FilterChainList::item:selected {
                background: #dbeafe;
                color: #0f172a;
            }
            """
        )
        frequency_group_layout.addWidget(self.frequency_filter_chain_list, stretch=1)
        chain_layout.addWidget(spatial_group, stretch=1)
        chain_layout.addWidget(frequency_group, stretch=1)

        editor_card = QtWidgets.QFrame()
        editor_card.setObjectName("Card")
        editor_layout = QtWidgets.QVBoxLayout(editor_card)
        editor_layout.setContentsMargins(10, 10, 10, 10)
        editor_layout.setSpacing(8)
        editor_title = QtWidgets.QLabel("Selected Filter")
        editor_title.setObjectName("PanelTitle")
        editor_layout.addWidget(editor_title)

        editor_head = QtWidgets.QHBoxLayout()
        editor_head.setContentsMargins(0, 0, 0, 0)
        editor_head.setSpacing(8)
        self.filter_kind_label = QtWidgets.QLabel("-")
        self.filter_kind_label.setObjectName("Subtle")
        editor_head.addWidget(self.filter_kind_label)
        editor_head.addStretch(1)
        editor_layout.addLayout(editor_head)

        self.filter_editor_stack = QtWidgets.QStackedWidget()
        editor_layout.addWidget(self.filter_editor_stack, stretch=1)

        page_gaussian = QtWidgets.QWidget()
        g_form = QtWidgets.QFormLayout(page_gaussian)
        g_form.setContentsMargins(0, 0, 0, 0)
        g_form.setSpacing(6)
        g_row = QtWidgets.QHBoxLayout()
        g_row.setSpacing(6)
        self.filter_gaussian_sigma_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.filter_gaussian_sigma_slider.setRange(0, 100)
        self.filter_gaussian_sigma_slider.valueChanged.connect(
            lambda v: self.controller.on_filter_set_param("sigma", float(v) / 10.0)
        )
        g_row.addWidget(self.filter_gaussian_sigma_slider, stretch=1)
        self.filter_gaussian_sigma_spin = QtWidgets.QDoubleSpinBox()
        self.filter_gaussian_sigma_spin.setRange(0.0, 10.0)
        self.filter_gaussian_sigma_spin.setSingleStep(0.1)
        self.filter_gaussian_sigma_spin.valueChanged.connect(
            lambda v: self.controller.on_filter_set_param("sigma", float(v))
        )
        g_row.addWidget(self.filter_gaussian_sigma_spin)
        g_form.addRow("Sigma", g_row)
        self.filter_editor_stack.addWidget(page_gaussian)

        page_median = QtWidgets.QWidget()
        m_form = QtWidgets.QFormLayout(page_median)
        m_form.setContentsMargins(0, 0, 0, 0)
        m_form.setSpacing(6)
        m_row = QtWidgets.QHBoxLayout()
        m_row.setSpacing(6)
        self.filter_median_ksize_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.filter_median_ksize_slider.setRange(1, 31)
        self.filter_median_ksize_slider.setSingleStep(2)
        self.filter_median_ksize_slider.valueChanged.connect(
            lambda v: self.controller.on_filter_set_param("ksize", int(v))
        )
        m_row.addWidget(self.filter_median_ksize_slider, stretch=1)
        self.filter_median_ksize_spin = QtWidgets.QSpinBox()
        self.filter_median_ksize_spin.setRange(1, 31)
        self.filter_median_ksize_spin.setSingleStep(2)
        self.filter_median_ksize_spin.valueChanged.connect(
            lambda v: self.controller.on_filter_set_param("ksize", int(v))
        )
        m_row.addWidget(self.filter_median_ksize_spin)
        m_form.addRow("Kernel", m_row)
        self.filter_editor_stack.addWidget(page_median)

        page_clahe = QtWidgets.QWidget()
        c_form = QtWidgets.QFormLayout(page_clahe)
        c_form.setContentsMargins(0, 0, 0, 0)
        c_form.setSpacing(6)
        c_row1 = QtWidgets.QHBoxLayout()
        c_row1.setSpacing(6)
        self.filter_clahe_clip_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.filter_clahe_clip_slider.setRange(1, 100)
        self.filter_clahe_clip_slider.valueChanged.connect(
            lambda v: self.controller.on_filter_set_param("clip", float(v) / 10.0)
        )
        c_row1.addWidget(self.filter_clahe_clip_slider, stretch=1)
        self.filter_clahe_clip_spin = QtWidgets.QDoubleSpinBox()
        self.filter_clahe_clip_spin.setRange(0.1, 10.0)
        self.filter_clahe_clip_spin.setSingleStep(0.1)
        self.filter_clahe_clip_spin.valueChanged.connect(
            lambda v: self.controller.on_filter_set_param("clip", float(v))
        )
        c_row1.addWidget(self.filter_clahe_clip_spin)
        c_form.addRow("Clip", c_row1)
        c_row2 = QtWidgets.QHBoxLayout()
        c_row2.setSpacing(6)
        self.filter_clahe_grid_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.filter_clahe_grid_slider.setRange(2, 32)
        self.filter_clahe_grid_slider.valueChanged.connect(
            lambda v: self.controller.on_filter_set_param("grid", int(v))
        )
        c_row2.addWidget(self.filter_clahe_grid_slider, stretch=1)
        self.filter_clahe_grid_spin = QtWidgets.QSpinBox()
        self.filter_clahe_grid_spin.setRange(2, 32)
        self.filter_clahe_grid_spin.valueChanged.connect(
            lambda v: self.controller.on_filter_set_param("grid", int(v))
        )
        c_row2.addWidget(self.filter_clahe_grid_spin)
        c_form.addRow("Grid", c_row2)
        self.filter_editor_stack.addWidget(page_clahe)

        page_unsharp = QtWidgets.QWidget()
        u_form = QtWidgets.QFormLayout(page_unsharp)
        u_form.setContentsMargins(0, 0, 0, 0)
        u_form.setSpacing(6)
        u_row1 = QtWidgets.QHBoxLayout()
        u_row1.setSpacing(6)
        self.filter_unsharp_amount_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.filter_unsharp_amount_slider.setRange(0, 50)
        self.filter_unsharp_amount_slider.valueChanged.connect(
            lambda v: self.controller.on_filter_set_param("amount", float(v) / 10.0)
        )
        u_row1.addWidget(self.filter_unsharp_amount_slider, stretch=1)
        self.filter_unsharp_amount_spin = QtWidgets.QDoubleSpinBox()
        self.filter_unsharp_amount_spin.setRange(0.0, 5.0)
        self.filter_unsharp_amount_spin.setSingleStep(0.1)
        self.filter_unsharp_amount_spin.valueChanged.connect(
            lambda v: self.controller.on_filter_set_param("amount", float(v))
        )
        u_row1.addWidget(self.filter_unsharp_amount_spin)
        u_form.addRow("Amount", u_row1)
        u_row2 = QtWidgets.QHBoxLayout()
        u_row2.setSpacing(6)
        self.filter_unsharp_sigma_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.filter_unsharp_sigma_slider.setRange(1, 100)
        self.filter_unsharp_sigma_slider.valueChanged.connect(
            lambda v: self.controller.on_filter_set_param("sigma", float(v) / 10.0)
        )
        u_row2.addWidget(self.filter_unsharp_sigma_slider, stretch=1)
        self.filter_unsharp_sigma_spin = QtWidgets.QDoubleSpinBox()
        self.filter_unsharp_sigma_spin.setRange(0.1, 10.0)
        self.filter_unsharp_sigma_spin.setSingleStep(0.1)
        self.filter_unsharp_sigma_spin.valueChanged.connect(
            lambda v: self.controller.on_filter_set_param("sigma", float(v))
        )
        u_row2.addWidget(self.filter_unsharp_sigma_spin)
        u_form.addRow("Sigma", u_row2)
        self.filter_editor_stack.addWidget(page_unsharp)

        page_lowpass = QtWidgets.QWidget()
        lp_form = QtWidgets.QFormLayout(page_lowpass)
        lp_form.setContentsMargins(0, 0, 0, 0)
        lp_form.setSpacing(6)
        lp_row = QtWidgets.QHBoxLayout()
        lp_row.setSpacing(6)
        self.filter_lowpass_cutoff_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.filter_lowpass_cutoff_slider.setRange(1, 100)
        self.filter_lowpass_cutoff_slider.valueChanged.connect(
            lambda v: self.controller.on_filter_set_param("cutoff", float(v) / 100.0)
        )
        lp_row.addWidget(self.filter_lowpass_cutoff_slider, stretch=1)
        self.filter_lowpass_cutoff_spin = QtWidgets.QSpinBox()
        self.filter_lowpass_cutoff_spin.setRange(1, 100)
        self.filter_lowpass_cutoff_spin.valueChanged.connect(
            lambda v: self.controller.on_filter_set_param("cutoff", float(v) / 100.0)
        )
        lp_row.addWidget(self.filter_lowpass_cutoff_spin)
        lp_form.addRow("Cutoff (%)", lp_row)
        self.filter_editor_stack.addWidget(page_lowpass)

        page_highpass = QtWidgets.QWidget()
        hp_form = QtWidgets.QFormLayout(page_highpass)
        hp_form.setContentsMargins(0, 0, 0, 0)
        hp_form.setSpacing(6)
        hp_row = QtWidgets.QHBoxLayout()
        hp_row.setSpacing(6)
        self.filter_highpass_cutoff_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.filter_highpass_cutoff_slider.setRange(1, 100)
        self.filter_highpass_cutoff_slider.valueChanged.connect(
            lambda v: self.controller.on_filter_set_param("cutoff", float(v) / 100.0)
        )
        hp_row.addWidget(self.filter_highpass_cutoff_slider, stretch=1)
        self.filter_highpass_cutoff_spin = QtWidgets.QSpinBox()
        self.filter_highpass_cutoff_spin.setRange(1, 100)
        self.filter_highpass_cutoff_spin.valueChanged.connect(
            lambda v: self.controller.on_filter_set_param("cutoff", float(v) / 100.0)
        )
        hp_row.addWidget(self.filter_highpass_cutoff_spin)
        hp_form.addRow("Cutoff (%)", hp_row)
        self.filter_editor_stack.addWidget(page_highpass)

        page_bandpass = QtWidgets.QWidget()
        bp_form = QtWidgets.QFormLayout(page_bandpass)
        bp_form.setContentsMargins(0, 0, 0, 0)
        bp_form.setSpacing(6)
        bp_row_inner = QtWidgets.QHBoxLayout()
        bp_row_inner.setSpacing(6)
        self.filter_bandpass_inner_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.filter_bandpass_inner_slider.setRange(0, 98)
        self.filter_bandpass_inner_slider.valueChanged.connect(
            lambda v: self.controller.on_filter_set_param("inner", float(v) / 100.0)
        )
        bp_row_inner.addWidget(self.filter_bandpass_inner_slider, stretch=1)
        self.filter_bandpass_inner_spin = QtWidgets.QSpinBox()
        self.filter_bandpass_inner_spin.setRange(0, 98)
        self.filter_bandpass_inner_spin.valueChanged.connect(
            lambda v: self.controller.on_filter_set_param("inner", float(v) / 100.0)
        )
        bp_row_inner.addWidget(self.filter_bandpass_inner_spin)
        bp_form.addRow("Inner (%)", bp_row_inner)
        bp_row_outer = QtWidgets.QHBoxLayout()
        bp_row_outer.setSpacing(6)
        self.filter_bandpass_outer_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.filter_bandpass_outer_slider.setRange(1, 100)
        self.filter_bandpass_outer_slider.valueChanged.connect(
            lambda v: self.controller.on_filter_set_param("outer", float(v) / 100.0)
        )
        bp_row_outer.addWidget(self.filter_bandpass_outer_slider, stretch=1)
        self.filter_bandpass_outer_spin = QtWidgets.QSpinBox()
        self.filter_bandpass_outer_spin.setRange(1, 100)
        self.filter_bandpass_outer_spin.valueChanged.connect(
            lambda v: self.controller.on_filter_set_param("outer", float(v) / 100.0)
        )
        bp_row_outer.addWidget(self.filter_bandpass_outer_spin)
        bp_form.addRow("Outer (%)", bp_row_outer)
        self.filter_editor_stack.addWidget(page_bandpass)

        page_notch = QtWidgets.QWidget()
        nt_form = QtWidgets.QFormLayout(page_notch)
        nt_form.setContentsMargins(0, 0, 0, 0)
        nt_form.setSpacing(6)
        nt_row1 = QtWidgets.QHBoxLayout()
        nt_row1.setSpacing(6)
        self.filter_notch_radius_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.filter_notch_radius_slider.setRange(1, 100)
        self.filter_notch_radius_slider.valueChanged.connect(
            lambda v: self.controller.on_filter_set_param("radius", float(v) / 100.0)
        )
        nt_row1.addWidget(self.filter_notch_radius_slider, stretch=1)
        self.filter_notch_radius_spin = QtWidgets.QSpinBox()
        self.filter_notch_radius_spin.setRange(1, 100)
        self.filter_notch_radius_spin.valueChanged.connect(
            lambda v: self.controller.on_filter_set_param("radius", float(v) / 100.0)
        )
        nt_row1.addWidget(self.filter_notch_radius_spin)
        nt_form.addRow("Radius (%)", nt_row1)

        nt_row2 = QtWidgets.QHBoxLayout()
        nt_row2.setSpacing(6)
        self.filter_notch_width_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.filter_notch_width_slider.setRange(1, 50)
        self.filter_notch_width_slider.valueChanged.connect(
            lambda v: self.controller.on_filter_set_param("width", float(v) / 100.0)
        )
        nt_row2.addWidget(self.filter_notch_width_slider, stretch=1)
        self.filter_notch_width_spin = QtWidgets.QSpinBox()
        self.filter_notch_width_spin.setRange(1, 50)
        self.filter_notch_width_spin.valueChanged.connect(
            lambda v: self.controller.on_filter_set_param("width", float(v) / 100.0)
        )
        nt_row2.addWidget(self.filter_notch_width_spin)
        nt_form.addRow("Mask size (%)", nt_row2)

        nt_row3 = QtWidgets.QHBoxLayout()
        nt_row3.setSpacing(6)
        self.filter_notch_angle_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.filter_notch_angle_slider.setRange(-180, 180)
        self.filter_notch_angle_slider.valueChanged.connect(
            lambda v: self.controller.on_filter_set_param("angle_deg", float(v))
        )
        nt_row3.addWidget(self.filter_notch_angle_slider, stretch=1)
        self.filter_notch_angle_spin = QtWidgets.QSpinBox()
        self.filter_notch_angle_spin.setRange(-180, 180)
        self.filter_notch_angle_spin.valueChanged.connect(
            lambda v: self.controller.on_filter_set_param("angle_deg", float(v))
        )
        nt_row3.addWidget(self.filter_notch_angle_spin)
        nt_form.addRow("Angle (deg)", nt_row3)
        self.filter_editor_stack.addWidget(page_notch)

        self._filter_editor_index_map = {
            "gaussian": 0,
            "median": 1,
            "clahe": 2,
            "unsharp": 3,
            "lowpass": 4,
            "highpass": 5,
            "bandpass": 6,
            "sym_notch": 7,
        }

        content_layout.addWidget(adjust_card, stretch=0)
        content_layout.addWidget(chain_card, stretch=1)
        content_layout.addWidget(editor_card, stretch=1)
        content_layout.addStretch(1)
        return page

    def _build_train_workspace(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        page_layout = QtWidgets.QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(8)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        page_layout.addWidget(scroll, stretch=1)

        content = QtWidgets.QWidget()
        scroll.setWidget(content)
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        monitor = QtWidgets.QFrame()
        monitor.setObjectName("Card")
        monitor_layout = QtWidgets.QVBoxLayout(monitor)
        monitor_layout.setContentsMargins(10, 10, 10, 10)
        monitor_layout.setSpacing(8)

        monitor_header = QtWidgets.QHBoxLayout()
        monitor_header.setContentsMargins(0, 0, 0, 0)
        monitor_header.setSpacing(6)
        monitor_title = QtWidgets.QLabel("Training Monitor")
        monitor_title.setObjectName("PanelTitle")
        monitor_header.addWidget(monitor_title)
        self.train_status_label = QtWidgets.QLabel("Idle")
        self.train_status_label.setObjectName("Subtle")
        monitor_header.addWidget(self.train_status_label, 0, QtCore.Qt.AlignLeft)
        monitor_header.addStretch(1)
        self.train_run_btn = QtWidgets.QPushButton("Run Train")
        self.train_run_btn.clicked.connect(self.controller.on_train_run)
        self.train_run_btn.setMinimumWidth(88)
        self.train_stop_btn = QtWidgets.QPushButton("Stop")
        self.train_stop_btn.clicked.connect(self.controller.on_train_stop)
        self.train_stop_btn.setEnabled(False)
        self.train_stop_btn.setMinimumWidth(88)
        monitor_header.addWidget(self.train_run_btn)
        monitor_header.addWidget(self.train_stop_btn)
        monitor_layout.addLayout(monitor_header)

        self.train_progress_bar = QtWidgets.QProgressBar()
        self.train_progress_bar.setRange(0, 1)
        self.train_progress_bar.setValue(0)
        self.train_progress_bar.setFormat("0/0")
        monitor_layout.addWidget(self.train_progress_bar)

        self.train_run_dir_label = QtWidgets.QLabel("Run dir: -")
        self.train_run_dir_label.setObjectName("Subtle")
        self.train_run_dir_label.setWordWrap(True)
        monitor_layout.addWidget(self.train_run_dir_label)

        self.train_metrics_summary_label = QtWidgets.QLabel("train: -, val: -, test: -")
        self.train_metrics_summary_label.setObjectName("Subtle")
        self.train_metrics_summary_label.setWordWrap(True)
        monitor_layout.addWidget(self.train_metrics_summary_label)

        self.train_graph_label = QtWidgets.QLabel("Run Train to view learning curves")
        self.train_graph_label.setAlignment(QtCore.Qt.AlignCenter)
        self.train_graph_label.setScaledContents(False)
        self.train_graph_label.setMinimumHeight(250)
        self.train_graph_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.train_graph_label.setStyleSheet(
            "QLabel { background: #f8fafc; border: 1px dashed #cfd9e7; border-radius: 8px; color: #4b5563; }"
        )
        monitor_layout.addWidget(self.train_graph_label)

        self.train_cmd_label = QtWidgets.QLabel("Command: -")
        self.train_cmd_label.setObjectName("Subtle")
        self.train_cmd_label.setWordWrap(True)
        monitor_layout.addWidget(self.train_cmd_label)
        layout.addWidget(monitor, stretch=0)

        controls = QtWidgets.QFrame()
        controls.setObjectName("Card")
        controls_layout = QtWidgets.QVBoxLayout(controls)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        controls_layout.setSpacing(8)
        controls_title = QtWidgets.QLabel("Training Settings")
        controls_title.setObjectName("PanelTitle")
        controls_layout.addWidget(controls_title)

        data_group = QtWidgets.QGroupBox("Data")
        data_layout = QtWidgets.QVBoxLayout(data_group)
        data_layout.setContentsMargins(8, 8, 8, 8)
        data_layout.setSpacing(6)
        row, self.train_image_dir_edit = self._create_path_row("Image dir", self.controller.on_train_browse_image_dir)
        data_layout.addWidget(row)
        row, self.train_mask_dir_edit = self._create_path_row("Mask dir", self.controller.on_train_browse_mask_dir)
        data_layout.addWidget(row)
        self.train_image_dir_edit.textChanged.connect(self.controller.on_train_preview_inputs_changed)
        self.train_mask_dir_edit.textChanged.connect(self.controller.on_train_preview_inputs_changed)
        row, self.train_output_dir_edit = self._create_path_row("Output dir", self.controller.on_train_browse_output_dir)
        data_layout.addWidget(row)
        row, self.train_sweep_edit = self._create_path_row("Sweep JSON", self.controller.on_train_browse_sweep_json)
        data_layout.addWidget(row)
        controls_layout.addWidget(data_group)

        model_group = QtWidgets.QGroupBox("Model")
        model_layout = QtWidgets.QVBoxLayout(model_group)
        model_layout.setContentsMargins(8, 8, 8, 8)
        model_layout.setSpacing(6)
        h1 = QtWidgets.QHBoxLayout()
        h1.setSpacing(6)
        h1.addWidget(QtWidgets.QLabel("Backend"))
        self.train_backend_combo = QtWidgets.QComboBox()
        self.train_backend_combo.addItems(["hf", "meta"])
        self.train_backend_combo.currentTextChanged.connect(self._on_train_backend_changed)
        h1.addWidget(self.train_backend_combo)
        h1.addWidget(QtWidgets.QLabel("Model type"))
        self.train_sam_model_combo = QtWidgets.QComboBox()
        self.train_sam_model_combo.addItems(["vit_b", "vit_l", "vit_h"])
        h1.addWidget(self.train_sam_model_combo)
        h1.addWidget(QtWidgets.QLabel("HF model"))
        self.train_hf_model_edit = QtWidgets.QLineEdit("facebook/sam-vit-base")
        h1.addWidget(self.train_hf_model_edit, stretch=1)
        h1.addWidget(QtWidgets.QLabel("HF size"))
        self.train_hf_input_size_spin = QtWidgets.QSpinBox()
        self.train_hf_input_size_spin.setRange(0, 4096)
        self.train_hf_input_size_spin.setValue(1024)
        h1.addWidget(self.train_hf_input_size_spin)
        h1.addStretch(1)
        model_layout.addLayout(h1)

        row, self.train_sam_ckpt_edit = self._create_path_row("SAM ckpt", self.controller.on_train_browse_sam_checkpoint)
        model_layout.addWidget(row)
        controls_layout.addWidget(model_group)

        optim_group = QtWidgets.QGroupBox("Optimization")
        optim_layout = QtWidgets.QVBoxLayout(optim_group)
        optim_layout.setContentsMargins(8, 8, 8, 8)
        optim_layout.setSpacing(6)
        h2 = QtWidgets.QHBoxLayout()
        h2.setSpacing(6)
        h2.addWidget(QtWidgets.QLabel("Epochs"))
        self.train_epochs_spin = QtWidgets.QSpinBox()
        self.train_epochs_spin.setRange(1, 20000)
        self.train_epochs_spin.setValue(50)
        h2.addWidget(self.train_epochs_spin)
        h2.addWidget(QtWidgets.QLabel("Batch"))
        self.train_batch_spin = QtWidgets.QSpinBox()
        self.train_batch_spin.setRange(1, 256)
        self.train_batch_spin.setValue(1)
        h2.addWidget(self.train_batch_spin)
        h2.addWidget(QtWidgets.QLabel("LR"))
        self.train_lr_edit = QtWidgets.QLineEdit("5e-5")
        self.train_lr_edit.setFixedWidth(92)
        h2.addWidget(self.train_lr_edit)
        h2.addWidget(QtWidgets.QLabel("Weight decay"))
        self.train_weight_decay_edit = QtWidgets.QLineEdit("0.0")
        self.train_weight_decay_edit.setFixedWidth(92)
        h2.addWidget(self.train_weight_decay_edit)
        h2.addWidget(QtWidgets.QLabel("Eval every"))
        self.train_eval_every_spin = QtWidgets.QSpinBox()
        self.train_eval_every_spin.setRange(1, 1000)
        self.train_eval_every_spin.setValue(1)
        h2.addWidget(self.train_eval_every_spin)
        h2.addStretch(1)
        optim_layout.addLayout(h2)
        controls_layout.addWidget(optim_group)

        lora_group = QtWidgets.QGroupBox("LoRA")
        lora_layout = QtWidgets.QVBoxLayout(lora_group)
        lora_layout.setContentsMargins(8, 8, 8, 8)
        lora_layout.setSpacing(6)
        h3 = QtWidgets.QHBoxLayout()
        h3.setSpacing(6)
        h3.addWidget(QtWidgets.QLabel("LoRA rank"))
        self.train_rank_spin = QtWidgets.QSpinBox()
        self.train_rank_spin.setRange(1, 1024)
        self.train_rank_spin.setValue(16)
        h3.addWidget(self.train_rank_spin)
        h3.addWidget(QtWidgets.QLabel("alpha"))
        self.train_alpha_edit = QtWidgets.QLineEdit("16.0")
        self.train_alpha_edit.setFixedWidth(74)
        h3.addWidget(self.train_alpha_edit)
        h3.addWidget(QtWidgets.QLabel("dropout"))
        self.train_dropout_edit = QtWidgets.QLineEdit("0.1")
        self.train_dropout_edit.setFixedWidth(74)
        h3.addWidget(self.train_dropout_edit)
        h3.addWidget(QtWidgets.QLabel("Sample"))
        self.train_sample_mode_combo = QtWidgets.QComboBox()
        self.train_sample_mode_combo.addItems(["instance", "instance_all", "image"])
        h3.addWidget(self.train_sample_mode_combo)
        h3.addWidget(QtWidgets.QLabel("Freeze"))
        self.train_freeze_combo = QtWidgets.QComboBox()
        self.train_freeze_combo.addItems(["vit_prompt", "prompt_mask", "none", "custom"])
        h3.addWidget(self.train_freeze_combo)
        h3.addStretch(1)
        lora_layout.addLayout(h3)

        h4 = QtWidgets.QHBoxLayout()
        h4.setSpacing(6)
        h4.addWidget(QtWidgets.QLabel("LoRA targets"))
        self.train_lora_targets_edit = QtWidgets.QLineEdit("q_proj k_proj v_proj out_proj qkv proj")
        h4.addWidget(self.train_lora_targets_edit, stretch=1)
        lora_layout.addLayout(h4)
        controls_layout.addWidget(lora_group)

        adv_group = QtWidgets.QGroupBox("Advanced")
        adv_layout = QtWidgets.QVBoxLayout(adv_group)
        adv_layout.setContentsMargins(8, 8, 8, 8)
        adv_layout.setSpacing(6)
        h5 = QtWidgets.QHBoxLayout()
        h5.setSpacing(6)
        h5.addWidget(QtWidgets.QLabel("Extra args"))
        self.train_extra_args_edit = QtWidgets.QLineEdit("")
        self.train_extra_args_edit.setPlaceholderText("--save-best --num-workers 4")
        h5.addWidget(self.train_extra_args_edit, stretch=1)
        adv_layout.addLayout(h5)
        controls_layout.addWidget(adv_group)
        controls_layout.addStretch(1)
        layout.addWidget(controls, stretch=0)

        logs_card = QtWidgets.QFrame()
        logs_card.setObjectName("Card")
        logs_layout = QtWidgets.QVBoxLayout(logs_card)
        logs_layout.setContentsMargins(10, 10, 10, 10)
        logs_layout.setSpacing(8)
        logs_title = QtWidgets.QLabel("Logs")
        logs_title.setObjectName("PanelTitle")
        logs_layout.addWidget(logs_title)
        self.train_log_edit = QtWidgets.QPlainTextEdit()
        self.train_log_edit.setReadOnly(True)
        self.train_log_edit.setPlaceholderText("train logs...")
        self.train_log_edit.setMinimumHeight(200)
        logs_layout.addWidget(self.train_log_edit, stretch=1)
        layout.addWidget(logs_card, stretch=1)

        self._on_train_backend_changed(self.train_backend_combo.currentText())
        self.set_train_status_text("Idle", level="idle")
        self.set_train_progress(0, 0)
        self.set_train_graph_pixmap(None)
        return page

    def _build_evaluation_workspace(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        controls_layout.setSpacing(6)

        path_row = QtWidgets.QHBoxLayout()
        path_row.setContentsMargins(0, 0, 0, 0)
        path_row.setSpacing(6)
        gt_label = QtWidgets.QLabel("GT Path")
        gt_label.setObjectName("PanelTitle")
        path_row.addWidget(gt_label)
        self.eval_gt_roi_edit = QtWidgets.QLineEdit()
        path_row.addWidget(self.eval_gt_roi_edit, stretch=1)
        self.eval_gt_roi_browse_btn = QtWidgets.QPushButton("Browse")
        self.eval_gt_roi_browse_btn.setFixedWidth(74)
        self.eval_gt_roi_browse_btn.clicked.connect(self.controller.on_eval_browse_gt_roi_path)
        path_row.addWidget(self.eval_gt_roi_browse_btn, 0, QtCore.Qt.AlignRight)
        self.eval_run_current_btn = QtWidgets.QPushButton("Eval Current")
        self.eval_run_current_btn.clicked.connect(self.controller.on_eval_run_current)
        self.eval_run_all_btn = QtWidgets.QPushButton("Eval All")
        self.eval_run_all_btn.clicked.connect(self.controller.on_eval_run_all)
        # Legacy alias for backward compatibility with existing controller paths.
        self.eval_run_btn = self.eval_run_current_btn
        controls_layout.addLayout(path_row)
        self.eval_gt_roi_edit.textChanged.connect(self.controller.on_eval_gt_roi_path_changed)

        self.eval_status_label = QtWidgets.QLabel("")
        self.eval_status_label.setObjectName("Subtle")
        self.eval_status_label.setWordWrap(True)
        self.eval_status_label.setVisible(False)

        plots_card = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(plots_card)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(8)

        self.eval_plot_label = QtWidgets.QLabel(self._eval_plot_placeholders["current"])
        self.eval_plot_label.setAlignment(QtCore.Qt.AlignCenter)
        self.eval_plot_label.setScaledContents(False)
        self.eval_plot_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.eval_plot_label.setMinimumHeight(260)
        self.eval_plot_label.setStyleSheet(
            "QLabel { background: #f8fafc; border: 1px dashed #cfd9e7; border-radius: 8px; color: #4b5563; }"
        )
        right_layout.addWidget(self.eval_plot_label, stretch=1)
        # Keep GT path controls directly under Validation header (Eval buttons are in section header).
        layout.addWidget(controls, stretch=0)
        layout.addWidget(plots_card, stretch=1)
        # Backward-compatible aliases used by existing controller helper paths.
        self.eval_current_plot_label = self.eval_plot_label
        self.eval_all_plot_label = self.eval_plot_label
        self.set_eval_scope("current")
        return page

    def _build_track_workspace(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        controls_layout.setSpacing(6)
        title = QtWidgets.QLabel("Track")
        title.setObjectName("PanelTitle")
        controls_layout.addWidget(title)

        row, self.track_image_edit = self._create_path_row("Track image", self.controller.on_track_browse_image)
        controls_layout.addWidget(row)
        self.track_image_edit.textChanged.connect(self.controller.on_track_preview_input_changed)

        note = QtWidgets.QLabel(
            "Import a track-specific image (or video file). "
            "This preview is independent from Segmentation and Train images."
        )
        note.setObjectName("Subtle")
        note.setWordWrap(True)
        controls_layout.addWidget(note)
        controls_layout.addStretch(1)

        layout.addWidget(controls, stretch=0)
        layout.addStretch(1)
        return page

    def _on_train_backend_changed(self, backend: str) -> None:
        b = (backend or "").strip().lower()
        is_meta = b == "meta"
        self.train_sam_ckpt_edit.setEnabled(is_meta)
        self.train_sam_model_combo.setEnabled(is_meta)
        self.train_hf_model_edit.setEnabled(not is_meta)
        self.train_hf_input_size_spin.setEnabled(not is_meta)

    def _on_mode_sam(self) -> None:
        if self.mode_sam_btn.isChecked():
            self.controller.on_change_mode("sam")

    def _on_mode_lora(self) -> None:
        if self.mode_lora_btn.isChecked():
            self.controller.on_change_mode("lora")

    def _on_mode_polygon(self) -> None:
        if self.mode_polygon_btn.isChecked():
            self.controller.on_change_mode("polygon")

    def _set_workspace_tab_ui(self, index: int) -> None:
        if not hasattr(self, "workspace_stack"):
            return
        if index < 0 or index >= self.workspace_stack.count():
            return
        if self.workspace_stack.currentIndex() != index:
            self.workspace_stack.setCurrentIndex(index)
        btn = self.workspace_buttons.get(index)
        if btn is not None and not btn.isChecked():
            with QtCore.QSignalBlocker(btn):
                btn.setChecked(True)
        for idx, action in self.workspace_menu_actions.items():
            with QtCore.QSignalBlocker(action):
                action.setChecked(idx == index)

    def _on_workspace_tab_changed(self, index: int) -> None:
        self._set_workspace_tab_ui(index)
        self.controller.on_workspace_tab_changed(index)

    def adapt_layout_for_image(self, image_w: int, image_h: int) -> None:
        if image_w <= 0 or image_h <= 0:
            return
        self._last_image_size = (int(image_w), int(image_h))
        win_w = max(1, self.width())
        win_h = max(1, self.height())
        profile = f"{int(image_w)}x{int(image_h)}:{win_w // 80}:{win_h // 80}"

        body_w = self.body_split.size().width()
        body_h = self.body_split.size().height()
        if body_w > 0 and self._body_layout_profile != profile:
            aspect = float(image_w) / float(max(1, image_h))
            # Estimate usable image height from actual visible controls.
            # This keeps image size responsive when Display/Scale layout changes.
            reserved_h = 0
            for name in ("image_title_strip", "image_toolbar_card", "image_actions_card", "display_strip", "scale_strip"):
                w = getattr(self, name, None)
                if isinstance(w, QtWidgets.QWidget) and w.isVisible():
                    reserved_h += int(w.sizeHint().height())
            required_left_w = 0
            for name in ("image_title_strip", "image_toolbar_card", "image_header_strip", "image_actions_card", "display_strip", "scale_strip"):
                w = getattr(self, name, None)
                if isinstance(w, QtWidgets.QWidget) and w.isVisible():
                    required_left_w = max(required_left_w, int(w.sizeHint().width()))
            top_row_h = max(
                int(self.btn_prev_image.sizeHint().height()),
                int(self.image_index_label.sizeHint().height()),
                int(self.image_name_label.sizeHint().height()),
            )
            reserved_h += top_row_h
            # Card margins/paddings/spacings and safety slack.
            # Keep extra room so image area does not overlap Set/Options/Scale strips.
            reserved_h += 76
            target_img_h = max(140, int(body_h) - reserved_h)
            self.image_label.setFixedHeight(target_img_h)
            target_img_w = int(round(target_img_h * aspect))
            if required_left_w > 0:
                required_left_w = min(required_left_w, target_img_w + 24)
            # Prioritize image height fit first, then minimize side blanks.
            # If needed, shrink the right pane to keep enough width for the image.
            edge_padding = 8 if aspect <= 1.15 else 12
            min_left = max(320, required_left_w + 12)
            left_needed = max(min_left, target_img_w + edge_padding)
            right_floor = 320
            right_comfort = 360
            right_min = right_comfort
            if body_w < (left_needed + right_comfort):
                right_min = right_floor
            max_left = max(min_left, body_w - right_min)
            left_px = min(left_needed, max_left)
            right_px = max(right_min, body_w - left_px)
            if left_px + right_px > body_w:
                left_px = max(min_left, body_w - right_px)
            self.body_split.setSizes([left_px, right_px])
            max_content_w = max(120, int(left_px) - 24)
            content_w = max(120, min(int(target_img_w), max_content_w))
            applied_w = self.set_left_content_width(content_w)
            # After width is constrained by layout, re-fit height to avoid vertical blank.
            # Keep height-first behavior but cap by width/aspect when width is tighter.
            if applied_w > 0 and aspect > 0.0:
                fitted_h = max(140, min(int(target_img_h), int(round(float(applied_w) / aspect))))
                self.image_label.setFixedHeight(fitted_h)
            self._sync_top_cards_with_body_split()
            self._body_layout_profile = profile

    def _sync_top_cards_with_body_split(self, *_args) -> None:
        return

    def _lock_body_splitter_handle(self) -> None:
        if not hasattr(self, "body_split"):
            return
        # Keep responsive ratio via setSizes(), but prevent manual drag-resize.
        self.body_split.setHandleWidth(14)
        handle = self.body_split.handle(1)
        if handle is None:
            return
        handle.setStyleSheet("QWidget { background: #edf2f9; }")
        handle.setCursor(QtCore.Qt.ArrowCursor)
        handle.setEnabled(False)
        handle.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)

    def _sync_image_header_balance(self) -> None:
        row_w = 0
        if hasattr(self, "image_header_strip"):
            row_w = int(self.image_header_strip.width())
        if row_w <= 0:
            row_w = self.left_card.width()
        center_w = 300
        if row_w > 0:
            center_w = max(170, min(560, int(row_w * 0.44)))
        self.image_name_label.setFixedWidth(center_w)
        side_w = max(
            86,
            self.image_index_label.sizeHint().width(),
            self.image_masks_label.sizeHint().width() if hasattr(self, "image_masks_label") else 0,
        )
        self.image_index_label.setFixedWidth(side_w)
        if hasattr(self, "image_masks_label"):
            self.image_masks_label.setFixedWidth(side_w)
        self._update_image_name_display()

    def _update_image_name_display(self) -> None:
        full = self._image_name_full or "-"
        avail = max(40, self.image_name_label.width() - 6)
        fm = self.image_name_label.fontMetrics()
        elided = fm.elidedText(full, QtCore.Qt.ElideMiddle, avail)
        self.image_name_label.setText(elided)
        self.image_name_label.setToolTip(full if elided != full else "")

    # Access helpers for controller --------------------------------------------
    def set_scale_text(self, text: str) -> None:
        # Legacy API kept for controller compatibility after scale-input redesign.
        self._scale_text_cache = str(text)

    def get_scale_text(self) -> str:
        return str(getattr(self, "_scale_text_cache", ""))

    def set_display_unit(self, unit: str) -> None:
        u = (unit or "nm").strip().lower().replace("µ", "u").replace("μ", "u")
        if u not in {"nm", "um", "mm"}:
            u = "nm"
        idx = {"nm": 0, "um": 1, "mm": 2}[u]
        with QtCore.QSignalBlocker(self.display_unit_combo):
            self.display_unit_combo.setCurrentIndex(idx)

    def get_display_unit(self) -> str:
        return self.display_unit_combo.currentText().strip()

    def set_scale_calibration_mode(self, enabled: bool) -> None:
        with QtCore.QSignalBlocker(self.scale_calib_toggle_btn):
            self.scale_calib_toggle_btn.setChecked(bool(enabled))
        self.scale_calib_toggle_btn.setText("Measuring..." if enabled else "Measure")

    def set_scale_bar_length_text(self, text: str) -> None:
        self.scale_bar_length_edit.setText(text)

    def get_scale_bar_length_text(self) -> str:
        return self.scale_bar_length_edit.text()

    def set_scale_px_text(self, text: str) -> None:
        self.scale_px_edit.setText(str(text))

    def get_scale_px_text(self) -> str:
        return self.scale_px_edit.text()

    def set_scale_preset_slider_value(self, value: int) -> None:
        if not hasattr(self, "scale_preset_slider"):
            return
        with QtCore.QSignalBlocker(self.scale_preset_slider):
            self.scale_preset_slider.setValue(int(value))

    def set_scale_preset_hint_text(self, text: str) -> None:
        if hasattr(self, "scale_preset_hint_label"):
            self.scale_preset_hint_label.setText(str(text))

    def set_scale_preset_ticks(self, ticks: Sequence[tuple[float, str]]) -> None:
        if hasattr(self, "scale_preset_ruler"):
            self.scale_preset_ruler.set_ticks(ticks)

    def set_scale_calibration_pixels(self, px: Optional[float]) -> None:
        if px is None:
            self.scale_px_edit.clear()
            return
        val = float(px)
        self.scale_px_edit.setText(f"{val:.3f}")

    def set_cluster_text(self, text: str) -> None:
        self.cluster_edit.setText(text)

    def get_cluster_text(self) -> str:
        return self.cluster_edit.text()

    def set_cluster_label_text(self, text: str) -> None:
        self.cluster_label.setText(str(text))

    def set_cluster_stats_summary(
        self,
        count_value: Optional[int],
        mean_value: Optional[float],
        min_value: Optional[float],
        max_value: Optional[float],
    ) -> None:
        def _fmt(v: Optional[float], digits: int) -> str:
            if v is None:
                return "-"
            return f"{float(v):.{digits}f}"

        def _fmt_count(v: Optional[int]) -> str:
            if v is None:
                return "-"
            return f"{int(v)}"

        self.cluster_count_label.setText(f"count:{_fmt_count(count_value)}")
        self.cluster_mean_label.setText(f"mean:{_fmt(mean_value, 1)}")
        self.cluster_min_label.setText(f"min:{_fmt(min_value, 0)}")
        self.cluster_max_label.setText(f"max:{_fmt(max_value, 0)}")

    def set_fractal_checked(self, checked: bool) -> None:
        self.fractal_checkbox.setChecked(checked)

    def set_realtime_calc_checked(self, checked: bool) -> None:
        self.realtime_calc_checkbox.setChecked(checked)

    def set_overlay_centroid_checked(self, checked: bool) -> None:
        self.overlay_centroid_checkbox.setChecked(checked)

    def set_show_current_checked(self, checked: bool) -> None:
        self.show_current_checkbox.setChecked(checked)

    def set_show_set_checked(self, checked: bool) -> None:
        self.show_set_checkbox.setChecked(checked)

    def set_show_prompts_checked(self, checked: bool) -> None:
        self.show_prompts_checkbox.setChecked(checked)

    def set_include_zero_checked(self, checked: bool) -> None:
        self.include_zero_checkbox.setChecked(checked)

    def set_show_nearest_checked(self, checked: bool) -> None:
        self.show_nearest_checkbox.setChecked(checked)

    def set_show_gt_checked(self, checked: bool) -> None:
        self.show_gt_checkbox.setChecked(checked)

    def set_show_bbox_checked(self, checked: bool) -> None:
        self.show_bbox_checkbox.setChecked(checked)

    def set_show_axes_checked(self, checked: bool) -> None:
        self.show_feret_checkbox.setChecked(checked)

    def set_show_feret_parallelogram_checked(self, checked: bool) -> None:
        self.show_feret_checkbox.setChecked(checked)

    def set_show_feret_checked(self, checked: bool) -> None:
        self.show_feret_checkbox.setChecked(checked)

    def set_show_ellipse_checked(self, checked: bool) -> None:
        self.show_ellipse_checkbox.setChecked(checked)

    def get_distribution_metric(self) -> str:
        data = self.distribution_metric_combo.currentData()
        return str(data) if data else "none"

    def get_main_graph_metric(self) -> str:
        data = self.main_graph_metric_combo.currentData()
        return str(data) if data else "ecd"

    def get_size_hist_metric(self) -> str:
        data = self.size_hist_metric_combo.currentData()
        return str(data) if data else "ecd"

    def get_nearest_hist_metric(self) -> str:
        data = self.nearest_hist_metric_combo.currentData()
        return str(data) if data else "nearest1"

    def set_nearest_hist_metric(self, metric: str) -> None:
        key = str(metric or "nearest1").strip().lower()
        for i in range(self.nearest_hist_metric_combo.count()):
            data = str(self.nearest_hist_metric_combo.itemData(i) or "").strip().lower()
            if data == key:
                with QtCore.QSignalBlocker(self.nearest_hist_metric_combo):
                    self.nearest_hist_metric_combo.setCurrentIndex(i)
                return
        with QtCore.QSignalBlocker(self.nearest_hist_metric_combo):
            self.nearest_hist_metric_combo.setCurrentIndex(0)

    def set_size_hist_metric(self, metric: str) -> None:
        key = str(metric or "ecd").strip().lower()
        for i in range(self.size_hist_metric_combo.count()):
            data = str(self.size_hist_metric_combo.itemData(i) or "").strip().lower()
            if data == key:
                with QtCore.QSignalBlocker(self.size_hist_metric_combo):
                    self.size_hist_metric_combo.setCurrentIndex(i)
                return
        with QtCore.QSignalBlocker(self.size_hist_metric_combo):
            self.size_hist_metric_combo.setCurrentIndex(0)

    def get_area_hist_metric(self) -> str:
        data = self.area_hist_metric_combo.currentData()
        return str(data) if data else "area"

    def get_aspect_hist_metric(self) -> str:
        data = self.aspect_hist_metric_combo.currentData()
        return str(data) if data else "feret"

    def set_area_hist_metric(self, metric: str) -> None:
        key = str(metric or "area").strip().lower()
        for i in range(self.area_hist_metric_combo.count()):
            data = str(self.area_hist_metric_combo.itemData(i) or "").strip().lower()
            if data == key:
                with QtCore.QSignalBlocker(self.area_hist_metric_combo):
                    self.area_hist_metric_combo.setCurrentIndex(i)
                return
        with QtCore.QSignalBlocker(self.area_hist_metric_combo):
            self.area_hist_metric_combo.setCurrentIndex(0)

    def set_aspect_hist_metric(self, metric: str) -> None:
        key = str(metric or "feret").strip().lower()
        for i in range(self.aspect_hist_metric_combo.count()):
            data = str(self.aspect_hist_metric_combo.itemData(i) or "").strip().lower()
            if data == key:
                with QtCore.QSignalBlocker(self.aspect_hist_metric_combo):
                    self.aspect_hist_metric_combo.setCurrentIndex(i)
                return
        with QtCore.QSignalBlocker(self.aspect_hist_metric_combo):
            self.aspect_hist_metric_combo.setCurrentIndex(0)

    def set_distribution_metric(self, metric: str) -> None:
        key = str(metric or "none").strip().lower()
        for i in range(self.distribution_metric_combo.count()):
            data = str(self.distribution_metric_combo.itemData(i) or "").strip().lower()
            if data == key:
                with QtCore.QSignalBlocker(self.distribution_metric_combo):
                    self.distribution_metric_combo.setCurrentIndex(i)
                return
        with QtCore.QSignalBlocker(self.distribution_metric_combo):
            self.distribution_metric_combo.setCurrentIndex(0)

    def set_main_graph_metric(self, metric: str) -> None:
        key = str(metric or "ecd").strip().lower()
        for i in range(self.main_graph_metric_combo.count()):
            data = str(self.main_graph_metric_combo.itemData(i) or "").strip().lower()
            if data == key:
                with QtCore.QSignalBlocker(self.main_graph_metric_combo):
                    self.main_graph_metric_combo.setCurrentIndex(i)
                return
        with QtCore.QSignalBlocker(self.main_graph_metric_combo):
            self.main_graph_metric_combo.setCurrentIndex(2)

    def get_distribution_edges_text(self) -> str:
        return self.distribution_edges_edit.text()

    def set_distribution_edges_text(self, text: str) -> None:
        self.distribution_edges_edit.setText(str(text or ""))

    def get_distribution_bins_count(self) -> int:
        if not hasattr(self, "distribution_bins_spin"):
            return 3
        return int(self.distribution_bins_spin.value())

    def set_distribution_bins_count(self, count: int) -> None:
        if not hasattr(self, "distribution_bins_spin"):
            return
        v = max(2, min(20, int(count)))
        with QtCore.QSignalBlocker(self.distribution_bins_spin):
            self.distribution_bins_spin.setValue(v)

    def set_distribution_bins_enabled(self, enabled: bool) -> None:
        if hasattr(self, "distribution_bins_spin"):
            self.distribution_bins_spin.setEnabled(bool(enabled))
        if hasattr(self, "distribution_bins_minus_btn"):
            self.distribution_bins_minus_btn.setEnabled(bool(enabled))
        if hasattr(self, "distribution_bins_plus_btn"):
            self.distribution_bins_plus_btn.setEnabled(bool(enabled))

    def set_distribution_slider_state(
        self,
        minimum: float,
        maximum: float,
        values: Sequence[float],
        enabled: bool,
    ) -> None:
        if not hasattr(self, "distribution_slider"):
            return
        slider = self.distribution_slider
        with QtCore.QSignalBlocker(slider):
            slider.setEnabled(bool(enabled))
            slider.setRange(float(minimum), float(maximum))
            slider.setValues(values)
        slider.update()

    def get_distribution_slider_values(self) -> list[float]:
        if not hasattr(self, "distribution_slider"):
            return []
        return self.distribution_slider.values()

    def set_distribution_status(self, text: str, error: bool = False) -> None:
        # Status text is intentionally hidden to keep Distribution row compact.
        _ = (text, error)
        if hasattr(self, "distribution_status_label"):
            self.distribution_status_label.clear()
            self.distribution_status_label.setVisible(False)

    def set_review_table_unit(self, length_unit: str) -> None:
        _ = length_unit
        self.review_table.setHorizontalHeaderLabels(
            [
                "ECD",
                "VESD",
                "Area(px)",
                "Area(BBox)",
                "Area(VESD)",
                "Aspect(Feret)",
                "Aspect(Ellipse)",
                "Feret Maj",
                "Feret Min",
                "Ellipse Maj",
                "Ellipse Min",
                "Cent X",
                "Cent Y",
                "BBox X",
                "BBox Y",
                "BBox Cx",
                "BBox Cy",
                "Score",
            ]
        )

    def set_show_gt_enabled(self, enabled: bool) -> None:
        self.show_gt_checkbox.setEnabled(enabled)

    def set_lora_mode_enabled(self, enabled: bool) -> None:
        self.mode_lora_btn.setEnabled(enabled)

    def set_lora_checkpoint_text(self, text: str) -> None:
        self.lora_path_edit.setText(text)

    def get_lora_checkpoint_text(self) -> str:
        return self.lora_path_edit.text()

    def set_lora_runtime_status(self, text: str = "", loaded: Optional[bool] = None) -> None:
        # Runtime indicator is intentionally hidden; mode availability communicates LoRA readiness.
        _ = (text, loaded)
        self.lora_state_label.setVisible(False)

    def set_mode(self, mode: str) -> None:
        m = (mode or "sam").lower()
        if m == "lora" and not self.mode_lora_btn.isEnabled():
            m = "sam"
        self.mode_sam_btn.setChecked(m == "sam")
        self.mode_lora_btn.setChecked(m == "lora")
        self.mode_polygon_btn.setChecked(m == "polygon")
        self.set_action_mode(m)

    def set_polygon_mode_checked(self, checked: bool) -> None:
        self.set_mode("polygon" if checked else "sam")

    def set_mask_info_text(self, text: str) -> None:
        self.mask_info_label.setText(text)

    def set_info_text(self, text: str) -> None:
        self.info_label.setText(text)

    def set_options_image_side_padding(self, side_px: int) -> None:
        self._options_image_side_px = max(0, int(side_px))
        self._apply_options_side_padding()

    def set_image_header_side_padding(self, side_px: int) -> None:
        self._header_image_side_px = max(0, int(side_px))
        self._apply_image_header_side_padding()

    def set_left_content_width(self, content_w: int) -> int:
        if content_w <= 0:
            return 0
        card_w = int(self.left_card.width()) if hasattr(self, "left_card") else 0
        if card_w > 0:
            content_w = min(int(content_w), max(120, card_w - 24))
        min_header_w = self._image_header_min_width()
        target_w = max(120, int(content_w), min_header_w)
        if card_w > 0:
            target_w = min(target_w, max(120, card_w - 24))
        for name in ("image_title_strip", "image_toolbar_card", "image_header_strip", "image_label", "image_actions_card", "display_strip", "scale_strip"):
            w = getattr(self, name, None)
            if not isinstance(w, QtWidgets.QWidget):
                continue
            if int(w.minimumWidth()) != target_w or int(w.maximumWidth()) != target_w:
                w.setFixedWidth(target_w)
        self._sync_image_header_balance()
        self._apply_image_header_side_padding()
        self._apply_options_side_padding()
        return int(target_w)

    def _image_header_min_width(self) -> int:
        left_w = int(self.image_index_label.sizeHint().width()) if hasattr(self, "image_index_label") else 48
        right_w = int(self.image_masks_label.sizeHint().width()) if hasattr(self, "image_masks_label") else 72
        prev_w = int(self.btn_prev_image.sizeHint().width()) if hasattr(self, "btn_prev_image") else 34
        next_w = int(self.btn_next_image.sizeHint().width()) if hasattr(self, "btn_next_image") else 34
        # Keep a compact but readable center name width.
        center_name_min = 180
        nav_w = prev_w + next_w + center_name_min + 24
        return max(280, left_w + nav_w + right_w + 24)

    def _options_min_row_width(self) -> int:
        if not hasattr(self, "_display_option_rows") or not hasattr(self, "display_grid"):
            return 0
        spacing = max(0, int(self.display_grid.horizontalSpacing()))
        req = 0
        for row in self._display_option_rows:
            visible = [w for w in row if w.isVisible()]
            if not visible:
                continue
            width = sum(max(1, int(w.sizeHint().width())) for w in visible)
            width += spacing * max(0, len(visible) - 1)
            req = max(req, width)
        return req

    def _apply_options_side_padding(self) -> None:
        if not hasattr(self, "display_grid") or not hasattr(self, "display_strip"):
            return
        side_img = max(0, int(getattr(self, "_options_image_side_px", 0)))
        soft = max(0, int(getattr(self, "_options_soft_margin_px", 0)))
        side = max(side_img, soft)
        strip_w = int(self.display_strip.width())
        strip_layout = self.display_strip.layout()
        if isinstance(strip_layout, QtWidgets.QLayout):
            margins = strip_layout.contentsMargins()
            inner_w = max(0, strip_w - int(margins.left()) - int(margins.right()))
        else:
            inner_w = max(0, strip_w)
        req_w = self._options_min_row_width()
        max_side = max(0, (inner_w - req_w) // 2)
        side = min(side, max_side)
        current = self.display_grid.contentsMargins()
        if int(current.left()) == side and int(current.right()) == side:
            return
        self.display_grid.setContentsMargins(side, 0, side, 0)

    def _apply_image_header_side_padding(self) -> None:
        if not hasattr(self, "image_header_strip"):
            return
        layout = self.image_header_strip.layout()
        if not isinstance(layout, QtWidgets.QLayout):
            return
        side = max(0, int(getattr(self, "_header_image_side_px", 0)))
        strip_w = int(self.image_header_strip.width())
        req_left = int(self.image_index_label.sizeHint().width()) if hasattr(self, "image_index_label") else 48
        req_right = int(self.image_masks_label.sizeHint().width()) if hasattr(self, "image_masks_label") else 72
        req_center = int(self.btn_prev_image.sizeHint().width()) + int(self.btn_next_image.sizeHint().width()) + int(
            self.image_name_label.width()
        ) + 24
        req_total = req_left + req_center + req_right + 24
        max_side = max(0, (strip_w - req_total) // 2)
        side = min(side, max_side)
        current = layout.contentsMargins()
        if int(current.left()) == side and int(current.right()) == side:
            return
        layout.setContentsMargins(side, 0, side, 0)

    def set_save_state(self, text: str, dirty: bool) -> None:
        self._save_state_full_text = text
        self._update_save_state_label_text()
        color = "#b91c1c" if dirty else "#166534"
        weight = "600" if dirty else "500"
        self.save_state_label.setStyleSheet(f"QLabel {{ color: {color}; font-weight: {weight}; }}")

    def _update_save_state_label_text(self) -> None:
        if not hasattr(self, "save_state_label"):
            return
        text = getattr(self, "_save_state_full_text", "") or ""
        fm = self.save_state_label.fontMetrics()
        avail = max(40, self.save_state_label.width() - 4)
        shown = fm.elidedText(text, QtCore.Qt.ElideRight, avail)
        self.save_state_label.setText(shown)
        self.save_state_label.setToolTip(text if shown != text else "")

    def set_calc_running(self, running: bool, text: str = "") -> None:
        self._calc_running = bool(running)
        if running:
            self.calc_progress.setVisible(True)
            self.calc_progress.setRange(0, 0)
        else:
            self.calc_progress.setVisible(False)
            self.calc_progress.setRange(0, 1)
            self.calc_progress.setValue(0)
        self._update_calc_state_label(text if text else None)

    def set_calc_pending(self, pending: bool) -> None:
        self._calc_pending = bool(pending)
        self._update_calc_state_label()

    def _update_calc_state_label(self, running_text: Optional[str] = None) -> None:
        if not hasattr(self, "calc_status_label"):
            return
        if self._calc_running:
            text = running_text if running_text else "Calculating..."
            self.calc_status_label.setText(text)
            self.calc_status_label.setStyleSheet("QLabel { color: #1d4ed8; font-weight: 600; }")
            return
        if self._calc_pending:
            self.calc_status_label.setText("Calc pending")
            self.calc_status_label.setStyleSheet("QLabel { color: #b91c1c; font-weight: 600; }")
            return
        self.calc_status_label.setText("Calculated")
        self.calc_status_label.setStyleSheet("QLabel { color: #166534; font-weight: 500; }")

    def set_review_rows(self, rows: Sequence[Sequence[str]], summary_text: str = "") -> None:
        self.review_table.setRowCount(len(rows))
        self.review_table.clearContents()
        for ri, row in enumerate(rows):
            for ci, val in enumerate(row[: self.review_table.columnCount()]):
                item = QtWidgets.QTableWidgetItem(str(val))
                item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                item.setToolTip(str(val))
                self.review_table.setItem(ri, ci, item)
        self.review_summary_label.setText(summary_text or f"Masks: {len(rows)}")
        if hasattr(self, "image_masks_label"):
            self.image_masks_label.setText(f"Masks={len(rows)}")
            self._sync_image_header_balance()
        self._apply_review_table_height()
        self._adjust_review_table_columns()

    def _apply_review_table_height(self) -> None:
        if not hasattr(self, "review_table"):
            return
        visible_rows = max(1, int(getattr(self, "_review_visible_rows", 10)))
        row_h = max(1, int(self.review_table.verticalHeader().defaultSectionSize()))
        header_h = max(0, int(self.review_table.horizontalHeader().height()))
        frame = int(self.review_table.frameWidth()) * 2
        total = header_h + row_h * visible_rows + frame + int(self._table_margin_px)
        self.review_table.setMinimumHeight(total)
        self.review_table.setMaximumHeight(total)

    def _adjust_review_table_columns(self) -> None:
        if not hasattr(self, "review_table"):
            return
        header = self.review_table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        header.setDefaultAlignment(QtCore.Qt.AlignCenter)
        width_by_label = {
            "ECD": 90,
            "VESD": 90,
            "Area(px)": 100,
            "Area(BBox)": 110,
            "Area(VESD)": 110,
            "Aspect(Feret)": 98,
            "Aspect(Ellipse)": 102,
            "Feret Maj": 96,
            "Feret Min": 96,
            "Ellipse Maj": 100,
            "Ellipse Min": 100,
            "Cent X": 88,
            "Cent Y": 88,
            "BBox X": 84,
            "BBox Y": 84,
            "BBox Cx": 92,
            "BBox Cy": 92,
            "Score": 80,
        }
        for ci in range(self.review_table.columnCount()):
            item = self.review_table.horizontalHeaderItem(ci)
            if item is None:
                continue
            label = item.text().strip()
            width = width_by_label.get(label, 88)
            self.review_table.setColumnWidth(ci, int(width))
        self.review_table.verticalHeader().setDefaultAlignment(QtCore.Qt.AlignCenter)

    def set_review_selected_rows(self, rows: Sequence[int]) -> None:
        self.review_table.clearSelection()
        for row in rows:
            idx = int(row)
            if 0 <= idx < self.review_table.rowCount():
                self.review_table.selectRow(idx)

    def set_review_selected_row(self, row: Optional[int]) -> None:
        if row is None:
            self.set_review_selected_rows([])
            return
        self.set_review_selected_rows([int(row)])

    def set_workspace_tab(self, index: int) -> None:
        self._set_workspace_tab_ui(index)

    def set_filter_input_source(self, source: str) -> None:
        token = str(source or "").strip().lower()
        target = "original" if token == "original" else "filtered"
        idx = 1 if target == "original" else 0
        with QtCore.QSignalBlocker(self.filter_input_source_combo):
            self.filter_input_source_combo.setCurrentIndex(idx)

    def set_filter_adjustments(self, brightness: int, contrast: float, gamma: float) -> None:
        b = int(max(-100, min(100, int(round(float(brightness))))))
        c = float(max(0.1, min(3.0, float(contrast))))
        g = float(max(0.2, min(3.0, float(gamma))))
        b_ui = int(round((float(b) + 100.0) * 0.5))
        if c >= 1.0:
            c_ui = int(round(50.0 + (c - 1.0) * (50.0 / 2.0)))
        else:
            c_ui = int(round((c - 0.1) * (50.0 / 0.9)))
        if g >= 1.0:
            g_ui = int(round(50.0 + (g - 1.0) * (50.0 / 2.0)))
        else:
            g_ui = int(round((g - 0.2) * (50.0 / 0.8)))
        b_ui = int(max(0, min(100, b_ui)))
        c_ui = int(max(0, min(100, c_ui)))
        g_ui = int(max(0, min(100, g_ui)))
        with QtCore.QSignalBlocker(self.filter_brightness_slider):
            self.filter_brightness_slider.setValue(b_ui)
        with QtCore.QSignalBlocker(self.filter_brightness_spin):
            self.filter_brightness_spin.setValue(b_ui)
        with QtCore.QSignalBlocker(self.filter_contrast_slider):
            self.filter_contrast_slider.setValue(c_ui)
        with QtCore.QSignalBlocker(self.filter_contrast_spin):
            self.filter_contrast_spin.setValue(c_ui)
        with QtCore.QSignalBlocker(self.filter_gamma_slider):
            self.filter_gamma_slider.setValue(g_ui)
        with QtCore.QSignalBlocker(self.filter_gamma_spin):
            self.filter_gamma_spin.setValue(g_ui)

    def set_filter_fft_mode(self, enabled: bool) -> None:
        mode = bool(enabled)
        self.filter_fft_btn.setEnabled(not mode)
        self.filter_ifft_btn.setEnabled(mode)
        self.filter_fft_mode_label.setText("Frequency view" if mode else "Spatial view")

    def get_filter_input_source(self) -> str:
        data = self.filter_input_source_combo.currentData()
        return str(data) if data else "filtered"

    def _on_filter_rows_moved(self, domain: str, source_start: int, source_end: int, destination_row: int) -> None:
        self.controller.on_filter_rows_moved(str(domain), int(source_start), int(source_end), int(destination_row))

    def set_spatial_filter_chain_rows(self, rows: Sequence[str], selected_row: int = -1) -> None:
        with QtCore.QSignalBlocker(self.spatial_filter_chain_list):
            self.spatial_filter_chain_list.clear()
            for text in rows:
                self.spatial_filter_chain_list.addItem(str(text))
            if rows:
                idx = int(max(0, min(int(selected_row), len(rows) - 1)))
                self.spatial_filter_chain_list.setCurrentRow(idx)
            else:
                self.spatial_filter_chain_list.setCurrentRow(-1)
        self.filter_remove_spatial_btn.setEnabled(bool(rows))

    def set_frequency_filter_chain_rows(self, rows: Sequence[str], selected_row: int = -1) -> None:
        with QtCore.QSignalBlocker(self.frequency_filter_chain_list):
            self.frequency_filter_chain_list.clear()
            for text in rows:
                self.frequency_filter_chain_list.addItem(str(text))
            if rows:
                idx = int(max(0, min(int(selected_row), len(rows) - 1)))
                self.frequency_filter_chain_list.setCurrentRow(idx)
            else:
                self.frequency_filter_chain_list.setCurrentRow(-1)
        self.filter_remove_frequency_btn.setEnabled(bool(rows))

    def set_filter_editor_from_step(self, step: Dict[str, Any]) -> None:
        if not isinstance(step, dict):
            self.clear_filter_editor()
            return
        kind = str(step.get("kind", "gaussian")).strip().lower()
        params = step.get("params", {}) if isinstance(step.get("params"), dict) else {}
        title_map = {
            "gaussian": "Gaussian",
            "median": "Median",
            "clahe": "CLAHE",
            "unsharp": "Unsharp",
            "lowpass": "Low-pass",
            "highpass": "High-pass",
            "bandpass": "Band-pass",
            "sym_notch": "Sym Notch",
        }
        idx_map = getattr(self, "_filter_editor_index_map", {})
        if kind not in idx_map:
            kind = "gaussian"
        self.filter_kind_label.setText(title_map.get(kind, kind.title()))
        self.filter_editor_stack.setCurrentIndex(int(idx_map.get(kind, 0)))
        if kind == "gaussian":
            sigma = float(params.get("sigma", 1.2))
            with QtCore.QSignalBlocker(self.filter_gaussian_sigma_slider):
                self.filter_gaussian_sigma_slider.setValue(int(round(sigma * 10.0)))
            with QtCore.QSignalBlocker(self.filter_gaussian_sigma_spin):
                self.filter_gaussian_sigma_spin.setValue(float(sigma))
        elif kind == "median":
            k = int(params.get("ksize", 3))
            if k % 2 == 0:
                k += 1 if k < 31 else -1
            with QtCore.QSignalBlocker(self.filter_median_ksize_slider):
                self.filter_median_ksize_slider.setValue(int(k))
            with QtCore.QSignalBlocker(self.filter_median_ksize_spin):
                self.filter_median_ksize_spin.setValue(int(k))
        elif kind == "clahe":
            clip = float(params.get("clip", 2.0))
            grid = int(params.get("grid", 8))
            with QtCore.QSignalBlocker(self.filter_clahe_clip_slider):
                self.filter_clahe_clip_slider.setValue(int(round(clip * 10.0)))
            with QtCore.QSignalBlocker(self.filter_clahe_clip_spin):
                self.filter_clahe_clip_spin.setValue(float(clip))
            with QtCore.QSignalBlocker(self.filter_clahe_grid_slider):
                self.filter_clahe_grid_slider.setValue(int(grid))
            with QtCore.QSignalBlocker(self.filter_clahe_grid_spin):
                self.filter_clahe_grid_spin.setValue(int(grid))
        elif kind == "unsharp":
            amount = float(params.get("amount", 1.0))
            sigma = float(params.get("sigma", 1.0))
            with QtCore.QSignalBlocker(self.filter_unsharp_amount_slider):
                self.filter_unsharp_amount_slider.setValue(int(round(amount * 10.0)))
            with QtCore.QSignalBlocker(self.filter_unsharp_amount_spin):
                self.filter_unsharp_amount_spin.setValue(float(amount))
            with QtCore.QSignalBlocker(self.filter_unsharp_sigma_slider):
                self.filter_unsharp_sigma_slider.setValue(int(round(sigma * 10.0)))
            with QtCore.QSignalBlocker(self.filter_unsharp_sigma_spin):
                self.filter_unsharp_sigma_spin.setValue(float(sigma))
        elif kind == "lowpass":
            cutoff = float(params.get("cutoff", 0.2))
            val = int(round(max(0.01, min(cutoff, 1.0)) * 100.0))
            with QtCore.QSignalBlocker(self.filter_lowpass_cutoff_slider):
                self.filter_lowpass_cutoff_slider.setValue(val)
            with QtCore.QSignalBlocker(self.filter_lowpass_cutoff_spin):
                self.filter_lowpass_cutoff_spin.setValue(val)
        elif kind == "highpass":
            cutoff = float(params.get("cutoff", 0.1))
            val = int(round(max(0.01, min(cutoff, 1.0)) * 100.0))
            with QtCore.QSignalBlocker(self.filter_highpass_cutoff_slider):
                self.filter_highpass_cutoff_slider.setValue(val)
            with QtCore.QSignalBlocker(self.filter_highpass_cutoff_spin):
                self.filter_highpass_cutoff_spin.setValue(val)
        elif kind == "bandpass":
            inner = float(params.get("inner", 0.08))
            outer = float(params.get("outer", 0.28))
            inner_v = int(round(max(0.0, min(inner, 0.98)) * 100.0))
            outer_v = int(round(max(inner + 0.01, min(outer, 1.0)) * 100.0))
            with QtCore.QSignalBlocker(self.filter_bandpass_inner_slider):
                self.filter_bandpass_inner_slider.setValue(inner_v)
            with QtCore.QSignalBlocker(self.filter_bandpass_inner_spin):
                self.filter_bandpass_inner_spin.setValue(inner_v)
            with QtCore.QSignalBlocker(self.filter_bandpass_outer_slider):
                self.filter_bandpass_outer_slider.setValue(outer_v)
            with QtCore.QSignalBlocker(self.filter_bandpass_outer_spin):
                self.filter_bandpass_outer_spin.setValue(outer_v)
        else:
            radius = float(params.get("radius", 0.35))
            width = float(params.get("width", 0.06))
            angle = float(params.get("angle_deg", 0.0))
            r_val = int(round(max(0.01, min(radius, 1.0)) * 100.0))
            w_val = int(round(max(0.005, min(width, 0.5)) * 100.0))
            with QtCore.QSignalBlocker(self.filter_notch_radius_slider):
                self.filter_notch_radius_slider.setValue(r_val)
            with QtCore.QSignalBlocker(self.filter_notch_radius_spin):
                self.filter_notch_radius_spin.setValue(r_val)
            with QtCore.QSignalBlocker(self.filter_notch_width_slider):
                self.filter_notch_width_slider.setValue(w_val)
            with QtCore.QSignalBlocker(self.filter_notch_width_spin):
                self.filter_notch_width_spin.setValue(w_val)
            with QtCore.QSignalBlocker(self.filter_notch_angle_slider):
                self.filter_notch_angle_slider.setValue(int(round(angle)))
            with QtCore.QSignalBlocker(self.filter_notch_angle_spin):
                self.filter_notch_angle_spin.setValue(int(round(angle)))

    def clear_filter_editor(self) -> None:
        self.filter_kind_label.setText("-")

    def get_review_sort_key(self) -> str:
        data = self.review_sort_key_combo.currentData()
        return str(data) if data else "index"

    def get_review_sort_desc(self) -> bool:
        data = self.review_sort_order_combo.currentData()
        return str(data) == "desc"

    def is_analyze_tab_active(self) -> bool:
        return True

    def is_evaluate_tab_active(self) -> bool:
        return getattr(self, "_infer_active_panel", "analyze") == "evaluate"

    def set_train_running(self, running: bool) -> None:
        self.train_run_btn.setEnabled(not running)
        self.train_stop_btn.setEnabled(running)

    def set_train_status_text(self, text: str, level: str = "idle") -> None:
        label = getattr(self, "train_status_label", None)
        if label is None:
            return
        lvl = (level or "idle").strip().lower()
        if lvl == "running":
            style = "QLabel { color: #1d4ed8; font-weight: 600; }"
        elif lvl in {"done", "ok", "success"}:
            style = "QLabel { color: #166534; font-weight: 600; }"
        elif lvl in {"error", "failed"}:
            style = "QLabel { color: #b91c1c; font-weight: 600; }"
        else:
            style = "QLabel { color: #5b6777; font-weight: 500; }"
        label.setStyleSheet(style)
        label.setText(text or "Idle")

    def set_train_progress(self, current: int, total: int) -> None:
        if not hasattr(self, "train_progress_bar"):
            return
        cur = max(0, int(current))
        tot = max(0, int(total))
        bar = self.train_progress_bar
        if tot <= 0:
            bar.setRange(0, 1)
            bar.setValue(0)
            bar.setFormat("0/0")
            return
        bar.setRange(0, tot)
        bar.setValue(min(cur, tot))
        bar.setFormat(f"{min(cur, tot)}/{tot}")

    def set_train_run_dir_text(self, text: str) -> None:
        if hasattr(self, "train_run_dir_label"):
            self.train_run_dir_label.setText(f"Run dir: {text.strip() if text else '-'}")

    def set_train_metrics_summary_text(self, text: str) -> None:
        if hasattr(self, "train_metrics_summary_label"):
            self.train_metrics_summary_label.setText(text or "train: -, val: -, test: -")

    def set_train_graph_pixmap(
        self,
        pix: Optional[QtGui.QPixmap],
        placeholder: str = "Run Train to view learning curves",
    ) -> None:
        self._train_graph_source_pixmap = pix.copy() if isinstance(pix, QtGui.QPixmap) else None
        self._refresh_train_graph_pixmap(placeholder=placeholder)

    def _refresh_train_graph_pixmap(self, placeholder: str = "Run Train to view learning curves") -> None:
        if not hasattr(self, "train_graph_label"):
            return
        label = self.train_graph_label
        pix = self._train_graph_source_pixmap
        if pix and not pix.isNull():
            shown = pix
            if label.width() > 0 and label.height() > 0:
                shown = pix.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            label.setPixmap(shown)
            label.setText("")
            return
        label.setPixmap(QtGui.QPixmap())
        label.setText(placeholder)

    def append_train_log(self, text: str) -> None:
        if not text:
            return
        self.train_log_edit.appendPlainText(text.rstrip("\n"))
        self.train_log_edit.verticalScrollBar().setValue(self.train_log_edit.verticalScrollBar().maximum())

    def clear_train_log(self) -> None:
        self.train_log_edit.clear()

    def set_train_command_text(self, text: str) -> None:
        self.train_cmd_label.setText(text)

    def set_eval_running(self, running: bool) -> None:
        if hasattr(self, "eval_run_current_btn"):
            self.eval_run_current_btn.setEnabled(not running)
        if hasattr(self, "eval_run_all_btn"):
            self.eval_run_all_btn.setEnabled(not running)

    def set_eval_scope(self, scope: str) -> None:
        key = "all" if (scope or "").strip().lower() == "all" else "current"
        self._eval_plot_scope = key
        self._refresh_eval_scope_plot()

    def set_eval_scope_plot(self, scope: str, pix: Optional[QtGui.QPixmap], placeholder: str) -> None:
        key = "all" if (scope or "").strip().lower() == "all" else "current"
        self._eval_plot_pixmaps[key] = pix.copy() if isinstance(pix, QtGui.QPixmap) else None
        self._eval_plot_placeholders[key] = placeholder or self._eval_plot_placeholders.get(key, "")
        if key == self._eval_plot_scope:
            self._refresh_eval_scope_plot()

    def set_eval_status_text(self, text: str) -> None:
        if hasattr(self, "eval_status_label"):
            self.eval_status_label.setText(text)

    def _refresh_eval_scope_plot(self) -> None:
        if not hasattr(self, "eval_plot_label"):
            return
        key = "all" if self._eval_plot_scope == "all" else "current"
        label = self.eval_plot_label
        pix = self._eval_plot_pixmaps.get(key)
        if pix and not pix.isNull():
            shown = pix
            if label.isVisible() and label.width() > 0 and label.height() > 0:
                target_w = label.width()
                target_h = label.height()
                shown = pix.scaled(target_w, target_h, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            label.setPixmap(shown)
            label.setText("")
            return
        label.setPixmap(QtGui.QPixmap())
        label.setText(self._eval_plot_placeholders.get(key, ""))

    def is_preview_tab_active(self) -> bool:
        return True

    def set_preview_analyze_tab(self, tab: str) -> None:
        t = (tab or "").strip().lower()
        if t == "evaluate":
            self._focus_infer_section("evaluate")
        elif t in {"preview", "table"}:
            self._focus_infer_section("preview")
        else:
            self._focus_infer_section("analyze")

    def set_action_mode(self, mode: str) -> None:
        m = (mode or "sam").lower()
        if m == "polygon":
            self.annotate_help_label.setText("Polygon: left=vertex, right=close. Set commits polygon as one instance.")
        elif m == "lora":
            self.annotate_help_label.setText(
                "LoRA: left=positive, right=negative, drag=box; right-click on mask toggles multi-select. Set commits current prediction."
            )
        else:
            self.annotate_help_label.setText(
                "SAM: left=positive, right=negative, drag=box; right-click on mask toggles multi-select. Set commits current prediction."
            )

    def set_image_status_text(self, text: str) -> None:
        left, sep, right = text.partition(":")
        index_text = left.strip() if sep else text.strip()
        if index_text.lower().startswith("image "):
            index_text = index_text[6:].strip()
        if not index_text:
            index_text = "-"
        if sep:
            self.image_index_label.setText(index_text)
            self._image_name_full = right.strip() if right.strip() else "-"
        else:
            self.image_index_label.setText(index_text)
            self._image_name_full = "-"
        self._sync_image_header_balance()

    def set_image_nav_enabled(self, prev_enabled: bool, next_enabled: bool) -> None:
        self.btn_prev_image.setEnabled(prev_enabled)
        self.btn_next_image.setEnabled(next_enabled)

    def set_graphs_pixmap(self, pix: Optional[QtGui.QPixmap], placeholder: str = "Graphs will appear after Calc") -> None:
        self.set_graph_panel_pixmaps(
            main=pix,
            top=pix,
            size=None,
            area=None,
            fractal=None,
            aspect=None,
            placeholder=placeholder,
        )

    def _set_graph_panel_label(
        self,
        label: QtWidgets.QLabel,
        pix: Optional[QtGui.QPixmap],
        placeholder: str = "",
    ) -> None:
        if pix is not None and not pix.isNull():
            shown = pix
            if label.width() > 0 and label.height() > 0:
                shown = pix.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            label.setPixmap(shown)
            label.setText("")
            return
        label.setPixmap(QtGui.QPixmap())
        label.setText(placeholder)

    def set_graph_panel_pixmaps(
        self,
        *,
        main: Optional[QtGui.QPixmap],
        top: Optional[QtGui.QPixmap],
        size: Optional[QtGui.QPixmap],
        area: Optional[QtGui.QPixmap],
        fractal: Optional[QtGui.QPixmap],
        aspect: Optional[QtGui.QPixmap],
        placeholder: str = "Graphs will appear after Calc",
    ) -> None:
        self._set_graph_panel_label(self.graphs_main_label, main, placeholder if main is None else "")
        self._set_graph_panel_label(self.graphs_top_label, top, placeholder if top is None else "")
        self._set_graph_panel_label(self.graphs_size_label, size, "No data")
        self._set_graph_panel_label(self.graphs_area_label, area, "No data")
        self._set_graph_panel_label(self.graphs_fractal_label, fractal, "No data")
        self._set_graph_panel_label(self.graphs_aspect_label, aspect, "No data")
        if hasattr(self, "graphs_reserved_label"):
            self.graphs_reserved_label.setPixmap(QtGui.QPixmap())
            self.graphs_reserved_label.setText("Reserved")

    def set_stats_table(self, headers: Sequence[str], rows: Sequence[str], values: Sequence[Sequence[str]]) -> None:
        self.stats_table.clearContents()
        self.stats_table.setColumnCount(len(headers))
        self.stats_table.setRowCount(len(rows))
        self.stats_table.setHorizontalHeaderLabels(headers)
        self.stats_table.setVerticalHeaderLabels(rows)
        for ri, row_vals in enumerate(values):
            for ci, text in enumerate(row_vals):
                item = QtWidgets.QTableWidgetItem(text)
                item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                if text == "-":
                    item.setForeground(QtGui.QColor("#94a3b8"))
                self.stats_table.setItem(ri, ci, item)
        self._fit_table_height()
        self._adjust_stats_table_columns()
        self._apply_stats_corner_text()

    def set_stats_summary_text(self, text: str) -> None:
        n_text = "n=-"
        raw = str(text or "").strip()
        idx = raw.lower().rfind("n=")
        if idx >= 0:
            n_text = raw[idx:].strip()
        self._stats_corner_text = n_text
        self._stats_corner_tooltip = raw
        self._apply_stats_corner_text()
        if hasattr(self, "stats_summary_label"):
            self.stats_summary_label.setText(raw)
            self.stats_summary_label.setVisible(False)

    def _apply_stats_corner_text(self) -> None:
        if not hasattr(self, "stats_table"):
            return
        btn = self.stats_table.findChild(QtWidgets.QAbstractButton, "qt_table_cornerbutton")
        if btn is None:
            for cand in self.stats_table.findChildren(QtWidgets.QAbstractButton):
                name = (cand.objectName() or "").strip().lower()
                if name == "qt_table_cornerbutton":
                    btn = cand
                    break
        if btn is None:
            for cand in self.stats_table.findChildren(QtWidgets.QAbstractButton):
                g = cand.geometry()
                if g.x() <= 2 and g.y() <= 2:
                    btn = cand
                    break
        if btn is None:
            QtCore.QTimer.singleShot(0, self._apply_stats_corner_text)
            return
        btn.setText(str(getattr(self, "_stats_corner_text", "n=-")))
        btn.setToolTip(str(getattr(self, "_stats_corner_tooltip", "")))
        btn.setCursor(QtCore.Qt.ArrowCursor)
        btn.setFocusPolicy(QtCore.Qt.NoFocus)
        btn.setStyleSheet(
            "QAbstractButton {"
            " background: #eef3fb;"
            " color: #1f344f;"
            " border: none;"
            " border-right: 1px solid #d8e2f1;"
            " border-bottom: 1px solid #d8e2f1;"
            " font-weight: 600;"
            " }"
        )
        btn.update()

    def _fit_table_height(self) -> None:
        row_h = max(1, int(self.stats_table.verticalHeader().defaultSectionSize()))
        header_h = max(0, int(self.stats_table.horizontalHeader().height()))
        frame = int(self.stats_table.frameWidth()) * 2
        rows = max(0, int(self.stats_table.rowCount()))
        # Keep Analysis table height exactly aligned to visible rows (no extra blank row under "max").
        total = header_h + row_h * rows + frame + 1
        self.stats_table.setMinimumHeight(total)
        self.stats_table.setMaximumHeight(total)

    def _adjust_stats_table_columns(self) -> None:
        if not hasattr(self, "stats_table"):
            return
        header = self.stats_table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        header.setDefaultAlignment(QtCore.Qt.AlignCenter)
        width_by_label = {
            "N1": 78,
            "N2": 78,
            "Cent1": 84,
            "Cent2": 84,
            "ECD": 82,
            "VESD": 82,
            "Area(px)": 96,
            "BBox Area": 98,
            "Area(VESD)": 104,
            "Feret Maj": 94,
            "Feret Min": 94,
            "Area(FeretRect)": 122,
            "Ellipse Maj": 98,
            "Ellipse Min": 98,
            "Aspect": 86,
            "Shape": 82,
        }
        for ci in range(self.stats_table.columnCount()):
            item = self.stats_table.horizontalHeaderItem(ci)
            if item is None:
                continue
            label = item.text().strip()
            self.stats_table.setColumnWidth(ci, int(width_by_label.get(label, 90)))

    # Resize hook --------------------------------------------------------------
    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:  # type: ignore[override]
        if event.type() == QtCore.QEvent.Type.MouseButtonPress:
            focused = QtWidgets.QApplication.focusWidget()
            if isinstance(focused, QtWidgets.QLineEdit):
                target: Optional[QtWidgets.QWidget] = None
                if isinstance(event, QtGui.QMouseEvent):
                    try:
                        target = QtWidgets.QApplication.widgetAt(event.globalPosition().toPoint())
                    except Exception:
                        target = None
                if not isinstance(target, QtWidgets.QLineEdit):
                    focused.clearFocus()
        return super().eventFilter(watched, event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._sync_top_cards_with_body_split()
        self._sync_image_header_balance()
        self._apply_image_header_side_padding()
        self._apply_options_side_padding()
        self._update_save_state_label_text()
        self._adjust_review_table_columns()
        self._adjust_stats_table_columns()
        self._apply_stats_corner_text()
        self._refresh_train_graph_pixmap()
        self._refresh_eval_scope_plot()
        self.controller.on_resize()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        if self._app_event_filter_installed:
            app = QtWidgets.QApplication.instance()
            if app is not None:
                app.removeEventFilter(self)
            self._app_event_filter_installed = False
        self.controller.on_app_close()
        super().closeEvent(event)
