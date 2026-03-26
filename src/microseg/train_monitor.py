from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


TRAIN_RUN_DIR_RE = re.compile(r"^\[INFO\]\s+Run dir:\s*(.+?)\s*$")
TRAIN_EPOCH_RE = re.compile(r"^\[INFO\]\s+Epoch\s+(\d+)\s*/\s*(\d+)\s*-\s*loss\s+([^\s]+)")


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


@dataclass
class TrainMetricsPayload:
    train_epochs: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    train_iou: List[float] = field(default_factory=list)
    train_dice: List[float] = field(default_factory=list)
    val_epochs: List[int] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_iou: List[float] = field(default_factory=list)
    val_dice: List[float] = field(default_factory=list)
    test_epochs: List[int] = field(default_factory=list)
    test_loss: List[float] = field(default_factory=list)
    test_iou: List[float] = field(default_factory=list)
    test_dice: List[float] = field(default_factory=list)
    summary_text: str = "train: -, val: -, test: -"


@dataclass
class TrainMonitorState:
    expected_epochs: int = 0
    stdout_buffer: str = ""
    run_dir: Optional[Path] = None
    metrics_path: Optional[Path] = None
    metrics_mtime_ns: Optional[int] = None
    last_epoch: int = 0

    def reset(self, expected_epochs: int) -> None:
        self.expected_epochs = max(0, int(expected_epochs))
        self.stdout_buffer = ""
        self.run_dir = None
        self.metrics_path = None
        self.metrics_mtime_ns = None
        self.last_epoch = 0

    def _apply_line(self, line: str, project_root: Path) -> None:
        text = str(line or "").strip()
        if not text:
            return
        run_match = TRAIN_RUN_DIR_RE.match(text)
        if run_match:
            run_dir = Path(run_match.group(1).strip()).expanduser()
            if not run_dir.is_absolute():
                run_dir = (project_root / run_dir).resolve()
            self.run_dir = run_dir
            self.metrics_path = run_dir / "metrics.jsonl"
            self.metrics_mtime_ns = None

        epoch_match = TRAIN_EPOCH_RE.match(text)
        if epoch_match:
            epoch_now = _to_int(epoch_match.group(1))
            epoch_total = _to_int(epoch_match.group(2))
            if epoch_now is not None:
                self.last_epoch = max(0, int(epoch_now))
            if epoch_total is not None and epoch_total > 0:
                self.expected_epochs = int(epoch_total)

    def consume_output(self, text: str, project_root: Path) -> None:
        if not text:
            return
        self.stdout_buffer += text
        lines = self.stdout_buffer.splitlines(keepends=True)
        if lines and not lines[-1].endswith(("\n", "\r")):
            self.stdout_buffer = lines.pop()
        else:
            self.stdout_buffer = ""
        for raw in lines:
            self._apply_line(raw.rstrip("\r\n"), project_root)

    def flush_pending_output(self, project_root: Path) -> None:
        if not self.stdout_buffer:
            return
        self._apply_line(self.stdout_buffer, project_root)
        self.stdout_buffer = ""

    def _load_metrics_rows(self, force: bool = False) -> List[Dict[str, Any]]:
        path = self.metrics_path
        if path is None or not path.exists() or not path.is_file():
            return []
        try:
            mtime_ns = path.stat().st_mtime_ns
        except Exception:
            mtime_ns = None
        if not force and mtime_ns is not None and mtime_ns == self.metrics_mtime_ns:
            return []
        self.metrics_mtime_ns = mtime_ns

        rows: List[Dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line_s = line.strip()
                    if not line_s:
                        continue
                    try:
                        payload = json.loads(line_s)
                    except Exception:
                        continue
                    if isinstance(payload, dict):
                        rows.append(payload)
        except Exception:
            return []
        return rows

    @staticmethod
    def _series_from_phase(phase_data: Dict[int, Dict[str, float]], key: str) -> Tuple[List[int], List[float]]:
        epochs: List[int] = []
        values: List[float] = []
        for ep in sorted(phase_data.keys()):
            epochs.append(int(ep))
            val = phase_data[ep].get(key)
            values.append(float(val) if val is not None else float("nan"))
        return epochs, values

    @staticmethod
    def _latest_metric_text(phase: str, phase_data: Dict[int, Dict[str, float]]) -> str:
        if not phase_data:
            return f"{phase}: -"
        last_epoch = max(phase_data.keys())
        vals = phase_data[last_epoch]

        def _fmt(v: Optional[float]) -> str:
            if v is None:
                return "-"
            return f"{float(v):.4f}"

        return (
            f"{phase} e{int(last_epoch)} "
            f"loss={_fmt(vals.get('loss'))}, "
            f"iou={_fmt(vals.get('iou'))}, "
            f"dice={_fmt(vals.get('dice'))}"
        )

    @staticmethod
    def _extract_phase_map(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[int, Dict[str, float]]]:
        phase_map: Dict[str, Dict[int, Dict[str, float]]] = {
            "train": {},
            "val": {},
            "test": {},
        }
        for row in rows:
            phase = str(row.get("phase", "")).strip().lower()
            if phase not in phase_map:
                continue
            epoch = _to_int(row.get("epoch"))
            if epoch is None or epoch < 0:
                continue
            phase_map[phase][epoch] = {
                "loss": _to_float(row.get("loss")),
                "iou": _to_float(row.get("iou")),
                "dice": _to_float(row.get("dice")),
            }
        return phase_map

    def build_metrics_payload(self, force: bool = False) -> Optional[TrainMetricsPayload]:
        rows = self._load_metrics_rows(force=force)
        if not rows:
            return None
        phase_map = self._extract_phase_map(rows)
        train_phase = phase_map.get("train", {})
        val_phase = phase_map.get("val", {})
        test_phase = phase_map.get("test", {})

        train_epochs, train_loss = self._series_from_phase(train_phase, "loss")
        _, train_iou = self._series_from_phase(train_phase, "iou")
        _, train_dice = self._series_from_phase(train_phase, "dice")
        val_epochs, val_loss = self._series_from_phase(val_phase, "loss")
        _, val_iou = self._series_from_phase(val_phase, "iou")
        _, val_dice = self._series_from_phase(val_phase, "dice")
        test_epochs, test_loss = self._series_from_phase(test_phase, "loss")
        _, test_iou = self._series_from_phase(test_phase, "iou")
        _, test_dice = self._series_from_phase(test_phase, "dice")

        summary = " | ".join(
            [
                self._latest_metric_text("train", train_phase),
                self._latest_metric_text("val", val_phase),
                self._latest_metric_text("test", test_phase),
            ]
        )
        return TrainMetricsPayload(
            train_epochs=train_epochs,
            train_loss=train_loss,
            train_iou=train_iou,
            train_dice=train_dice,
            val_epochs=val_epochs,
            val_loss=val_loss,
            val_iou=val_iou,
            val_dice=val_dice,
            test_epochs=test_epochs,
            test_loss=test_loss,
            test_iou=test_iou,
            test_dice=test_dice,
            summary_text=summary,
        )
