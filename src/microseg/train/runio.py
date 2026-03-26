from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    return value


def _slugify(value: str) -> str:
    safe = []
    for ch in value:
        if ch.isascii() and (ch.isalnum() or ch in ("-", "_")):
            safe.append(ch)
        else:
            safe.append("-")
    collapsed = "".join(safe)
    while "--" in collapsed:
        collapsed = collapsed.replace("--", "-")
    return collapsed.strip("-") or "run"


def create_run_dir(output_root: Path, run_tag: Optional[str] = None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = timestamp
    if run_tag:
        base_name = f"{timestamp}_{_slugify(run_tag)}"
    run_dir = output_root / base_name
    if run_dir.exists():
        suffix = 1
        while True:
            candidate = output_root / f"{base_name}_{suffix}"
            if not candidate.exists():
                run_dir = candidate
                break
            suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_run_metadata(
    run_dir: Path,
    *,
    args: argparse.Namespace,
    freeze_config: Dict[str, bool],
    freeze_label: str,
    device: str,
    log_dir: Path,
    run_tag: Optional[str] = None,
) -> None:
    payload: Dict[str, Any] = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "device": device,
        "log_dir": str(log_dir),
        "freeze_config": freeze_config,
        "freeze_label": freeze_label,
        "run_tag": run_tag,
        "args": _jsonify(vars(args)),
    }
    config_path = run_dir / "run_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    command_path = run_dir / "command.txt"
    command_path.write_text(payload["command"] + "\n", encoding="utf-8")


def write_split_metadata(
    run_dir: Path,
    image_files: List[str],
    mask_files: List[str],
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
) -> None:
    def build_pairs(indices: List[int]) -> List[Dict[str, str]]:
        return [{"image": image_files[i], "mask": mask_files[i]} for i in indices]

    payload: Dict[str, Any] = {
        "counts": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx),
        },
        "indices": {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        },
        "files": {
            "train": build_pairs(train_idx),
            "val": build_pairs(val_idx),
            "test": build_pairs(test_idx),
        },
    }
    split_path = run_dir / "splits.json"
    with split_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def append_metrics(run_dir: Path, phase: str, epoch: int, metrics: Any, *, extra: Optional[Dict[str, Any]] = None) -> None:
    payload = {
        "phase": phase,
        "epoch": epoch,
        "loss": metrics.loss,
        "iou": metrics.iou,
        "dice": metrics.dice,
        "samples": metrics.samples,
    }
    if extra:
        payload.update(extra)
    path = run_dir / "metrics.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def write_named_metrics(run_dir: Path, name: str, epoch: int, metrics: Any) -> None:
    payload = {
        "epoch": epoch,
        "loss": metrics.loss,
        "iou": metrics.iou,
        "dice": metrics.dice,
        "samples": metrics.samples,
    }
    path = run_dir / f"{name}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_cv_summary(run_dir: Path, payload: Dict[str, Any]) -> None:
    path = run_dir / "cv_summary.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
