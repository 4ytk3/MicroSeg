from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

try:
    from microseg.controller import launch_app
    from microseg.config import DEFAULT_SCALE
    from PySide6 import QtWidgets
except ModuleNotFoundError:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from microseg.controller import launch_app
    from microseg.config import DEFAULT_SCALE
    from PySide6 import QtWidgets

            
def main() -> None:
    parser = argparse.ArgumentParser(description="MicroSeg: Qt GUI for SAM mask segmentation and analysis")
    image_group = parser.add_mutually_exclusive_group(required=False)
    image_group.add_argument(
        "--image",
        type=Path,
        nargs="+",
        help="One or more base/original image paths",
    )
    image_group.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Directory containing base/original images",
    )
    parser.add_argument(
        "--image-exts",
        type=str,
        default="jpg,jpeg,png,tif,tiff,bmp",
        help="Comma-separated image extensions used with --image-dir",
    )
    mask_group = parser.add_mutually_exclusive_group()
    mask_group.add_argument(
        "--init-mask-id",
        type=Path,
        default=None,
        help="Optional instance-id mask image to preload (e.g., instance_ids.tiff)",
    )
    mask_group.add_argument(
        "--init-mask-dir",
        type=Path,
        default=None,
        help="Optional directory of per-instance binary masks to preload",
    )
    parser.add_argument("--hf-model-id", type=str, default="facebook/sam-vit-base")
    parser.add_argument("--lora-checkpoint", type=Path, default=None)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--scale-nm-per-px", type=float, default=DEFAULT_SCALE)
    parser.add_argument("--max-distance-nm", type=float, default=0.0, help="0 = unlimited")
    parser.add_argument("--fractal-slides", type=int, default=0, help="Sliding offsets per box size for fractal dimension (0=off, 20 recommended)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/microseg"), help="Base output directory (default: outputs/microseg)")
    parser.add_argument(
        "--qt-platform",
        type=str,
        choices=["auto", "wayland", "xcb"],
        default="auto",
        help="Qt platform backend (default: auto)",
    )
    args = parser.parse_args()

    def _configure_qt_platform(mode: str) -> None:
        selected = (mode or "auto").strip().lower()
        if selected in ("wayland", "xcb"):
            os.environ["QT_QPA_PLATFORM"] = selected
            return
        current = (os.environ.get("QT_QPA_PLATFORM") or "").strip().lower()
        if os.environ.get("WAYLAND_DISPLAY"):
            # Prefer xcb for stability on Wayland sessions unless explicitly forced.
            # This also overrides inherited WAYLAND defaults in shell profiles.
            if current in ("", "wayland", "wayland-egl"):
                os.environ["QT_QPA_PLATFORM"] = "xcb"
            return

    _configure_qt_platform(args.qt_platform)

    def _parse_exts(raw: str) -> set[str]:
        return {f".{e.strip().lower().lstrip('.')}" for e in str(raw).split(",") if e.strip()}

    def _collect_images_from_dir(image_dir: Path, exts: set[str]) -> list[Path]:
        if not image_dir.exists() or not image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        image_paths = sorted([p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
        if not image_paths:
            raise ValueError(f"No images found in {image_dir} for extensions: {sorted(exts)}")
        return image_paths

    def _dedup_keep_order(paths: Iterable[Path]) -> list[Path]:
        out: list[Path] = []
        seen: set[str] = set()
        for p in paths:
            key = str(p.resolve()) if p.exists() else str(p)
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
        return out

    def _prompt_startup_images(exts: set[str]) -> list[Path]:
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        while True:
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle("Open Images")
            msg.setText("Select image source to start MicroSeg.")
            file_btn = msg.addButton("Select Files", QtWidgets.QMessageBox.AcceptRole)
            dir_btn = msg.addButton("Select Folder", QtWidgets.QMessageBox.ActionRole)
            msg.addButton(QtWidgets.QMessageBox.Cancel)
            msg.setIcon(QtWidgets.QMessageBox.Question)
            msg.exec()
            clicked = msg.clickedButton()
            if clicked is file_btn:
                patterns = " ".join(f"*{ext}" for ext in sorted(exts))
                files, _ = QtWidgets.QFileDialog.getOpenFileNames(
                    None,
                    "Select image files",
                    str(Path.cwd()),
                    f"Images ({patterns});;All Files (*)",
                )
                if not files:
                    return []
                image_paths = _dedup_keep_order([Path(f) for f in files])
                missing = [p for p in image_paths if not p.exists() or not p.is_file()]
                if missing:
                    QtWidgets.QMessageBox.warning(None, "Invalid Selection", f"Could not read image: {missing[0]}")
                    continue
                return image_paths
            if clicked is dir_btn:
                picked = QtWidgets.QFileDialog.getExistingDirectory(
                    None,
                    "Select image directory",
                    str(Path.cwd()),
                )
                if not picked:
                    return []
                try:
                    return _collect_images_from_dir(Path(picked), exts)
                except Exception as exc:
                    QtWidgets.QMessageBox.warning(None, "Invalid Directory", str(exc))
                    continue
            return []

    exts = _parse_exts(args.image_exts)
    image_paths: list[Path]
    if args.image is not None:
        expanded: list[Path] = []
        for p in [Path(x) for x in args.image]:
            if p.is_dir():
                expanded.extend(_collect_images_from_dir(p, exts))
            else:
                expanded.append(p)
        image_paths = _dedup_keep_order(expanded)
        missing = [p for p in image_paths if not p.exists() or not p.is_file()]
        if missing:
            first = missing[0]
            raise FileNotFoundError(
                f"Could not read image: {first}\n"
                "Tip: pass files via --image, or pass a directory via --image-dir."
            )
    elif args.image_dir is not None:
        image_paths = _collect_images_from_dir(Path(args.image_dir), exts)
    else:
        image_paths = _prompt_startup_images(exts)
        if not image_paths:
            return

    max_dist = None if args.max_distance_nm == 0 else args.max_distance_nm
    launch_app(
        image_paths=image_paths,
        init_mask_id_path=args.init_mask_id,
        init_mask_dir=args.init_mask_dir,
        output_dir=args.output_dir,
        hf_model_id=args.hf_model_id,
        lora_checkpoint=args.lora_checkpoint,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        mask_threshold=args.mask_threshold,
        scale_nm_per_px=args.scale_nm_per_px,
        max_distance_nm=max_dist,
        fractal_slides=args.fractal_slides,
    )


if __name__ == "__main__":
    main()
