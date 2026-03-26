# -*- mode: python ; coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules


PROJECT_ROOT = Path(SPECPATH).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
ENTRY = SRC_ROOT / "microseg" / "main.py"


def _dedupe(items):
    out = []
    seen = set()
    for item in items:
        key = tuple(item) if isinstance(item, (tuple, list)) else item
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _safe_collect_all(package_name: str):
    try:
        return collect_all(package_name)
    except Exception:
        return [], [], []


datas = []
binaries = []
hiddenimports = []

# Third-party runtime dependencies used by microseg.
for pkg in [
    "PySide6",
    "cv2",
    "numpy",
    "scipy",
    "skimage",
    "matplotlib",
    "PIL",
    "torch",
    "torchvision",
    "segment_anything",
    "transformers",
    "peft",
    "roifile",
]:
    d, b, h = _safe_collect_all(pkg)
    datas.extend(d)
    binaries.extend(b)
    hiddenimports.extend(h)

# Local modules: microseg now owns all runtime logic needed for GUI distribution.
for local_pkg in ["microseg"]:
    try:
        hiddenimports.extend(collect_submodules(local_pkg))
    except Exception:
        pass
    try:
        datas.extend(collect_data_files(local_pkg, include_py_files=True))
    except Exception:
        pass

# Include local config templates if present.
config_dir = PROJECT_ROOT / "config"
if config_dir.exists():
    datas.append((str(config_dir), "config"))

datas = _dedupe(datas)
binaries = _dedupe(binaries)
hiddenimports = _dedupe(hiddenimports)

runtime_hooks = [str(PROJECT_ROOT / "packaging" / "runtime_qt_platform.py")]

block_cipher = None

a = Analysis(
    [str(ENTRY)],
    pathex=[str(SRC_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=runtime_hooks,
    excludes=["tkinter", "pytest", "IPython"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="microseg",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="microseg",
)
