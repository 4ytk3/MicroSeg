from __future__ import annotations

import os


def _configure_qt_platform() -> None:
    current = (os.environ.get("QT_QPA_PLATFORM") or "").strip().lower()
    if current:
        return
    if os.environ.get("WAYLAND_DISPLAY"):
        # Prefer xcb by default on Wayland session to avoid surface-size protocol issues.
        os.environ["QT_QPA_PLATFORM"] = "xcb"


_configure_qt_platform()
