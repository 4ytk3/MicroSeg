#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build frozen MicroSeg distribution with PyInstaller.")
    p.add_argument("--clean", action="store_true", help="Clean PyInstaller cache/work directory before build")
    p.add_argument("--noconfirm", action="store_true", help="Replace existing dist/work outputs without prompt")
    p.add_argument("--distpath", type=Path, default=None, help="Output directory for built app")
    p.add_argument("--workpath", type=Path, default=None, help="Working directory for build intermediates")
    p.add_argument("--spec", type=Path, default=Path("packaging/microseg.spec"), help="PyInstaller spec file")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    spec_path = (repo_root / args.spec).resolve() if not args.spec.is_absolute() else args.spec.resolve()
    if not spec_path.exists():
        raise FileNotFoundError(f"Spec file not found: {spec_path}")

    cmd = [sys.executable, "-m", "PyInstaller", str(spec_path)]
    if args.clean:
        cmd.append("--clean")
    if args.noconfirm:
        cmd.append("--noconfirm")
    if args.distpath is not None:
        cmd.extend(["--distpath", str(args.distpath.resolve())])
    if args.workpath is not None:
        cmd.extend(["--workpath", str(args.workpath.resolve())])

    env = dict(os.environ)
    src_path = str((repo_root / "src").resolve())
    current_pythonpath = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = src_path if not current_pythonpath else f"{src_path}{os.pathsep}{current_pythonpath}"

    print("[INFO] Running:", " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.run(cmd, cwd=str(repo_root), env=env)
    if proc.returncode != 0:
        return proc.returncode

    dist_dir = args.distpath.resolve() if args.distpath is not None else (repo_root / "dist")
    app_dir = dist_dir / "microseg"
    print(f"[INFO] Build complete: {app_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
