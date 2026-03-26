#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.12 .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  uv pip install -e ".[dist]"
else
  PY_BIN="${PYTHON_BIN:-python3.12}"
  "$PY_BIN" -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install -e ".[dist]"
fi

python scripts/build_microseg.py --clean --noconfirm "$@"
