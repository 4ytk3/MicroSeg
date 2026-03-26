param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$BuildArgs
)

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir

if (Get-Command uv -ErrorAction SilentlyContinue) {
    uv venv --python 3.12 .venv
    . .\.venv\Scripts\Activate.ps1
    uv pip install -e ".[dist]"
}
else {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        py -3.12 -m venv .venv
    }
    else {
        python -m venv .venv
    }
    . .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -e ".[dist]"
}

python scripts/build_microseg.py --clean --noconfirm @BuildArgs
