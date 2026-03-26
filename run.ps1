param(
    [switch]$SkipVenv
)

$ErrorActionPreference = "Stop"

if (-not $SkipVenv) {
    if (-not (Test-Path ".venv")) {
        python -m venv .venv
    }
    . .\.venv\Scripts\Activate.ps1
}

python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.train_and_evaluate
