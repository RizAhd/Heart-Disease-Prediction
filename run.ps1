param(
    [switch]$SkipVenv
)

$ErrorActionPreference = "Stop"

if (-not $SkipVenv) {
    if (-not (Test-Path "heartdiseaseprediction")) {
        python -m venv heartdiseaseprediction
    }
    . .\heartdiseaseprediction\Scripts\Activate.ps1
}

python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.train_and_evaluate
