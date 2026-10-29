$ErrorActionPreference = "Continue"

Set-Location $PSScriptRoot

Write-Host "=== Setup ==="
pip install -r requirements.txt

Write-Host "=== Data ==="
Push-Location data
python getdata.py
Pop-Location
python -c "from src import data_preparation; df=data_preparation.load_data(); data_preparation.validate_data(df); data_preparation.eda(df)"

Write-Host "=== Train ==="
python -c "from src import train; train.train()"

Write-Host "=== Evaluate ==="
python -c "from src import evaluate; evaluate.evaluate()"

Write-Host "=== Test ==="
pytest -v

Write-Host "=== Done ==="
