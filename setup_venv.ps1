# setup_venv.ps1 — create venv and install requirements
# Usage: Open PowerShell in project root and run: .\setup_venv.ps1

python -m venv venv
& .\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
if (Test-Path requirements.txt) {
    pip install -r requirements.txt
} else {
    Write-Host "No requirements.txt found; activate venv and install packages manually."
}
