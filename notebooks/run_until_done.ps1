# Keeps (re)starting the baseline run until notebooks\result.txt exists.
# Each restart resumes from disk (index cache, downloaded payloads, scores).
# Launch:  powershell -ExecutionPolicy Bypass -File notebooks\run_until_done.ps1
$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repo

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "FATAL: .venv\Scripts\python.exe not found in $repo" -ForegroundColor Red
    exit 1
}
if (Test-Path "notebooks\result.txt") {
    Write-Host "notebooks\result.txt already exists - delete it first if you want a fresh run."
    exit 0
}

while ($true) {
    Write-Host "=== starting run: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" -ForegroundColor Cyan
    & ".venv\Scripts\python.exe" -u "notebooks\run_img_baseline.py"
    if ((Test-Path "notebooks\result.txt")) {
        Write-Host "=== FINISHED: notebooks\result.txt ===" -ForegroundColor Green
        break
    }
    Write-Host "run ended without result.txt (exit $LASTEXITCODE) - retrying in 2 minutes..." -ForegroundColor Yellow
    Start-Sleep -Seconds 120
}
