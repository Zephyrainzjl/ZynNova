param(
    [string]$Python = "python",
    [string]$Extras = "zynmorph-all"
)

$ErrorActionPreference = "Stop"

Write-Host "ZynNova Windows native reinstall" -ForegroundColor Cyan
Write-Host "Close ALL Jupyter kernels / VS Code Python terminals that imported zynnova._native before continuing." -ForegroundColor Yellow

& $Python scripts/cleanup_zynnova_native_windows.py --remove
if ($LASTEXITCODE -ne 0) {
    throw "Native cleanup failed. Close the process locking the .pyd and run this script again."
}

if (Test-Path .\build) {
    Remove-Item -Recurse -Force .\build
}

& $Python -m pip install -e ".[${Extras}]" -v
if ($LASTEXITCODE -ne 0) {
    throw "pip editable install failed."
}

Write-Host "\nNative status:" -ForegroundColor Cyan
& $Python scripts/diagnose_tetgen_native.py
if ($LASTEXITCODE -ne 0) {
    throw "TetGen native diagnostic failed."
}

Write-Host "\nReinstall complete. Start a NEW Jupyter kernel before importing ZynNova." -ForegroundColor Green
