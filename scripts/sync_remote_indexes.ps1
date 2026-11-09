# ==============================================================================
# Sync Remote Indexes Script for AIC-HCMC-26
# Transfers updated Batch 1 + Batch 2 indexes from 192.168.28.112 to local machine
# ==============================================================================

param (
    [string]$ServerIP = "192.168.28.112",
    [string]$User = "24521668",
    [string]$RemotePath = "~/AIC-HCMC-26/indexes"
)

$LocalIndexDir = Join-Path $PSScriptRoot "..\indexes"
if (-not (Test-Path $LocalIndexDir)) {
    New-Item -ItemType Directory -Path $LocalIndexDir -Force | Out-Null
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Fetching indexes from remote server ($User@$ServerIP)..." -ForegroundColor Cyan
Write-Host " Remote path: $RemotePath" -ForegroundColor Cyan
Write-Host " Local path:  $LocalIndexDir" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$scpCmd = "scp -r ${User}@${ServerIP}:${RemotePath}/* `"$LocalIndexDir`""
Write-Host "Running: $scpCmd" -ForegroundColor Yellow
Invoke-Expression $scpCmd

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nSuccessfully synchronized indexes!" -ForegroundColor Green
    Get-ChildItem -Path $LocalIndexDir | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
} else {
    Write-Host "`nSCP failed with exit code $LASTEXITCODE. Please check SSH credentials and network connectivity." -ForegroundColor Red
}
