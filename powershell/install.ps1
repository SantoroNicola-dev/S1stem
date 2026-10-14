# ==============================================================================
# S1stem Showcase Application Installer
# ==============================================================================
# Installs S1stem PowerShell showcase assistant into user's profile.
# ==============================================================================

Write-Host ""
Write-Host " +-------------------------------------------------------------------------+" -ForegroundColor Cyan
Write-Host " |  S1STEM SHOWCASE // POWERSHELL ASSISTANT APPLICATION INSTALLER          |" -ForegroundColor Cyan
Write-Host " +-------------------------------------------------------------------------+" -ForegroundColor Cyan
Write-Host ""

# 1. Verify Node.js
$nodeCmd = Get-Command "node" -ErrorAction SilentlyContinue
if (-not $nodeCmd) {
    Write-Error "Node.js is not installed or not present in system PATH. Install Node.js v18+ before continuing."
    exit 1
}

$nodeVer = & node -v
Write-Host " [ok] Detected Node.js: $nodeVer" -ForegroundColor Green

# 2. Locate S1stem Module
$modulePath = Join-Path $PSScriptRoot "S1stem.psm1"
if (-not (Test-Path $modulePath)) {
    Write-Error "S1stem.psm1 module not found in $modulePath."
    exit 1
}
Write-Host " [ok] S1stem module found: $modulePath" -ForegroundColor Green

# 3. Configure PowerShell Profiles (both PowerShell 7+ Core and Windows PowerShell 5.1)
$docsPath = [Environment]::GetFolderPath("MyDocuments")
$targetProfiles = @(
    (Join-Path $docsPath "PowerShell\Microsoft.PowerShell_profile.ps1"),
    (Join-Path $docsPath "WindowsPowerShell\Microsoft.PowerShell_profile.ps1")
)
if ($PROFILE -and -not ($targetProfiles -contains $PROFILE)) {
    $targetProfiles += $PROFILE
}

$importStatement = "Import-Module `"$modulePath`""
$entry = "`n# S1stem Neural Assistant Integration`n$importStatement`n"

foreach ($prof in $targetProfiles) {
    $profDir = Split-Path $prof -Parent
    if (-not (Test-Path $profDir)) {
        New-Item -ItemType Directory -Path $profDir -Force | Out-Null
        Write-Host " [ok] Created profile directory: $profDir" -ForegroundColor Green
    }
    if (-not (Test-Path $prof)) {
        New-Item -ItemType File -Path $prof -Force | Out-Null
        Write-Host " [ok] Created profile file: $prof" -ForegroundColor Green
    }
    $profContent = Get-Content $prof -Raw -ErrorAction SilentlyContinue
    if ($profContent -match [regex]::Escape($modulePath)) {
        Write-Host " [ok] S1stem is already registered in profile: $prof" -ForegroundColor DarkGreen
    } else {
        Add-Content -Path $prof -Value $entry
        Write-Host " [ok] Added Import-Module to profile: $prof" -ForegroundColor Green
    }
}

# 5. Import into current session
Import-Module $modulePath -Force
Write-Host " [ok] Module imported into current PowerShell session." -ForegroundColor Green

# 6. Start daemon service
Write-Host ""
Write-Host " Starting neural daemon in background..." -ForegroundColor Yellow
Start-S1stemDaemon

Write-Host ""
Write-Host " =============================================================================" -ForegroundColor Cyan
Write-Host " Installation completed successfully!" -ForegroundColor Green
Write-Host " S1stem is registered for both PowerShell 7 (pwsh) and Windows PowerShell." -ForegroundColor White
Write-Host ""
Write-Host " To activate 's1' IMMEDIATELY in this open window, type:" -ForegroundColor Yellow
Write-Host "   Import-Module .\powershell\S1stem.psm1" -ForegroundColor Cyan
Write-Host " (or: . `$PROFILE)" -ForegroundColor DarkGray
Write-Host ""
Write-Host " In all NEW PowerShell terminal windows it will be available automatically." -ForegroundColor White
Write-Host " Examples:" -ForegroundColor Yellow
Write-Host '   s1 what time is it' -ForegroundColor White
Write-Host '   s1 command to create a new file in D:\' -ForegroundColor White
Write-Host '   s1 command to list files in a directory in c:\test' -ForegroundColor White
Write-Host '   s1 create a new directory and enter inside, directory name new_dir' -ForegroundColor White
Write-Host " =============================================================================" -ForegroundColor Cyan
Write-Host ""
