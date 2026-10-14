# ==============================================================================
# S1stem Showcase Application: PowerShell Terminal Assistant
# ==============================================================================
# Example implementation demonstrating S1stem as a System 1 fast neural decision
# layer for command-line environments and operating systems.
#
# Usage:
#   s1 command to create a new file in D:\
#   s1 command to list files in a directory in c:\test
#   s1 create a new directory and enter inside, directory name new_dir
# ==============================================================================

$script:S1stemPort = if ($env:S1STEM_PORT) { [int]$env:S1STEM_PORT } else { 9191 }
$script:S1stemHost = "127.0.0.1"
$script:DaemonScript = Join-Path $PSScriptRoot "s1stem-daemon.js"
$script:CliScript    = Join-Path $PSScriptRoot "s1stem-cli.js"

function Get-S1stemDaemonUrl {
    return "http://${script:S1stemHost}:${script:S1stemPort}"
}

function Test-S1stemDaemon {
    try {
        $url = "$(Get-S1stemDaemonUrl)/health"
        $resp = Invoke-RestMethod -Uri $url -Method Get -TimeoutSec 1 -ErrorAction Stop
        return ($resp.status -eq "ready")
    } catch {
        return $false
    }
}

function Start-S1stemDaemon {
    [CmdletBinding()]
    param()

    if (Test-S1stemDaemon) {
        Write-Host " [s1stem] Local daemon already active on $(Get-S1stemDaemonUrl)." -ForegroundColor DarkGreen
        return $true
    }

    Write-Host " [s1stem] Starting background neural daemon..." -ForegroundColor DarkYellow
    
    # Check node presence
    $nodeCmd = Get-Command "node" -ErrorAction SilentlyContinue
    if (-not $nodeCmd) {
        Write-Error "Node.js is not installed or not present in system PATH."
        return $false
    }

    # Launch daemon hidden
    $process = Start-Process -FilePath "node" `
        -ArgumentList "`"$script:DaemonScript`"" `
        -WorkingDirectory $PSScriptRoot `
        -WindowStyle Hidden `
        -PassThru

    # Wait for daemon to become ready (up to 5 seconds)
    $attempts = 0
    while ($attempts -lt 25) {
        Start-Sleep -Milliseconds 200
        if (Test-S1stemDaemon) {
            Write-Host " [s1stem] Daemon ready on $(Get-S1stemDaemonUrl) (PID: $($process.Id))." -ForegroundColor Green
            return $true
        }
        $attempts++
    }

    Write-Warning " [s1stem] Daemon did not respond in time. Falling back to in-process CLI."
    return $false
}

function Stop-S1stemDaemon {
    [CmdletBinding()]
    param()

    if (-not (Test-S1stemDaemon)) {
        Write-Host " [s1stem] No active daemon." -ForegroundColor DarkGray
        return
    }

    try {
        $url = "$(Get-S1stemDaemonUrl)/shutdown"
        $null = Invoke-RestMethod -Uri $url -Method Post -TimeoutSec 2 -ErrorAction SilentlyContinue
        Write-Host " [s1stem] Shutdown signal sent to daemon." -ForegroundColor DarkYellow
    } catch {
        Write-Warning "Unable to stop daemon via HTTP."
    }
}

function s1 {
    [CmdletBinding()]
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Prompt
    )

    $rawPrompt = ($Prompt -join ' ').Trim()

    # Show help banner if prompt is empty
    if ([string]::IsNullOrWhiteSpace($rawPrompt) -or $rawPrompt -eq "-h" -or $rawPrompt -eq "--help") {
        Show-S1stemHelp
        return
    }

    # Step 1: Ensure daemon is running or fall back
    $isDaemonReady = Test-S1stemDaemon
    if (-not $isDaemonReady) {
        $isDaemonReady = Start-S1stemDaemon
    }

    # Step 2: Query resolution
    $result = $null
    $tStart = [System.Diagnostics.Stopwatch]::StartNew()

    if ($isDaemonReady) {
        try {
            $body = @{ prompt = $rawPrompt } | ConvertTo-Json -Compress
            $url = "$(Get-S1stemDaemonUrl)/resolve"
            $result = Invoke-RestMethod -Uri $url -Method Post -Body $body -ContentType "application/json" -TimeoutSec 5 -ErrorAction Stop
        } catch {
            Write-Warning " [s1stem] Communication error with daemon: $_. Falling back to CLI."
        }
    }

    # CLI fallback if daemon failed
    if (-not $result) {
        try {
            $cliOutput = & node "$script:CliScript" "$rawPrompt" 2>$null
            if ($cliOutput) {
                $result = $cliOutput | ConvertFrom-Json
            }
        } catch {
            Write-Error " [s1stem] CLI fallback failed: $_"
            return
        }
    }

    $tStart.Stop()
    $roundTripMs = [math]::Round($tStart.Elapsed.TotalMilliseconds, 1)

    if (-not $result -or -not $result.command) {
        Write-Error " [s1stem] Could not resolve a command for: '$rawPrompt'"
        return
    }

    # Step 3: Render Interactive Terminal UI
    Show-S1stemResultCard -Result $result -RoundTripMs $roundTripMs

    # Step 4: Interactive Choice Prompt [E]xecute / [C]opy / [M]odify / [A]bort
    Prompt-S1stemAction -Command $result.command -IsFlagged $result.security.isFlagged
}

function Show-S1stemResultCard {
    param(
        [PSCustomObject]$Result,
        [double]$RoundTripMs
    )

    $cmd        = $Result.command
    $desc       = $Result.explanation
    $conf       = $Result.confidence
    $risk       = $Result.risk.score
    $isFlagged  = $Result.security.isFlagged
    $secLabel   = if ($isFlagged) { "FLAGGED (HIGH RISK)" } else { "ALLOW (SAFE)" }

    $secColor   = if ($isFlagged) { "Red" } else { "Green" }
    $riskColor  = switch ($risk) {
        1 { "Green" }
        2 { "Cyan" }
        3 { "Yellow" }
        4 { "DarkYellow" }
        5 { "Red" }
        default { "White" }
    }

    Write-Host ""
    Write-Host " +-------------------------------------------------------------------------+" -ForegroundColor DarkCyan
    Write-Host " |  S1STEM // System 1 Neural Command Arbiter                              |" -ForegroundColor DarkCyan
    Write-Host " +-------------------------------------------------------------------------+" -ForegroundColor DarkCyan
    Write-Host ""
    Write-Host "  > SUGGESTED COMMAND:" -ForegroundColor DarkGray
    Write-Host "    $cmd" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Explanation : " -NoNewline -ForegroundColor DarkGray
    Write-Host "$desc" -ForegroundColor White
    Write-Host "  Confidence  : " -NoNewline -ForegroundColor DarkGray
    Write-Host "$conf%" -NoNewline -ForegroundColor Cyan
    Write-Host " | Security: " -NoNewline -ForegroundColor DarkGray
    Write-Host "[$secLabel]" -NoNewline -ForegroundColor $secColor
    Write-Host " | Risk: " -NoNewline -ForegroundColor DarkGray
    Write-Host "$risk/5" -NoNewline -ForegroundColor $riskColor
    Write-Host " | Latency: " -NoNewline -ForegroundColor DarkGray
    Write-Host "$($Result.latencyMs) ms" -ForegroundColor Green
    Write-Host ""
}

function Prompt-S1stemAction {
    param(
        [string]$Command,
        [bool]$IsFlagged
    )

    if ($IsFlagged) {
        Write-Host "  WARNING: This command modifies system state or is potentially destructive." -ForegroundColor Red
    }

    Write-Host "  Options:" -ForegroundColor DarkGray
    Write-Host "    [E] Execute immediately (Enter)" -ForegroundColor Green
    Write-Host "    [C] Copy to clipboard" -ForegroundColor Cyan
    Write-Host "    [M] Modify on command line" -ForegroundColor Yellow
    Write-Host "    [A] Cancel" -ForegroundColor DarkGray
    Write-Host ""

    $choice = Read-Host "  Choose an action [E/c/m/a]"
    if ([string]::IsNullOrWhiteSpace($choice)) {
        $choice = "E"
    }
    $choice = $choice.Trim().ToUpper()

    switch ($choice) {
        "E" {
            Write-Host "`n  [s1stem] Executing command...`n" -ForegroundColor DarkGreen
            Invoke-Expression $Command
        }
        "C" {
            try {
                Set-Clipboard -Value $Command
                Write-Host "`n  [s1stem] Command successfully copied to clipboard!" -ForegroundColor Green
            } catch {
                Write-Warning "Unable to copy to clipboard: $_"
            }
        }
        "M" {
            Write-Host "`n  [s1stem] Inserted into command line ready for editing:" -ForegroundColor Yellow
            try {
                if (Get-Module -Name "PSReadLine") {
                    [Microsoft.PowerShell.PSConsoleReadLine]::Insert($Command)
                } else {
                    $host.UI.RawUI.FlushInputBuffer()
                    [System.Windows.Forms.SendKeys]::SendWait($Command)
                }
            } catch {
                # Fallback: offer interactive modification
                $edited = Read-Host "  Edit and press Enter to execute"
                if (-not [string]::IsNullOrWhiteSpace($edited)) {
                    Invoke-Expression $edited
                }
            }
        }
        "A" {
            Write-Host "`n  [s1stem] Operation cancelled." -ForegroundColor DarkGray
        }
        default {
            Write-Host "`n  [s1stem] Unrecognized choice. Operation cancelled." -ForegroundColor DarkGray
        }
    }
    Write-Host ""
}

function Show-S1stemHelp {
    Write-Host ""
    Write-Host " S1stem v1.1.0 Showcase // Terminal Neural Assistant (Example Application)" -ForegroundColor Cyan
    Write-Host " =========================================================================" -ForegroundColor DarkGray
    Write-Host " Demonstrates the S1stem System 1 engine for fast command resolution." -ForegroundColor White
    Write-Host ""
    Write-Host " Usage:" -ForegroundColor Yellow
    Write-Host "   s1 <description of command to run>" -ForegroundColor White
    Write-Host ""
    Write-Host " Examples:" -ForegroundColor Yellow
    Write-Host '   s1 command to create a new file in D:\' -ForegroundColor White
    Write-Host '   s1 command to list files in a directory in c:\test' -ForegroundColor White
    Write-Host '   s1 create a new directory and enter inside, directory name new_dir' -ForegroundColor White
    Write-Host '   s1 show processes consuming the most memory' -ForegroundColor White
    Write-Host '   s1 ping google to check network connection' -ForegroundColor White
    Write-Host '   s1 check available disk space on drives' -ForegroundColor White
    Write-Host ""
    Write-Host " Management Commands:" -ForegroundColor Yellow
    Write-Host "   Start-S1stemDaemon    - Starts local neural daemon in background" -ForegroundColor White
    Write-Host "   Stop-S1stemDaemon     - Stops local neural daemon" -ForegroundColor White
    Write-Host "   Uninstall-S1stem      - Removes S1stem from PowerShell profile" -ForegroundColor White
    Write-Host ""
}

function Uninstall-S1stem {
    [CmdletBinding()]
    param()

    Stop-S1stemDaemon

    $profilePath = $PROFILE.CurrentUserCurrentHost
    if (-not $profilePath) {
        $profilePath = $PROFILE
    }

    if (Test-Path $profilePath) {
        $lines = Get-Content $profilePath
        $filtered = $lines | Where-Object { $_ -notmatch 'S1stem\.psm1' -and $_ -notmatch 'Import-Module.*WebLLM.*powershell' }
        Set-Content -Path $profilePath -Value $filtered
        Write-Host " [s1stem] Removed S1stem reference from profile $profilePath." -ForegroundColor Green
    } else {
        Write-Host " [s1stem] No profile file found." -ForegroundColor DarkGray
    }
}

# Export functions
Export-ModuleMember -Function s1, Start-S1stemDaemon, Stop-S1stemDaemon, Uninstall-S1stem
