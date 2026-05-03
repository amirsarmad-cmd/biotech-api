# Install the SA scraper as a Windows Scheduled Task that runs every 6h.
# Run this once from PowerShell as the same user that has Chrome logged
# into Seeking Alpha. No admin privileges needed.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File install_sa_scraper_task.ps1

$ErrorActionPreference = "Stop"

$TaskName  = "BiotechSAScraper"
$Python    = (Get-Command python).Source
$ScriptDir = Split-Path -Parent $PSCommandPath
$Script    = Join-Path $ScriptDir "home_pc_sa_scraper.py"
$LogFile   = Join-Path $env:USERPROFILE ".biotech-news-scraper\last-run.log"

if (-not (Test-Path $Script)) {
    Write-Error "Could not find $Script - run this script from biotech-api/scripts/"
    exit 1
}

$ConfigDir = Join-Path $env:USERPROFILE ".biotech-news-scraper"
foreach ($f in @("sa-cookies.json", "ingest-token.txt")) {
    $p = Join-Path $ConfigDir $f
    if (-not (Test-Path $p)) {
        Write-Error "Missing config file: $p"
        exit 1
    }
}

# Action: cmd.exe wrapper because >> redirection is a shell feature, not
# something python.exe parses. Without cmd, Python sees ">>" and "log_path"
# as argv and argparse rejects them with exit 2.
$cmdLine = "/c `"`"$Python`" -u `"$Script`" >> `"$LogFile`" 2>&1`""
$Action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument $cmdLine

# Trigger every 6 hours starting 5 minutes from now.
# Use a long but finite RepetitionDuration; Task Scheduler rejects
# [TimeSpan]::MaxValue (P99999999...). 10 years = 3650 days is plenty.
$Start  = (Get-Date).AddMinutes(5)
$Trigger = New-ScheduledTaskTrigger -Once -At $Start `
    -RepetitionInterval (New-TimeSpan -Hours 6) `
    -RepetitionDuration (New-TimeSpan -Days 3650)

# Settings: don't run if on battery, allow start if missed, run as current user
$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopOnIdleEnd `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1) `
    -MultipleInstances IgnoreNew `
    -DisallowDemandStart:$false

# Run as the current interactive user (so cookies file is readable)
$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive

# Replace any existing task with the same name
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask -TaskName $TaskName `
    -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal `
    -Description "Biotech SA scraper - pulls Premium article bodies via residential IP"

Write-Host ""
Write-Host "[OK] Task '$TaskName' registered. First run at: $Start" -ForegroundColor Green
Write-Host "  Log: $LogFile"
Write-Host ""
Write-Host 'Manage via:  Get-ScheduledTask -TaskName BiotechSAScraper | Get-ScheduledTaskInfo'
Write-Host 'Run now:     Start-ScheduledTask -TaskName BiotechSAScraper'
Write-Host 'Remove:      Unregister-ScheduledTask -TaskName BiotechSAScraper -Confirm:$false'
