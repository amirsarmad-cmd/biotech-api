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
    Write-Error "Could not find $Script — run this script from biotech-api/scripts/"
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

# Action: redirect both stdout and stderr to the log
$argLine = "-u `"$Script`" >> `"$LogFile`" 2>&1"
$Action = New-ScheduledTaskAction -Execute $Python -Argument $argLine

# Trigger every 6 hours starting 5 minutes from now
$Start  = (Get-Date).AddMinutes(5)
$Trigger = New-ScheduledTaskTrigger -Once -At $Start `
    -RepetitionInterval (New-TimeSpan -Hours 6) `
    -RepetitionDuration ([TimeSpan]::MaxValue)

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
    -Description "Biotech SA scraper — pulls Premium article bodies via residential IP"

Write-Host ""
Write-Host "✓ Task '$TaskName' registered. First run at: $Start" -ForegroundColor Green
Write-Host "  Log: $LogFile"
Write-Host ""
Write-Host "Manage via:  Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo"
Write-Host "Run now:     Start-ScheduledTask -TaskName $TaskName"
Write-Host "Remove:      Unregister-ScheduledTask -TaskName $TaskName -Confirm:`$false"
