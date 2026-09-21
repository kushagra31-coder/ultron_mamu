<#
.SYNOPSIS
    Register Windows Task Scheduler jobs for Ultron's proactive briefings.

.DESCRIPTION
    Creates two scheduled tasks (no admin rights needed for per-user tasks):
      - "Ultron Morning Briefing" — daily at 8:00 AM, speaks the morning briefing.
      - "Ultron Reminder Nudge"  — every 30 minutes 8 AM–8 PM, announces
        reminders due within the next 30 minutes (stays silent otherwise).

    Edit ~/.ultron/reminders.json to manage reminders:
        [{"text": "Standup meeting", "time": "10:00"}]

.PARAMETER RepoDir
    Path to the ultron_mamu checkout. Defaults to this script's directory.

.PARAMETER PythonExe
    Python executable to run briefings.py. Defaults to "python" on PATH.
    Use "pythonw" to avoid a console window popping up.
#>
param(
    [string]$RepoDir = $PSScriptRoot,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

function Register-UltronTask {
    param([string]$Name, [string]$Arguments, $Trigger)

    $action = New-ScheduledTaskAction -Execute $PythonExe `
        -Argument "`"$RepoDir\briefings.py`" $Arguments" `
        -WorkingDirectory $RepoDir
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries -StartWhenAvailable

    $existing = Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue
    if ($existing) {
        Set-ScheduledTask -TaskName $Name -Action $action -Trigger $Trigger -Settings $settings | Out-Null
        Write-Host "Updated scheduled task: $Name"
    } else {
        Register-ScheduledTask -TaskName $Name -Action $action -Trigger $Trigger `
            -Settings $settings -Description "Ultron proactive briefing" | Out-Null
        Write-Host "Registered scheduled task: $Name"
    }
}

# Morning briefing — daily 8:00 AM, spoken aloud.
$morningTrigger = New-ScheduledTaskTrigger -Daily -At 8:00AM
Register-UltronTask -Name "Ultron Morning Briefing" `
    -Arguments "morning --speak" -Trigger $morningTrigger

# Reminder nudge — every 30 minutes between 8 AM and 8 PM.
$nudgeTrigger = New-ScheduledTaskTrigger -Daily -At 8:00AM
$nudgeTrigger.Repetition = $(New-ScheduledTaskTrigger -Once -At 8:00AM `
    -RepetitionInterval (New-TimeSpan -Minutes 30) `
    -RepetitionDuration (New-TimeSpan -Hours 12)).Repetition
Register-UltronTask -Name "Ultron Reminder Nudge" `
    -Arguments "nudge --speak" -Trigger $nudgeTrigger

Write-Host ""
Write-Host "Done. Manage reminders in $HOME\.ultron\reminders.json"
Write-Host 'Example: [{"text": "Standup meeting", "time": "10:00"}]'
