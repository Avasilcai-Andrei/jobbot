# Registers the "jobbot" scheduled task: runs scripts\jobbot-autostart.ps1 hidden at your logon, and restarts the
# wrapper itself if it ever exits with an error. No administrator rights needed (the task runs as you, only while
# you are logged on, which is also when the GPU and the Ollama tray app are available).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\install-autostart.ps1 [-StartNow]
param([switch]$StartNow)

$ErrorActionPreference = 'Stop'
$taskName = 'jobbot'
$wrapper = Join-Path $PSScriptRoot 'jobbot-autostart.ps1'
$root = Split-Path -Parent $PSScriptRoot
$user = "$env:USERDOMAIN\$env:USERNAME"

$action = New-ScheduledTaskAction -Execute 'powershell.exe' -WorkingDirectory $root `
    -Argument "-NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$wrapper`""
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $user
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew `
    -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1)
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings `
    -Principal $principal -Description 'jobbot job-alert bot: starts Ollama, then python -m jobbot run' -Force |
    Out-Null
Write-Host "Registered scheduled task '$taskName' (runs at logon of $user)."

if ($StartNow) {
    Start-ScheduledTask -TaskName $taskName
    Write-Host "Started it. Logs: $root\data\logs\autostart.log and jobbot.log"
}
