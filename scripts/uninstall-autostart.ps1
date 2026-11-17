# Stops jobbot and removes the "jobbot" scheduled task. Use -KeepTask to only stop it (it starts again at the
# next logon, or with Start-ScheduledTask -TaskName jobbot). Ollama is left running; the tray app owns it.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\uninstall-autostart.ps1 [-KeepTask]
param([switch]$KeepTask)

$taskName = 'jobbot'
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($task) {
    Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
}

# Ending the task stops the wrapper; also stop the Python child in case it outlived the wrapper.
$procs = Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'powershell.exe'" |
    Where-Object { $_.CommandLine -match '-m jobbot .*\brun\b' -or $_.CommandLine -match 'jobbot-autostart\.ps1' }
foreach ($p in $procs) {
    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    Write-Host "Stopped $($p.Name) (pid $($p.ProcessId))"
}

if ($task -and -not $KeepTask) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Host "Removed scheduled task '$taskName'."
} elseif (-not $task) {
    Write-Host "No scheduled task '$taskName' was registered."
} else {
    Write-Host "Stopped; the task stays registered and starts again at the next logon."
}
