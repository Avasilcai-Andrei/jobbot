# jobbot autostart wrapper (Windows PowerShell 5.1). Run by the "jobbot" scheduled task at logon.
#  1. makes sure Ollama answers on localhost:11434 (starts the tray app, or `ollama serve` if there is no tray app)
#  2. runs `python -m jobbot run`, logging to data\logs\jobbot.log (rotated by jobbot at 5 MB, 5 old files kept)
#  3. restarts it after a crash: 10 s, then doubling up to 10 min; back to 10 s after a run of 30+ min
# Exit 2 (configuration problem, e.g. no DISCORD_WEBHOOK_URL in .env yet): it checks again every 5 min, so fixing
# .env is enough. It stops for good on 3 (another jobbot run is already running) or 130 (stopped on purpose).
# Its own log: data\logs\autostart.log.

$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
$logDir = Join-Path $root 'data\logs'
$wrapperLog = Join-Path $logDir 'autostart.log'
$stderrLog = Join-Path $logDir 'jobbot-stderr.log'
$ollamaDir = Join-Path $env:LOCALAPPDATA 'Programs\Ollama'
$wrapperStarted = Get-Date
$settleSeconds = 60     # after logon, before this wrapper starts Ollama
$gpuWaitSeconds = 180   # max wait for the NVIDIA driver after that
$ollamaRestarts = 0     # restarts because Ollama came up on the CPU (at most 2 per wrapper)
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Write-Log([string]$message) {
    if ((Test-Path $wrapperLog) -and (Get-Item $wrapperLog).Length -gt 1MB) {
        Move-Item -Force $wrapperLog "$wrapperLog.1"
    }
    "$((Get-Date).ToUniversalTime().ToString('yyyy-MM-dd HH:mm:ss')) UTC  $message" |
        Out-File -Append -Encoding utf8 -FilePath $wrapperLog
}

function Test-Ollama {
    try {
        Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 -Uri 'http://localhost:11434/api/version' | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Test-GpuDriver {
    # $true/$false = whether the NVIDIA driver answers; $null = no nvidia-smi (no NVIDIA GPU to wait for)
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { return $null }
    & nvidia-smi -L *> $null
    return ($LASTEXITCODE -eq 0)
}

function Wait-GpuReady {
    # Ollama started right at logon timed out on GPU discovery and ran on the CPU. Before starting
    # it, wait until logon has settled ($settleSeconds after the wrapper started) and the NVIDIA driver answers.
    $settle = $wrapperStarted.AddSeconds($settleSeconds) - (Get-Date)
    if ($settle.TotalSeconds -gt 0) { Start-Sleep -Seconds ([Math]::Ceiling($settle.TotalSeconds)) }
    $deadline = (Get-Date).AddSeconds($gpuWaitSeconds)
    while ((Test-GpuDriver) -eq $false) {
        if ((Get-Date) -ge $deadline) {
            Write-Log "The NVIDIA driver did not answer within $gpuWaitSeconds s; starting Ollama anyway"
            return
        }
        Start-Sleep -Seconds 5
    }
}

function Get-OllamaCompute([int]$waitSeconds = 30) {
    # 'gpu', 'cpu' or 'unknown', from the tray app's server.log (rewritten at each server start), whose
    # 'msg="inference compute"' lines say what the server found: library=CUDA, or only library=cpu.
    $serverLog = Join-Path $env:LOCALAPPDATA 'Ollama\server.log'
    $server = Get-Process -Name 'ollama' -ErrorAction SilentlyContinue | Sort-Object StartTime | Select-Object -First 1
    if (-not (Get-Process -Name 'ollama app' -ErrorAction SilentlyContinue) -or -not $server) { return 'unknown' }
    $deadline = (Get-Date).AddSeconds($waitSeconds)
    while ($true) {
        if ((Test-Path $serverLog) -and (Get-Item $serverLog).LastWriteTime -ge $server.StartTime.AddSeconds(-5)) {
            $compute = @(Select-String -Path $serverLog -Pattern 'msg="inference compute"' | ForEach-Object { $_.Line })
            if ($compute.Count -gt 0) {
                if (@($compute | Where-Object { $_ -notmatch 'library=cpu' }).Count -gt 0) { return 'gpu' }
                return 'cpu'
            }
        }
        if ((Get-Date) -ge $deadline) { return 'unknown' }
        Start-Sleep -Seconds 2
    }
}

function Restart-OllamaOnCpu {
    if ($script:ollamaRestarts -ge 2 -or (Test-GpuDriver) -ne $true) { return }
    if ((Get-OllamaCompute) -ne 'cpu') { return }
    $script:ollamaRestarts++
    Write-Log 'Ollama found no GPU (it started before the NVIDIA driver was ready) and runs on the CPU; restarting it'
    Get-Process | Where-Object { $_.Path -and $_.Path.StartsWith($ollamaDir, 'OrdinalIgnoreCase') } |
        Stop-Process -Force -ErrorAction SilentlyContinue
    $deadline = (Get-Date).AddSeconds(30)
    while ((Test-Ollama) -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 1 }
    Start-OllamaIfNeeded
}

function Start-OllamaIfNeeded([int]$waitSeconds = 120) {
    if (Test-Ollama) {
        $compute = Get-OllamaCompute
        Write-Log "Ollama is already running (compute: $compute)"
        if ($compute -eq 'cpu') { Restart-OllamaOnCpu }
        return
    }
    Wait-GpuReady
    $trayApp = Join-Path $ollamaDir 'ollama app.exe'
    $cli = Join-Path $ollamaDir 'ollama.exe'
    if (Test-Path $trayApp) {
        if (-not (Get-Process -Name 'ollama app' -ErrorAction SilentlyContinue)) {
            Start-Process -FilePath $trayApp
            Write-Log 'Started the Ollama tray app'
        }
    } elseif (Test-Path $cli) {
        if (-not (Get-Process -Name 'ollama' -ErrorAction SilentlyContinue)) {
            Start-Process -FilePath $cli -ArgumentList 'serve' -WindowStyle Hidden
            Write-Log 'Started ollama serve (no tray app found)'
        }
    } else {
        Write-Log "Ollama is not installed in $ollamaDir; jobbot will send rule matches [unscored]"
        return
    }
    $deadline = (Get-Date).AddSeconds($waitSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-Ollama) {
            $compute = Get-OllamaCompute
            Write-Log "Ollama is up (compute: $compute)"
            if ($compute -eq 'cpu') { Restart-OllamaOnCpu }
            return
        }
        Start-Sleep -Seconds 2
    }
    Write-Log "Ollama did not answer within $waitSeconds s; starting jobbot anyway (it checks again every run)"
}

if (-not (Test-Path $python)) {
    Write-Log "No virtualenv Python at $python; create it first (see README)"
    exit 2
}

Write-Log "Autostart wrapper started (pid $PID)"
$delay = 10
while ($true) {
    Start-OllamaIfNeeded
    $started = Get-Date
    Write-Log 'Starting: python -m jobbot run'
    $proc = Start-Process -FilePath $python -WorkingDirectory $root -NoNewWindow -PassThru `
        -ArgumentList '-m jobbot --log-file data\logs\jobbot.log run' -RedirectStandardError $stderrLog
    $null = $proc.Handle   # keeps the handle so ExitCode is available after the exit
    $proc.WaitForExit()
    $code = $proc.ExitCode
    if ($code -eq 3 -or $code -eq 130) {
        Write-Log "jobbot exited with $code (3 = already running, 130 = stopped); not restarting"
        exit $code
    }
    if ($code -eq 2) {
        # Configuration problem (e.g. no DISCORD_WEBHOOK_URL in .env yet): wait for it to be fixed, then go on.
        Write-Log 'jobbot exited with 2 (configuration problem, see data\logs\jobbot.log); checking again in 5 min'
        Start-Sleep -Seconds 300
        $delay = 10
        continue
    }
    if (((Get-Date) - $started).TotalMinutes -ge 30) { $delay = 10 }
    Write-Log "jobbot exited with code $code; restarting in $delay s (stderr: data\logs\jobbot-stderr.log)"
    Start-Sleep -Seconds $delay
    $delay = [Math]::Min($delay * 2, 600)
}
