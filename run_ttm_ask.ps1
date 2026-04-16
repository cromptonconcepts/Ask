$ErrorActionPreference = 'Stop'

$baseDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$launcherPy = Join-Path $baseDir 'launcher.py'
$appPy = Join-Path $baseDir 'app.py'
$setupScript = Join-Path $baseDir 'setup_windows.ps1'
$logDir = Join-Path $baseDir 'logs'
$runLog = Join-Path $logDir 'run.log'
$setupStateFile = Join-Path $logDir 'setup_state.json'
$launcherOutLog = Join-Path $logDir 'launcher_stdout.log'
$launcherErrLog = Join-Path $logDir 'launcher_stderr.log'
$backendOutLog = Join-Path $logDir 'backend_stdout.log'
$backendErrLog = Join-Path $logDir 'backend_stderr.log'

# Runtime LLM key defaults (kept in local environment scope)
$env:CLOUD_OLLAMA_BASE_URL = 'https://ollama.com/50e6696a54e54b2696adf8d2d5a362d8.MjUpxeBZniOf8zep49P5wdWW'
$env:PAID_GEMINI_API_KEY = 'AIzaSyBy_EemK4Ya8HEYmZS7MtQMn6dpBIvbrFg'
$env:FREE_GEMINI_API_KEY = 'AIzaSyDkM9kW0JZ5ns5yVqm4MPPKH-ZWV146ArU'

if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

function Write-RunLog {
    param([Parameter(Mandatory = $true)][string]$Message)
    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Add-Content -Path $runLog -Value "[$timestamp] $Message"
}

function Get-AppPort {
    $configuredPort = [string]$env:TTM_ASK_PORT
    if ([string]::IsNullOrWhiteSpace($configuredPort)) {
        return 5000
    }

    $parsedPort = 0
    if ([int]::TryParse($configuredPort, [ref]$parsedPort) -and $parsedPort -gt 0 -and $parsedPort -le 65535) {
        return $parsedPort
    }

    Write-RunLog "Invalid TTM_ASK_PORT '$configuredPort'. Falling back to 5000."
    return 5000
}

function Resolve-VenvPython {
    $candidates = @(
        (Join-Path $baseDir '.venv\Scripts\python.exe'),
        (Join-Path $baseDir '.venv-1\Scripts\python.exe'),
        (Join-Path $baseDir 'venv\Scripts\python.exe')
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    return $null
}

function Ensure-SetupStateFile {
    param([Parameter(Mandatory = $true)][string]$PythonPath)

    if (Test-Path $setupStateFile) {
        return
    }

    $state = [ordered]@{
        completedAt = (Get-Date).ToString('o')
        pythonPath = $PythonPath
        repairedBy = 'run_ttm_ask.ps1'
    }
    $state | ConvertTo-Json -Depth 3 | Set-Content -Path $setupStateFile -Encoding ASCII
    Write-RunLog "Backfilled missing setup state at $setupStateFile"
}

function Test-Url {
    param([Parameter(Mandatory = $true)][string]$Url)
    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
        return $response.StatusCode -ge 200 -and $response.StatusCode -lt 500
    } catch {
        return $false
    }
}

function Wait-ForLauncher {
    for ($i = 0; $i -lt 30; $i++) {
        if (Test-Url 'http://127.0.0.1:5001/status') {
            return $true
        }
        Start-Sleep -Milliseconds 800
    }
    return $false
}

function Wait-ForBackend {
    for ($i = 0; $i -lt 120; $i++) {
        if (Test-Url $healthUrl) {
            return $true
        }
        Start-Sleep -Seconds 1
    }
    return $false
}

function Start-LauncherProcess {
    if (Test-Url 'http://127.0.0.1:5001/status') {
        Write-RunLog 'Launcher already online.'
        return $true
    }

    Write-RunLog 'Starting launcher process.'
    $launcherArgs = '-u "{0}"' -f $launcherPy
    Start-Process -FilePath $venvPython -ArgumentList $launcherArgs -WorkingDirectory $baseDir -WindowStyle Hidden -RedirectStandardOutput $launcherOutLog -RedirectStandardError $launcherErrLog | Out-Null
    return (Wait-ForLauncher)
}

function Start-BackendProcess {
    if (Test-Url $healthUrl) {
        Write-RunLog 'Backend already online before direct start.'
        return $true
    }

    Write-RunLog 'Starting backend process directly.'
    $backendArgs = '-u "{0}"' -f $appPy
    Start-Process -FilePath $venvPython -ArgumentList $backendArgs -WorkingDirectory $baseDir -WindowStyle Hidden -RedirectStandardOutput $backendOutLog -RedirectStandardError $backendErrLog | Out-Null
    return (Wait-ForBackend)
}

function Start-BackendIfNeeded {
    if (Test-Url $healthUrl) {
        Write-RunLog 'Backend already online.'
        return 'online'
    }

    $launcherOnline = Start-LauncherProcess
    if (-not $launcherOnline) {
        Write-RunLog 'Launcher failed to come online in time. Falling back to direct backend start.'
        if (Start-BackendProcess) {
            Write-RunLog 'Backend started directly while launcher remained offline.'
            return 'online'
        }
        Write-RunLog 'Direct backend start is still warming up.'
        return 'warming'
    }

    try {
        Invoke-WebRequest -Uri 'http://127.0.0.1:5001/start' -UseBasicParsing -TimeoutSec 5 | Out-Null
        Write-RunLog 'Launcher /start request sent successfully.'
    } catch {
        Write-RunLog 'Launcher /start request failed; attempting direct backend start.'
        if (Start-BackendProcess) {
            Write-RunLog 'Backend started directly after launcher /start failed.'
            return 'online'
        }
    }

    if (Wait-ForBackend) {
        Write-RunLog 'Backend is online.'
        return 'online'
    }

    if (Start-BackendProcess) {
        Write-RunLog 'Backend came online after direct fallback start.'
        return 'online'
    }

    Write-RunLog 'Backend did not come online yet; continuing while it warms up.'
    return 'warming'
}

$appPort = Get-AppPort
$appUrl = "http://127.0.0.1:$appPort/"
$healthUrl = "http://127.0.0.1:$appPort/health"

$venvPython = Resolve-VenvPython
if (-not $venvPython) {
    Write-Host 'No virtual environment found. Running first-time setup ...' -ForegroundColor Yellow
    Write-RunLog 'No virtual environment found. Running setup.'
    & powershell -ExecutionPolicy Bypass -File $setupScript
    if ($LASTEXITCODE -ne 0) {
        throw 'Initial setup failed.'
    }
    $venvPython = Resolve-VenvPython
}

Ensure-SetupStateFile -PythonPath $venvPython

$backendStartupState = Start-BackendIfNeeded
if ($backendStartupState -eq 'failed') {
    Write-Warning 'Backend did not start. Running setup repair once and retrying.'
    Write-RunLog 'Backend offline after first attempt. Running setup repair.'
    & powershell -ExecutionPolicy Bypass -File $setupScript
    if ($LASTEXITCODE -eq 0) {
        $venvPython = Resolve-VenvPython
        $backendStartupState = Start-BackendIfNeeded
        if ($backendStartupState -eq 'failed') {
            Write-Warning 'Backend is still offline. Check logs\\setup.log and logs\\run.log.'
            Write-RunLog 'Backend still offline after repair.'
        }
    } else {
        Write-Warning 'Setup repair failed. Check logs\\setup.log for details.'
        Write-RunLog 'Setup repair failed.'
    }
}

if ($backendStartupState -eq 'online') {
    Start-Process -FilePath $appUrl | Out-Null
    Write-RunLog "Local app launched at $appUrl"
} else {
    Start-Process -FilePath $appUrl | Out-Null
    Write-RunLog "Launched $appUrl while backend continues to warm up."
}
