$ErrorActionPreference = 'Stop'

$baseDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvDir = Join-Path $baseDir '.venv'
$venvPython = Join-Path $venvDir 'Scripts\python.exe'
$requirementsFile = Join-Path $baseDir 'requirements.txt'
$logDir = Join-Path $baseDir 'logs'
$setupLog = Join-Path $logDir 'setup.log'
$setupStateFile = Join-Path $logDir 'setup_state.json'

if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

function Write-SetupLog {
    param([Parameter(Mandatory = $true)][string]$Message)
    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Add-Content -Path $setupLog -Value "[$timestamp] $Message"
}

function Invoke-CheckedCommand {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $FilePath $($Arguments -join ' ')"
    }
}

function Invoke-CheckedExternal {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $FilePath $($Arguments -join ' ')"
    }
}

function Resolve-PythonBootstrap {
    # Prefer Python 3.11 explicitly — 3.12+ has PyTorch/transformers compatibility issues.
    $py311Path = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311\python.exe'
    if (Test-Path $py311Path) {
        Write-SetupLog "Using Python 3.11 directly at: $py311Path"
        return @{ FilePath = $py311Path; Arguments = @() }
    }

    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        # Try py -3.11 first, fall back to py -3
        $ver = & $py.Source -3.11 --version 2>$null
        if ($LASTEXITCODE -eq 0 -and $ver -match '^Python 3\.11') {
            Write-SetupLog "Using Python launcher (3.11) at: $($py.Source)"
            return @{ FilePath = $py.Source; Arguments = @('-3.11') }
        }
        Write-SetupLog "Using Python launcher (generic) at: $($py.Source)"
        return @{ FilePath = $py.Source; Arguments = @('-3') }
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        Write-SetupLog "Using Python executable at: $($python.Source)"
        return @{ FilePath = $python.Source; Arguments = @() }
    }

    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
        Write-Host 'Python 3.11 not found. Installing Python 3.11 with winget ...' -ForegroundColor Yellow
        Write-SetupLog 'Python not found. Attempting winget install: Python.Python.3.11'
        & $winget.Source install -e --id Python.Python.3.11 --accept-source-agreements --accept-package-agreements
        if ($LASTEXITCODE -eq 0) {
            if (Test-Path $py311Path) {
                Write-SetupLog "Python 3.11 available after install: $py311Path"
                return @{ FilePath = $py311Path; Arguments = @() }
            }

            $py = Get-Command py -ErrorAction SilentlyContinue
            if ($py) {
                Write-SetupLog "Python launcher available after install: $($py.Source)"
                return @{ FilePath = $py.Source; Arguments = @('-3.11') }
            }

            $python = Get-Command python -ErrorAction SilentlyContinue
            if ($python) {
                Write-SetupLog "Python executable available after install: $($python.Source)"
                return @{ FilePath = $python.Source; Arguments = @() }
            }
        }
    }

    throw 'Python 3 is not installed. Install Python 3.10+ and rerun setup_windows.ps1.'
}

function Resolve-OllamaPath {
    $ollama = Get-Command ollama -ErrorAction SilentlyContinue
    if ($ollama) {
        return $ollama.Source
    }

    $commonPaths = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe'),
        'C:\Program Files\Ollama\ollama.exe'
    )

    foreach ($path in $commonPaths) {
        if (Test-Path $path) {
            return $path
        }
    }

    return $null
}

function Wait-ForOllama {
    for ($i = 0; $i -lt 15; $i++) {
        try {
            $response = Invoke-WebRequest -Uri 'http://localhost:11434/' -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                return $true
            }
        } catch {
            Start-Sleep -Seconds 1
        }
    }
    return $false
}

function Get-RequiredOllamaModels {
    return @(
        'qwen2.5:7b',
        'nomic-embed-text',
        'qwen2.5vl'
    )
}

function Get-InstalledOllamaModels {
    param([Parameter(Mandatory = $true)][string]$OllamaPath)

    $rawOutput = & $OllamaPath list 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $rawOutput) {
        return @()
    }

    $installedModels = @()
    foreach ($line in $rawOutput | Select-Object -Skip 1) {
        if ($line -match '^([^\s]+)') {
            $installedModels += $matches[1]
        }
    }

    return $installedModels
}

function Invoke-OllamaPullWithRetry {
    param(
        [Parameter(Mandatory = $true)][string]$OllamaPath,
        [Parameter(Mandatory = $true)][string]$Model
    )

    for ($attempt = 1; $attempt -le 2; $attempt++) {
        Write-SetupLog "Pulling model '$Model' (attempt $attempt)"
        & $OllamaPath pull $Model
        if ($LASTEXITCODE -eq 0) {
            return
        }
        Start-Sleep -Seconds 2
    }

    throw "Failed to pull Ollama model '$Model'."
}

function Save-SetupState {
    param(
        [Parameter(Mandatory = $true)][string]$PythonPath,
        [string]$OllamaPath,
        [string[]]$Models
    )

    $state = [ordered]@{
        completedAt = (Get-Date).ToString('o')
        pythonPath = $PythonPath
        ollamaPath = $OllamaPath
        models = @($Models)
    }

    $state | ConvertTo-Json -Depth 3 | Set-Content -Path $setupStateFile -Encoding ASCII
    Write-SetupLog "Setup state written to $setupStateFile"
}

Write-Host '==================================================' -ForegroundColor Cyan
Write-Host ' TTM Ask Setup' -ForegroundColor Cyan
Write-Host '==================================================' -ForegroundColor Cyan
Write-SetupLog 'Starting setup run.'

$bootstrap = Resolve-PythonBootstrap

if (-not (Test-Path $venvPython)) {
    Write-Host 'Creating virtual environment at .venv ...' -ForegroundColor Yellow
    Invoke-CheckedCommand -FilePath $bootstrap.FilePath -Arguments ($bootstrap.Arguments + @('-m', 'venv', $venvDir))
} else {
    Write-Host 'Using existing virtual environment at .venv' -ForegroundColor Green
}

Write-Host 'Upgrading pip ...' -ForegroundColor Yellow
Invoke-CheckedCommand -FilePath $venvPython -Arguments @('-m', 'pip', 'install', '--upgrade', 'pip')

Write-Host 'Installing Python requirements ...' -ForegroundColor Yellow
Invoke-CheckedCommand -FilePath $venvPython -Arguments @('-m', 'pip', 'install', '-r', $requirementsFile)
Write-SetupLog 'Python requirements installed successfully.'

$ollamaPath = Resolve-OllamaPath
if (-not $ollamaPath) {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
        Write-Host 'Installing Ollama with winget ...' -ForegroundColor Yellow
        & $winget.Source install -e --id Ollama.Ollama --accept-source-agreements --accept-package-agreements
        $ollamaPath = Resolve-OllamaPath
    }
}

if ($ollamaPath) {
    Write-Host 'Ensuring Ollama service is running ...' -ForegroundColor Yellow
    Write-SetupLog "Ollama found at: $ollamaPath"
    if ([string]::IsNullOrWhiteSpace($env:OLLAMA_KEEP_ALIVE)) {
        $env:OLLAMA_KEEP_ALIVE = '-1'
    }
    Write-SetupLog "Using OLLAMA_KEEP_ALIVE=$env:OLLAMA_KEEP_ALIVE"
    Start-Process -FilePath $ollamaPath -ArgumentList 'serve' -WindowStyle Hidden -ErrorAction SilentlyContinue | Out-Null
    if (-not (Wait-ForOllama)) {
        throw 'Ollama service failed to start on localhost:11434.'
    }

    $requiredModels = Get-RequiredOllamaModels
    $installedModels = Get-InstalledOllamaModels -OllamaPath $ollamaPath
    foreach ($model in $requiredModels) {
        if ($installedModels -contains $model) {
            Write-Host "Ollama model already available: $model" -ForegroundColor Green
            Write-SetupLog "Ollama model already available: $model"
            continue
        }

        Write-Host "Pulling Ollama model: $model ..." -ForegroundColor Yellow
        Invoke-OllamaPullWithRetry -OllamaPath $ollamaPath -Model $model
    }
    Write-SetupLog 'Ollama models are ready.'
} else {
    Write-Warning 'Ollama could not be installed automatically. Install it from https://ollama.com/download and then run:'
    Write-Warning '  ollama pull qwen2.5:7b'
    Write-Warning '  ollama pull nomic-embed-text'
    Write-Warning '  ollama pull qwen2.5vl'
}

Save-SetupState -PythonPath $venvPython -OllamaPath $ollamaPath -Models (Get-RequiredOllamaModels)

Write-Host ''
Write-Host 'Setup complete.' -ForegroundColor Green
Write-Host 'First run shows this setup window when dependencies are missing. Later launches run quietly.' -ForegroundColor Green
Write-Host 'To start the app, run Start TTM Ask.vbs, Start Crompton_AI.vbs, or launch.bat' -ForegroundColor Green
Write-SetupLog 'Setup completed successfully.'